"""Download/adapt published data, reserve held-outs first, and emit audited splits."""
from __future__ import annotations

import collections
import concurrent.futures
import hashlib
import heapq
import json
import os
import random
import re
import inspect
from pathlib import Path

from .common import canonical, digest, env, family, normalized, read_jsonl, validate_record, write_json, write_jsonl
from .sources import SPECS, Rejected, adapt, source_rows
from .admission import rejection_reason


def preflight(row):
    validate_record(row)
    if reason := rejection_reason(row):
        raise Rejected(reason)
    meta = row["verification"]
    if row["verifier"] == "instruction":
        from ifbench import instructions_registry
        for turn in row.get("turns", [meta]):
            for ident, kwargs in zip(turn["ids"], turn["kwargs"]):
                cls = instructions_registry.INSTRUCTION_DICT.get(ident)
                if cls is None:
                    raise Rejected("unsupported_instruction:" + ident)
                checker = cls(ident)
                checker.build_description(**{k: v for k, v in (kwargs or {}).items() if v is not None})
    if row["verifier"] == "rubric":
        if not meta["checks"] or any(c.get("pass_criteria", "YES") not in ("YES", "NO") for c in meta["checks"]):
            raise Rejected("invalid_rubric")
    if row["verifier"] == "structure" and meta.get("schema"):
        import jsonschema
        try:
            jsonschema.Draft202012Validator.check_schema(meta["schema"])
        except jsonschema.SchemaError as e:
            raise Rejected('malformed_published_schema') from e
    if row["verifier"] == "calendar":
        for event in meta["calendar"].values():
            c = event.get("constraint")
            if c and not re.match(r"^(before |after |between |at )", c):
                raise Rejected("unsupported_calendar_constraint")


def cache_source(spec, root):
    path = root / "adapted" / (spec[0] + ".jsonl")
    audit_path = path.with_suffix(".audit.json")
    version = digest([inspect.getsource(preflight), Path(__file__).with_name('sources.py').read_text()])
    if path.exists() and audit_path.exists():
        cached = json.loads(audit_path.read_text())
        if cached.get('adapter_hash') == version:
            return cached
    audit = {"task": spec[0], "requested_source": spec[1], "seen": 0, "eligible": 0, "rejections": {}, "status": "ok", 'adapter_hash': version}
    rejection = collections.Counter()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".partial")
    try:
        with temp.open("w") as out:
            for i, (raw, rev) in enumerate(source_rows(spec, root)):
                audit["seen"] += 1
                try:
                    row = adapt(spec, raw, i, rev)
                    preflight(row)
                    out.write(canonical(row) + "\n")
                    audit["eligible"] += 1
                except (Rejected, ValueError, KeyError, TypeError) as e:
                    rejection[str(e)[:160]] += 1
        temp.replace(path)
    except Exception as e:
        audit["status"], audit["error"] = "source_error", f"{type(e).__name__}: {e}"
        # A partial source is never represented as fully scanned.
    audit["rejections"] = dict(rejection)
    write_json(audit_path, audit)
    print(json.dumps(audit), flush=True)
    return audit


def balanced(rows, n, seed):
    buckets = collections.defaultdict(list)
    for row in rows:
        buckets[row["stratum"]].append(row)
    for key in buckets:
        buckets[key].sort(key=lambda r: digest([seed, r["id"]]))
    selected = []
    keys = sorted(buckets)
    offset = 0
    while len(selected) < n:
        found = False
        for key in keys:
            if offset < len(buckets[key]) and len(selected) < n:
                selected.append(buckets[key][offset])
                found = True
        if not found:
            break
        offset += 1
    return selected


def shingles(text):
    words = normalized(text).split()
    return {int.from_bytes(hashlib.blake2b(' '.join(words[i:i+5]).encode(), digest_size=8).digest(), 'big') for i in range(len(words)-4)}


class HeldoutIndex:
    """Exact family and high-overlap lexical screening; not semantic contamination certification."""
    def __init__(self):
        self.families = set()
        self.inverted = collections.defaultdict(set)
        self.docs = []

    def add(self, row):
        self.families.update(family_keys(row))
        doc = shingles(row["family_text"])
        ident = len(self.docs)
        self.docs.append(doc)
        for token in sorted(doc)[:16]:
            self.inverted[token].add(ident)

    def overlaps(self, row):
        if family_keys(row) & self.families:
            return True
        doc = shingles(row["family_text"])
        if len(doc) < 10:
            return False
        candidates = set()
        for token in sorted(doc)[:16]:
            candidates.update(self.inverted.get(token, ()))
        for i in candidates:
            other = self.docs[i]
            if len(doc & other) / max(1, len(doc | other)) >= .85:
                return True
        return False


def family_keys(row):
    from .common import family
    keys = {row['family_id']}
    # Conversation descendants stay together even when their final user request differs.
    first = next((m['content'] for m in row.get('messages',[]) if m['role'] == 'user'), '')
    if len(normalized(first).split()) >= 10:
        keys.add(family(first))
    keys.update(row.get('family_aliases', []))
    return keys


def build_splits(root, specs=SPECS, tokenizer=None):
    root = Path(root)
    index, own_train, selected = HeldoutIndex(), set(), {"rl_train": [], "rl_val": [], "main_test": []}
    main_pool = []
    report = {"schema_version": 1, "seed": env("SEED", 42, int), "tasks": {},
              "review_status": "automated_provisional_requires_semantic_audit", "errors": [],
              "semantic_refusal_free_certified": False, "test_is_selection_set": True}
    for split, quota_index in (("main_test", 8), ("rl_val", 7), ("rl_train", 6)):
        if split == 'rl_val':
            # Reserve full official evaluation sources, not only the scored subsample.
            for heldout_spec in specs:
                path = root / 'adapted' / (heldout_spec[0] + '.jsonl')
                if heldout_spec[8] and path.exists():
                    for heldout in read_jsonl(path):
                        heldout['family_id'] = family(heldout['family_text'])
                        if not rejection_reason(heldout) and heldout.get('verification',{}).get('task_id') != 'HumanEval/32':
                            main_pool.append(heldout)
                        if heldout['family_id'] not in index.families:
                            index.add(heldout)
        for spec in specs:
            task, desired = spec[0], spec[quota_index]
            if not desired:
                continue
            path = root / "adapted" / (task + ".jsonl")
            report["tasks"].setdefault(task, {})[split] = {"requested": desired, "selected": 0}
            if not path.exists():
                report["errors"].append(f"{task}: source unavailable")
                continue
            rows, seen = [], set()
            pools = collections.defaultdict(list)
            drops = collections.Counter()
            for row in read_jsonl(path):
                row['family_id'] = family(row['family_text'])
                if reason := rejection_reason(row):
                    drops[reason] += 1
                    continue
                if row.get('verifier') == 'humanevalplus' and row.get('verification',{}).get('task_id') == 'HumanEval/32':
                    drops['quarantined_reference_and_native_oracle_failure'] += 1
                    continue
                fid = row["family_id"]
                if fid in seen:
                    drops["duplicate_within_source"] += 1
                    continue
                seen.add(fid)
                rank = -int(digest([report['seed'], row['id']]),16)
                heap = pools[row['stratum']]
                item = (rank, row['id'], row)
                if len(heap) < desired * 3 + 128:
                    heapq.heappush(heap, item)
                elif rank > heap[0][0]:
                    heapq.heapreplace(heap,item)
            candidates = balanced([item[2] for heap in pools.values() for item in heap], desired * 3 + 128, report['seed'])
            for row in candidates:
                if len(rows) >= desired:
                    break
                fid = row['family_id']
                # Same family may have final-test length/measurement variants. Else disjoint.
                if split != "main_test" and index.overlaps(row):
                    drops["heldout_family_or_near_overlap"] += 1
                    continue
                if split == "rl_train" and fid in own_train:
                    drops["cross_train_duplicate"] += 1
                    continue
                if tokenizer is not None and row.get('domain') != 'long_context':
                    tokens = tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=True,
                                                          enable_thinking=False)
                    window = row.get("context_window", env("MODEL_CONTEXT", 8192, int))
                    if len(tokens) + row["max_new_tokens"] > window:
                        drops["reference_context_overflow"] += 1
                        continue
                    row["reference_prompt_tokens"] = len(tokens)
                rows.append(row)
            chosen = balanced(rows, desired, report["seed"])
            for row in chosen:
                row["split"] = split
                if split == "rl_train":
                    own_train.add(row["family_id"])
                else:
                    index.add(row)
                selected[split].append(row)
            report["tasks"][task][split].update(selected=len(chosen), rejections=dict(drops))
            report['tasks'][task][split]['deterministic_candidate_pool'] = len(candidates)
            print(json.dumps({'split':split, 'task':task, 'selected':len(chosen), 'requested':desired}), flush=True)
            if len(chosen) != desired:
                report["errors"].append(f"{task}/{split}: {len(chosen)}/{desired}")
    if len(selected["main_test"]) > min(4000, env("TEST_MAX_ITEMS", 4000, int)):
        raise ValueError("main_test exceeds hard 4,000 cap")
    report["splits"] = {}
    for split, rows in selected.items():
        write_jsonl(root / "splits" / (split + ".jsonl"), rows)
        report["splits"][split] = {"rows": len(rows), "families": len({r["family_id"] for r in rows}),
                                  "domains": dict(collections.Counter(r["domain"] for r in rows)),
                                  "hash": digest(rows)}
    sets = {s: {r["family_id"] for r in rows} for s, rows in selected.items()}
    report["exact_family_overlap"] = {a+"/"+b: len(sets[a] & sets[b]) for a, b in
                                      (("rl_train", "rl_val"), ("rl_train", "main_test"), ("rl_val", "main_test"))}
    report["complete_inventory"] = not report["errors"]
    report['main_test_inventory_complete'] = all(t.get('main_test',{}).get('selected') == t.get('main_test',{}).get('requested') for t in report['tasks'].values())
    report['tokenizer_admission'] = 'reference_tokenizer' if tokenizer is not None else 'NOT_CHECKED'
    report['source_adapter_audits'] = {p.stem: json.loads(p.read_text()) for p in (root/'adapted').glob('*.audit.json')}
    pool_keys = set().union(*(family_keys(r) for r in main_pool)) if main_pool else set()
    val_keys = set().union(*(family_keys(r) for r in selected['rl_val'])) if selected['rl_val'] else set()
    train_keys = set().union(*(family_keys(r) for r in selected['rl_train'])) if selected['rl_train'] else set()
    report['whole_pool_overlap'] = {'main_pool/rl_train':len(pool_keys & train_keys),
                                   'main_pool/rl_val':len(pool_keys & val_keys), 'rl_val/rl_train':len(val_keys & train_keys)}
    if any(report['whole_pool_overlap'].values()):
        raise ValueError('Split-family leakage detected; refusing manifest publication')
    write_jsonl(root/'pools'/'main_test.jsonl', main_pool)
    report['main_pool'] = {'rows':len(main_pool),'tasks':dict(collections.Counter(r['task'] for r in main_pool)),
                           'hash':digest(main_pool), 'purpose':'Reserved source pool; runtime context admission still applies.'}
    write_json(root / "manifest.json", report)
    return report


def prepare(root, only=None):
    root = Path(root)
    from .long_context import specs as long_specs, prepare_long
    all_specs = SPECS + long_specs()
    counts = json.loads(env('TASK_SAMPLE_COUNTS_JSON', '{}'))
    unknown = set(counts) - {s[0] for s in all_specs}
    if unknown:
        raise ValueError('Unknown sample-count task IDs: ' + str(sorted(unknown)))
    configured = [(*s[:8], counts.get(s[0], s[8])) for s in all_specs]
    if any(type(s[8]) is not int or s[8] < 0 for s in configured) or sum(s[8] for s in configured) > 4000:
        raise ValueError('Main-test task counts must be nonnegative integers totalling at most 4000')
    specs = [s for s in SPECS if not only or s[0] in only]
    # Serialize tasks sharing a source to avoid concurrent source-lock mutation.
    groups = collections.defaultdict(list)
    for spec in specs:
        groups[spec[1]].append(spec)
    def group_work(group):
        return [cache_source(s, root) for s in group]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(group_work, groups.values()))
    tokenizer = None
    if os.environ.get("PREP_TOKENIZER"):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(os.environ["PREP_TOKENIZER"], local_files_only=True)
    if env('PREP_LONG_CONTEXT', 1, int):
        prepare_long(root, tokenizer)
    report = build_splits(root, specs=configured, tokenizer=tokenizer)
    print(json.dumps({"splits": report["splits"], "errors": report["errors"]}, indent=2))
    return report


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=env("DATA_DIR", "data"))
    p.add_argument("--tasks", default="")
    args = p.parse_args()
    prepare(args.data_dir, args.tasks.split(",") if args.tasks else None)
