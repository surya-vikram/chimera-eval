"""Versioned repair of v2 MCQA extraction, without re-splitting or changing prompts."""
import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path
import shutil

from .common import digest, read_jsonl, write_json, write_jsonl
from .graders import Grader, extract_final
from .prepare import family_keys

POLICY = 'quality-v3-mcqa-extraction'
BROKEN_BOX = r'\boxed\{\s*([A-Za-z0-9])\s*\}'
CORRECT_BOX = r'\\boxed\{\s*([A-Za-z0-9])\s*\}'
ANSWER = r'Answer\s*:\s*(?!Answer)\s*([A-Za-z0-9])\s*'


def invalid_choice_metadata(row):
    if row.get('task') != 'mcqa':
        return False
    meta = row['verification']
    labels = meta['labels']
    return str(meta['answer']) not in labels or len(set(labels)) != len(labels) or len(labels) < 2


def repair_row(original):
    row = copy.deepcopy(original)
    if row.get('task') != 'mcqa':
        return row, False
    pattern = row['verification'].get('output_regex')
    if pattern not in (BROKEN_BOX, CORRECT_BOX, ANSWER):
        raise ValueError('Unrecognized MCQA metadata; do not guess a repair')
    if pattern != BROKEN_BOX:
        return row, False
    row['verification']['output_regex'] = CORRECT_BOX
    row['id'] = digest([original['id'], POLICY, row['verification']])
    return row, True


def audit_mcqa(row):
    gold = str(row['verification']['answer'])
    labels = row['verification']['labels']
    if gold not in labels or len(set(labels)) != len(labels):
        raise ValueError('Invalid gold or duplicate labels')
    pattern = row['verification']['output_regex']
    if pattern not in (CORRECT_BOX, ANSWER):
        raise ValueError('Broken/unsupported MCQA extraction')
    def render(label):
        return 'Answer: \\boxed{' + label + '}' if pattern == CORRECT_BOX else 'Answer: ' + label
    grader = Grader()
    for label in labels:
        text = render(label)
        score = grader.grade(row, {'text': text, 'finish_reason': 'stop'})['score']
        if score != int(label == gold):
            raise ValueError('Choice extraction/grade roundtrip failed')
    # A stray mention of the right label must not override a final wrong label.
    wrong = next(label for label in labels if label != gold)
    for text in ('Unrelated discussion of ' + gold + '\n' + render(wrong), '', 'oxed{' + gold + '}'):
        if grader.grade(row, {'text': text, 'finish_reason': 'stop'})['score'] != 0:
            raise ValueError('Malformed/wrong candidate received reward')
    if grader.grade(row, {'text': render(gold), 'finish_reason': 'length'})['score'] != 0:
        raise ValueError('Truncation policy changed')


def repair(source, destination):
    source, dest = Path(source).resolve(), Path(destination).resolve()
    if source == dest or dest.exists():
        raise ValueError('Destination must be a new directory; never overwrite frozen data')
    original = json.loads((source / 'manifest.json').read_text())
    manifest = copy.deepcopy(original)
    report = {'policy': POLICY, 'parent_manifest_hash': digest(original), 'changes': [], 'quarantined': [],
              'splits': {}, 'mcqa_rows_audited': 0, 'all_regex_rows_audited': 0}
    keys = {}
    for split in ('rl_train', 'rl_val', 'main_test'):
        rows = list(read_jsonl(source / 'splits' / f'{split}.jsonl'))
        if digest(rows) != original['splits'][split]['hash']:
            raise ValueError('Source split hash mismatch: ' + split)
        revised = []
        changed = 0
        for old in rows:
            if invalid_choice_metadata(old):
                report['quarantined'].append({'split': split, 'id': old['id'],
                                              'reason': 'invalid_gold_or_duplicate_option_labels'})
                continue
            row, modified = repair_row(old)
            if modified:
                changed += 1
                report['changes'].append({'split': split, 'old_id': old['id'], 'new_id': row['id'],
                                           'field': 'verification.output_regex'})
                assert row['messages'] == old['messages'] and row['family_id'] == old['family_id']
            if row['task'] == 'mcqa':
                audit_mcqa(row)
                report['mcqa_rows_audited'] += 1
            pattern = row.get('verification', {}).get('output_regex')
            if pattern:
                # Syntax check every retained regex, not just the repaired task.
                import regex
                regex.compile(pattern)
                report['all_regex_rows_audited'] += 1
            revised.append(row)
        if len({r['id'] for r in revised}) != len(revised):
            raise ValueError('Duplicate revised record ID')
        keys[split] = set().union(*(family_keys(r) for r in revised))
        target = dest / 'splits' / f'{split}.jsonl'
        target.parent.mkdir(parents=True, exist_ok=True)
        if changed or len(revised) != len(rows):
            write_jsonl(target, revised)
        else:
            shutil.copyfile(source / 'splits' / f'{split}.jsonl', target)
        manifest['splits'][split]['hash'] = digest(revised)
        manifest['splits'][split]['rows'] = len(revised)
        manifest['splits'][split]['domains'] = dict(collections.Counter(r['domain'] for r in revised))
        manifest['task_counts'][split] = dict(collections.Counter(r['task'] for r in revised))
        report['splits'][split] = {'rows': len(revised), 'changed': changed, 'bytes': target.stat().st_size,
                                  'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
    overlaps = {a + '/' + b: len(keys[a] & keys[b]) for a, b in
                [('rl_train', 'rl_val'), ('rl_train', 'main_test'), ('rl_val', 'main_test')]}
    pool = source / 'pools' / 'main_test.jsonl'
    if pool.exists():
        pool_keys = set().union(*(family_keys(r) for r in read_jsonl(pool)))
        overlaps.update({s + '/main_pool': len(keys[s] & pool_keys) for s in ('rl_train', 'rl_val')})
        (dest / 'pools').mkdir()
        shutil.copyfile(pool, dest / 'pools' / 'main_test.jsonl')
    if any(overlaps.values()):
        raise ValueError('Cross-split family overlap')
    prior_quarantine = json.loads((source / 'quarantine.json').read_text()) if (source / 'quarantine.json').exists() else []
    write_json(dest / 'quarantine.json', prior_quarantine + report['quarantined'])
    manifest.update(data_policy=POLICY, schema_version=3, source_manifest_hash=digest(original),
                    exact_family_overlap=overlaps, metadata_repair={
                        'policy': POLICY, 'changed_rows': len(report['changes']),
                        'quarantined_rows': len(report['quarantined']),
                        'main_test_unchanged': report['splits']['main_test']['changed'] == 0,
                        'report': 'metadata_repair_report.json'})
    manifest['quarantine_count'] = len(prior_quarantine) + len(report['quarantined'])
    report['exact_family_overlap'] = overlaps
    write_json(dest / 'manifest.json', manifest)
    write_json(dest / 'metadata_repair_report.json', report)
    print(json.dumps({**{k: v for k, v in report.items() if k not in ('changes', 'quarantined')},
                      'quarantined': dict(collections.Counter(r['split'] for r in report['quarantined']))}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--destination', required=True)
    args = parser.parse_args()
    repair(args.source, args.destination)
