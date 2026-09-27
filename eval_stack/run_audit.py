"""Exhaustive saved-run coverage and token audit. Does not call model endpoints."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import fcntl
import json
import math
from pathlib import Path
import statistics

from .common import digest, read_jsonl, validate_record, write_json, write_jsonl


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def distribution(values):
    """Linear-interpolated percentiles. Missing values are never imputed as zero."""
    good = sorted(v for v in values if numeric(v))
    def percentile(q):
        if not good:
            return None
        i = (len(good) - 1) * q
        lo, hi = math.floor(i), math.ceil(i)
        return good[lo] + (good[hi] - good[lo]) * (i - lo)
    return {'count': len(good), 'missing': len(values) - len(good),
            'total': sum(good), 'min': min(good) if good else None,
            'max': max(good) if good else None, 'mean': statistics.mean(good) if good else None,
            'stddev': statistics.pstdev(good) if good else None,
            **{f'p{p}': percentile(p / 100) for p in (50, 90, 95, 99)}}


def token_distribution(values):
    report = distribution(values)
    edges = [0, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    counts = {f'[{a},{b})': 0 for a, b in zip(edges, edges[1:])}
    counts['[131072,infinity)'] = 0
    for v in values:
        if not numeric(v):
            continue
        for a, b in zip(edges, edges[1:]):
            if a <= v < b:
                counts[f'[{a},{b})'] += 1
                break
        else:
            counts['[131072,infinity)'] += 1
    report['histogram'] = counts
    return report


def selected_rows(rows, c):
    selected = [r for r in rows if r.get('length_bucket', 0) <= c['MODEL_CONTEXT'] and
                (not r.get('length_bucket') or not c['EVAL_CONTEXT_BUCKETS'] or
                 r['length_bucket'] in c['EVAL_CONTEXT_BUCKETS'])]
    if c['TASKS']:
        selected = [r for r in selected if r['task'] in c['TASKS'].split(',')]
    if c['LIMIT_PER_TASK']:
        counts = Counter()
        limited = []
        for row in selected:
            counts[row['task']] += 1
            if counts[row['task']] <= c['LIMIT_PER_TASK']:
                limited.append(row)
        selected = limited
    if c['SPLIT'] == 'main_test' and c['TASK_SAMPLE_COUNTS']:
        limited = []
        for task in sorted({r['task'] for r in selected}):
            group = [r for r in selected if r['task'] == task]
            want = c['TASK_SAMPLE_COUNTS'].get(task, len(group))
            if c['LIMIT_PER_TASK']:
                want = min(want, c['LIMIT_PER_TASK'])
            limited.extend(group[:want])
        selected = limited
    return selected


def compact_record(record, row):
    """Keep audit metadata, excluding prompts, responses, gold, and raw HTTP bodies."""
    if record.get('task') != row['task'] or record.get('domain') != row['domain']:
        raise ValueError('Sample task/domain differs from frozen row')
    grade = record.get('grade', {})
    if grade.get('status') == 'valid':
        if not numeric(grade.get('score')) or grade['score'] > 1:
            raise ValueError('Invalid normalized grade')
        if row.get('binary', True) and type(grade.get('passed')) is not bool:
            raise ValueError('Binary grade missing boolean passed')
    elif grade and grade.get('status') != 'error':
        raise ValueError('Unknown grade status')
    turns = []
    for t in record.get('turns', []):
        r = t['response']
        usage = r.get('usage') or {}
        details = usage.get('completion_tokens_details') or {}
        turns.append({'finish_reason': r.get('finish_reason'),
                      'prompt_tokens': usage.get('prompt_tokens'),
                      'completion_tokens': usage.get('completion_tokens'),
                      'reasoning_tokens': details.get('reasoning_tokens'),
                      'latency_seconds': r.get('latency'), 'grading_seconds': t.get('grading_seconds'),
                      'empty_text': not (r.get('text') or '').strip(),
                      'output_cap': (r.get('request') or {}).get('max_tokens'),
                      'kv_admission': r.get('kv_admission')})
    return {'id': record['id'], 'sample': record['sample'], 'task': row['task'], 'domain': row['domain'],
            'binary': row.get('binary', True), 'grade': grade, 'turns': turns}


def summarize(rows, records, n):
    turns = [t for r in records for t in r['turns']]
    valid = [r for r in records if r['grade'].get('status') == 'valid']
    failed = [r for r in valid if r['binary'] and r['grade'].get('passed') is False]
    passed = [r for r in valid if r['binary'] and r['grade'].get('passed') is True]
    errors = [r for r in records if r['grade'].get('status') == 'error']
    truncated = [r for r in records if any(t['finish_reason'] == 'length' for t in r['turns'])]
    by_id = defaultdict(list)
    for r in valid:
        by_id[r['id']].append(r)
    complete = [row for row in rows if len(by_id[row['id']]) == n]
    generated = sum(bool(r['turns']) for r in records)
    trajectory = {}
    for key in ('prompt_tokens', 'completion_tokens'):
        # Include partial trajectories, but omit their sum if even one turn lacks usage.
        values = [sum(t[key] for t in r['turns']) if all(numeric(t[key]) for t in r['turns']) else None
                  for r in records if r['turns']]
        trajectory[key] = token_distribution(values)
    return {'expected_prompts': len(rows), 'expected_samples': len(rows) * n,
            'frozen_reference_prompt_tokens': token_distribution([r.get('reference_prompt_tokens') for r in rows]),
            'saved_samples': len(records), 'missing_samples': len(rows) * n - len(records),
            'generated_samples': generated, 'valid_samples': len(valid),
            'pending_saved_samples': len(records) - len(valid) - len(errors),
            'grading_error_samples': len(errors), 'complete_prompts': len(complete),
            'incomplete_prompts': len(rows) - len(complete),
            'binary_passed_samples': len(passed), 'binary_failed_samples': len(failed),
            'continuous_scored_samples': sum(not r['binary'] for r in valid),
            'prompts_all_binary_samples_failed': sum(row.get('binary', True) and
                all(r['grade']['passed'] is False for r in by_id[row['id']]) for row in complete),
            'observed_binary_sample_pass_rate': len(passed) / (len(passed) + len(failed)) if passed or failed else None,
            'truncated_samples': len(truncated),
            'truncated_sample_rate_among_generated': len(truncated) / generated if generated else None,
            'generated_turns': len(turns), 'truncated_turns': sum(t['finish_reason'] == 'length' for t in turns),
            'empty_answer_turns': sum(t['empty_text'] for t in turns),
            'finish_reasons': dict(Counter(str(t['finish_reason']) for t in turns)),
            'error_messages': dict(Counter(r['grade'].get('error', 'unspecified') for r in errors)),
            'per_turn': {**{key: token_distribution([t[key] for t in turns]) for key in
                           ('prompt_tokens', 'completion_tokens', 'reasoning_tokens')},
                         **{key: distribution([t[key] for t in turns]) for key in ('latency_seconds', 'grading_seconds')},
                         'output_cap_utilization': distribution([t['completion_tokens'] / t['output_cap']
                             if numeric(t['completion_tokens']) and numeric(t['output_cap']) and t['output_cap'] else None
                             for t in turns])},
            'per_generated_trajectory': trajectory}


def is_running(root):
    if not (root / 'run.lock').exists():
        return False
    with (root / 'run.lock').open('rb') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def audit_run(run_dir, data_dir, output_dir=None, running=None):
    start = datetime.now(timezone.utc).isoformat()
    root, data = Path(run_dir), Path(data_dir)
    saved = json.loads((root / 'config.json').read_text())
    if saved.get('no_new_target_generations'):
        raise ValueError('Audit the original evaluation run; regrade selection uses a different fingerprint')
    c = saved['config']
    rows = [validate_record(r) for r in read_jsonl(data / 'splits' / (c['SPLIT'] + '.jsonl'))]
    expected_hash = saved.get('data_manifest', {}).get('splits', {}).get(c['SPLIT'], {}).get('hash')
    if expected_hash and digest(rows) != expected_hash:
        raise ValueError('Dataset split hash differs from the run manifest')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Dataset contains duplicate row IDs')
    selected = selected_rows(rows, c)
    if len(selected) != saved['selected_rows'] or digest([c, selected, saved['source_hash']]) != saved['fingerprint']:
        raise ValueError('Cannot reproduce the exact frozen run selection/fingerprint')
    rowmap = {r['id']: r for r in selected}
    records, seen, issues = [], set(), []
    active = is_running(root) if running is None else running
    paths = sorted((root / 'samples').glob('*.json'))
    for p in paths:
        try:
            record = json.loads(p.read_text())
            key = (record['id'], record['sample'])
            if key[0] not in rowmap or type(key[1]) is not int or not 0 <= key[1] < c['N_SAMPLES']:
                raise ValueError('Unexpected sample identity/index')
            if key in seen:
                raise ValueError('Duplicate sample identity')
            if p.name != f'{key[0]}.{key[1]}.json':
                raise ValueError('Filename does not match sample identity')
            compact = compact_record(record, rowmap[key[0]])
            seen.add(key)
            records.append(compact)
        except (ValueError, KeyError, TypeError, AttributeError, OSError) as e:
            issues.append({'file': p.name, 'error': str(e)})
    n = c['N_SAMPLES']
    overall = summarize(selected, records, n)
    groups = {}
    for key in ('domain', 'task'):
        groups[key + 's'] = {name: summarize([r for r in selected if r[key] == name],
                                             [r for r in records if r[key] == name], n)
                            for name in sorted({r[key] for r in selected})}
    sample_index = {(r['id'], r['sample']): r for r in records}
    catalogue = []
    violations = []
    for row in selected:
        for sample in range(n):
            r = sample_index.get((row['id'], sample))
            grade = r['grade'] if r else {}
            turns = r['turns'] if r else []
            capped = any(t['finish_reason'] == 'length' for t in turns)
            entry = {'id': row['id'], 'sample': sample, 'task': row['task'], 'domain': row['domain'],
                     'status': grade.get('status', 'pending' if r else 'missing'),
                     'score': grade.get('score'), 'passed': grade.get('passed'),
                     'truncated': capped, 'error': grade.get('error'), 'turns': turns}
            catalogue.append(entry)
            if grade.get('status') == 'valid':
                if not turns or (len(turns) != len(row.get('turns', [None])) and not
                                 (capped and turns[-1]['finish_reason'] == 'length' and
                                  len(turns) < len(row.get('turns', [None])))):
                    violations.append({'id': row['id'], 'sample': sample, 'error': 'Valid grade on incomplete trajectory'})
                if capped and (grade.get('score') != 0 or grade.get('passed') is True):
                    violations.append({'id': row['id'], 'sample': sample, 'error': 'Truncation-zero policy violated'})
    judge, judge_issues = [], []
    for p in sorted((root / 'judge_attempts').glob('*.json')):
        try:
            a = json.loads(p.read_text())
            response = a.get('response') or {}
            judge.append({'status': a.get('status'), 'retry': a.get('attempt', 0) > 0,
                          'error': a.get('error'), 'finish_reason': response.get('finish_reason'),
                          'usage': response.get('usage') or {}, 'latency': response.get('latency')})
        except (ValueError, TypeError, AttributeError, OSError) as e:
            judge_issues.append({'file': p.name, 'error': str(e)})
    tasks = groups['tasks']
    from .metrics import aggregate
    score_records = [{**r, 'turns': [{'response': {'finish_reason': t['finish_reason'],
                        'usage': {'prompt_tokens': t['prompt_tokens']}}} for t in r['turns']]}
                     for r in records]
    manifest = saved.get('data_manifest', {})
    complete_scope = (len(selected) == len(rows) and not c['TASKS'] and not c['LIMIT_PER_TASK'] and
                      c['SPLIT'] == 'main_test' and manifest.get('main_test_inventory_complete',
                                                               manifest.get('complete_inventory', False)))
    scores = aggregate(selected, score_records, n, c.get('PASS_K', [n]), complete_scope)
    # Saved metrics may be absent or stale during a run/resume. Compare only
    # after the lock is released, using exactly the samples read in this scan.
    consistency = {'checked': False, 'differences': []}
    if not active and (root / 'metrics.json').exists():
        existing = json.loads((root / 'metrics.json').read_text())
        consistency['checked'] = True
        for key in ('truncated_samples', 'infrastructure_errors', 'incomplete_prompts',
                    'aggregate_score_0_100', 'valid_full_benchmark'):
            if existing.get(key) != scores[key]:
                consistency['differences'].append({'metric': key, 'saved': existing.get(key), 'recomputed': scores[key]})
    report = {'schema_version': 1, 'snapshot_started_at': start,
              'snapshot_finished_at': datetime.now(timezone.utc).isoformat(),
              'running': active, 'run_dir': str(root.resolve()), 'run_fingerprint': saved['fingerprint'],
              'coverage_complete': overall['valid_samples'] == overall['expected_samples'] and not issues and not judge_issues and not violations,
              'scope': {'split': c['SPLIT'], 'frozen_prompts': len(rows), 'selected_prompts': len(selected),
                        'excluded_prompts': len(rows) - len(selected), 'n_samples': n,
                        'covers_entire_frozen_split': len(rows) == len(selected),
                        'split_hash_verified': expected_hash is not None, 'selection_fingerprint_verified': True,
                        'sample_files_scanned': len(paths)},
              'overall': overall, **groups,
              'scores': scores, 'saved_metrics_consistency': consistency,
              'task_summary': {'expected': len(tasks),
                  'complete': sum(v['complete_prompts'] == v['expected_prompts'] for v in tasks.values()),
                  'with_binary_failures': [k for k, v in tasks.items() if v['binary_failed_samples']],
                  'with_grading_errors': [k for k, v in tasks.items() if v['grading_error_samples']],
                  'with_truncation': [k for k, v in tasks.items() if v['truncated_samples']],
                  'with_no_passes_when_complete': [k for k, v in tasks.items() if v['binary_failed_samples'] and
                      not v['binary_passed_samples'] and v['complete_prompts'] == v['expected_prompts']],
                  'incomplete': [k for k, v in tasks.items() if v['incomplete_prompts']]},
              'judge': {'attempts': len(judge), 'failed_attempts': sum(a['status'] != 'valid' for a in judge),
                        'retries': sum(a['retry'] for a in judge),
                        'truncated_attempts': sum(a['finish_reason'] == 'length' for a in judge),
                        'error_messages': dict(Counter(a['error'] for a in judge if a['error'])),
                        **{key: token_distribution([a['usage'].get(key) for a in judge])
                           for key in ('prompt_tokens', 'completion_tokens')},
                        'latency_seconds': distribution([a['latency'] for a in judge])},
              'integrity_issues': issues + judge_issues, 'policy_violations': violations,
              'notes': ['Running snapshots are not atomic: records can complete during the scan.',
                        'Token counts are server-reported usage, not independent retokenization.',
                        'Frozen reference prompt counts cover all selected rows using the preparation tokenizer; these differ from actual per-turn server usage.',
                        'Observed partial pass rates are not final benchmark scores.',
                        'Per-trajectory prompt totals sum every turn, including repeated conversation history.',
                        'Reasoning-token counts may be zero/unavailable depending on server accounting.',
                        'Continuous quality scores have no binary failure/pass@k threshold.']}
    destination = Path(output_dir) if output_dir else root / 'audit'
    write_json(destination / 'report.json', report)
    write_jsonl(destination / 'samples.jsonl', catalogue)
    (destination / 'report.md').write_text(markdown(report))
    return report


def markdown(report):
    o = report['overall']
    state = 'RUNNING SNAPSHOT' if report['running'] else ('COMPLETE COVERAGE' if report['coverage_complete'] else 'INCOMPLETE COVERAGE')
    lines = [f'# Evaluation audit — {state}', '',
             f"Snapshot: {report['snapshot_finished_at']}", '',
             f"Selected {report['scope']['selected_prompts']} / {report['scope']['frozen_prompts']} frozen prompts; "
             f"{o['valid_samples']} / {o['expected_samples']} valid samples.", '',
             f"Binary failures: {o['binary_failed_samples']}; grading errors: {o['grading_error_samples']}; "
             f"missing: {o['missing_samples']}; saved but pending: {o['pending_saved_samples']}.", '',
             f"Truncation: {o['truncated_samples']} samples / {o['truncated_turns']} turns. "
             f"Integrity issues: {len(report['integrity_issues'])}; policy violations: {len(report['policy_violations'])}.", '',
             '## Domain coverage and failures', '',
             '| Domain | Valid / expected | Binary failed | Grading errors | Truncated samples |',
             '|---|---:|---:|---:|---:|']
    for name, d in report['domains'].items():
        lines.append(f"| {name} | {d['valid_samples']} / {d['expected_samples']} | {d['binary_failed_samples']} | {d['grading_error_samples']} | {d['truncated_samples']} |")
    def fmt(v):
        return '—' if v is None else f'{v:,.1f}'
    lines += ['', '## Frozen prompt inventory (preparation tokenizer; all selected prompts)', '',
              '| Domain | Count | Missing | Mean | p50 | p90 | p95 | p99 | Max |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, d in report['domains'].items():
        s = d['frozen_reference_prompt_tokens']
        lines.append('| ' + name + ' | ' + ' | '.join(fmt(s[k]) for k in
                     ('count', 'missing', 'mean', 'p50', 'p90', 'p95', 'p99', 'max')) + ' |')
    for key, title in [('prompt_tokens', 'Prompt'), ('completion_tokens', 'Response')]:
        lines += ['', f'## {title} tokens per generated turn', '',
                  '| Domain | Count | Missing | Total | Mean | Stddev | Min | p50 | p90 | p95 | p99 | Max |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for name, d in report['domains'].items():
            s = d['per_turn'][key]
            lines.append('| ' + name + ' | ' + ' | '.join(fmt(s[k]) for k in
                         ('count', 'missing', 'total', 'mean', 'stddev', 'min', 'p50', 'p90', 'p95', 'p99', 'max')) + ' |')
    lines += ['', '## Task status', '',
              '| Task | Valid / expected | Binary failed | Grading errors | Truncated samples |',
              '|---|---:|---:|---:|---:|---:|']
    for name, d in report['tasks'].items():
        lines.append(f"| {name} | {d['valid_samples']} / {d['expected_samples']} | {d['binary_failed_samples']} | {d['grading_error_samples']} | {d['truncated_samples']} |")
    lines += ['', '## Interpretation', ''] + ['- ' + x for x in report['notes']]
    lines += ['', 'Full distributions, histogram buckets, judge usage, and task summaries: `report.json`.',
              'Every expected sample, including missing ones: `samples.jsonl`.', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    result = audit_run(args.run_dir, args.data_dir, args.output_dir)
    print(json.dumps({'coverage_complete': result['coverage_complete'], 'running': result['running'],
                      'overall': {k: v for k, v in result['overall'].items() if not isinstance(v, dict)}}, indent=2))
