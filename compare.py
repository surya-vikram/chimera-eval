#!/usr/bin/env python3
"""Compare every finished run in an outputs folder: one CSV row per run, best aggregate first.

    python3 compare.py outputs                 # writes outputs/comparison.csv
    python3 compare.py outputs -o models.csv

Columns: run, model_path (from the run's saved launch settings), model and judge names, the
selection and sampling settings, aggregate score and pass@k, then score and pass@k for every
domain. Scores are percentages. A value the run does not have (a domain it did not evaluate,
a k it did not request, pass@k for quality) is left empty. Standard library only.
"""
import argparse
import csv
import json
from pathlib import Path
import shlex
import statistics

DOMAINS = ("math", "knowledge", "logic", "grounding", "quality", "instruction",
           "multiturn", "structure", "python", "long_context")


def launch_settings(run):
    """The values run_eval.sh used (launch_config.env), else [KEY]=value from its saved copy."""
    settings = {}
    env = run / 'launch_config.env'
    if env.exists():
        for line in env.read_text().splitlines():
            if line and not line.startswith('#') and '=' in line:
                key, _, value = line.partition('=')
                settings[key] = ''.join(shlex.split(value))
    elif (run / 'run_eval.sh').exists():
        for line in (run / 'run_eval.sh').read_text().splitlines():
            line = line.strip()
            if line.startswith('[') and ']=' in line:
                key, _, value = line[1:].partition(']=')
                settings[key] = ''.join(shlex.split(value, comments=True))
    return settings


def percent(value):
    return None if value is None else round(100 * value, 2)


def summarize(run):
    metrics = json.loads((run / 'metrics.json').read_text())
    config = json.loads((run / 'config.json').read_text()).get('config', {}) if (run / 'config.json').exists() else {}
    launch = launch_settings(run)
    ks = [str(k) for k in metrics.get('pass_k', [])]
    domains = metrics.get('domains') or {}
    scored = [d for d in domains.values() if d]
    # Metrics written before 2026-09-29 have no always-on aggregate: rebuild it the same way.
    aggregate = metrics.get('aggregate_score_0_100')
    if aggregate is None and scored:
        aggregate = 100 * statistics.mean(d['score'] for d in scored)
    passes = metrics.get('aggregate_pass_at_k') or {
        k: 100 * statistics.mean(v) if v else None
        for k, v in ((k, [d['pass'][k] for d in scored if d.get('pass', {}).get(k) is not None]) for k in ks)}
    sampling = config.get('MODEL_SAMPLING', {})
    row = {'run': run.name,
           'model_path': launch.get('MODEL_PATH', ''),
           'model_name': config.get('MODEL_NAME', launch.get('MODEL_NAME', '')),
           'judge_name': config.get('JUDGE_NAME', launch.get('JUDGE_NAME', '')),
           'split': config.get('SPLIT', ''), 'tasks': config.get('TASKS', '') or 'all',
           'limit_per_task': config.get('LIMIT_PER_TASK', ''), 'n_samples': metrics.get('n_samples', ''),
           'model_context': config.get('MODEL_CONTEXT', ''),
           'temperature': sampling.get('temperature', ''), 'top_p': sampling.get('top_p', ''),
           'top_k': sampling.get('top_k', ''),
           'aggregate_score': None if aggregate is None else round(aggregate, 2),
           'aggregate_complete': metrics.get('aggregate_complete', metrics.get('valid_full_benchmark', '')),
           'truncated_samples': metrics.get('truncated_samples', ''),
           'grading_errors': metrics.get('infrastructure_errors', '')}
    row.update({f'aggregate_pass@{k}': None if passes.get(k) is None else round(passes[k], 2) for k in ks})
    for name in DOMAINS:
        d = domains.get(name) or {}
        row[f'{name}_score'] = percent(d.get('score'))
        row.update({f'{name}_pass@{k}': percent((d.get('pass') or {}).get(k)) for k in ks})
    return row


def columns(rows):
    """Fixed columns first, then pass@k in numeric order of k, domains in protocol order."""
    ks = sorted({key.split('@')[1] for r in rows for key in r if '@' in key}, key=int)
    fixed = [k for k in rows[0] if '@' not in k and not any(k.startswith(d + '_') for d in DOMAINS)]
    return (fixed + [f'aggregate_pass@{k}' for k in ks] +
            [c for d in DOMAINS for c in [f'{d}_score', *(f'{d}_pass@{k}' for k in ks)]])


def compare(outputs, destination=None):
    runs = sorted(p.parent for p in Path(outputs).glob('*/metrics.json') if not p.parent.name.startswith('.'))
    rows = [summarize(run) for run in runs]
    rows.sort(key=lambda r: (r['aggregate_score'] is None, -(r['aggregate_score'] or 0), r['run']))
    destination = Path(destination) if destination else Path(outputs) / 'comparison.csv'
    if rows:
        names = columns(rows)
        with destination.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=names, restval='')
            writer.writeheader()
            writer.writerows({k: '' if v is None else v for k, v in r.items()} for r in rows)
    return rows, destination


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('outputs', help='folder holding one subfolder per run (OUTPUT_PATH)')
    parser.add_argument('-o', '--output', help='CSV path (default: <outputs>/comparison.csv)')
    args = parser.parse_args()
    rows, destination = compare(args.outputs, args.output)
    if not rows:
        raise SystemExit(f'No finished runs (no */metrics.json) under {args.outputs}')
    ks = sorted({key.split('@')[1] for r in rows for key in r if key.startswith('aggregate_pass@')}, key=int)
    header = ['#', 'run', 'model', 'aggregate', *(f'pass@{k}' for k in ks), 'all graded']
    table = [[str(i), r['run'], r['model_path'] or r['model_name'],
              '' if r['aggregate_score'] is None else f"{r['aggregate_score']:.1f}",
              *('' if r.get(f'aggregate_pass@{k}') in (None, '') else f"{r[f'aggregate_pass@{k}']:.1f}" for k in ks),
              'yes' if r['aggregate_complete'] is True else 'no'] for i, r in enumerate(rows, 1)]
    widths = [max(len(x) for x in col) for col in zip(header, *table)]
    for line in [header, *table]:
        print('  '.join(v.ljust(w) if i in (1, 2) else v.rjust(w) for i, (v, w) in enumerate(zip(line, widths))))
    print(f'\n{len(rows)} runs -> {destination}')


if __name__ == '__main__':
    main()
