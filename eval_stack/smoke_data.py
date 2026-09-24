"""Diagnostic published-row subset only; never a fourth production split."""
import argparse
from pathlib import Path
from .common import digest, read_jsonl, write_json, write_jsonl


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--include-long', action='store_true')
    p.add_argument('--long-lengths', default='4096,8192', help='Diagnostic long-context profiles, comma separated')
    args = p.parse_args()
    source, output = Path(args.data_dir), Path(args.output)
    if output.exists():
        raise ValueError('Use a new diagnostic output directory; do not overwrite prepared splits')
    names = ['gsm8k', 'arc', 'triviaqa', 'bbh_boolean_expressions', 'biggen',
             'hotpot', 'ifeval', 'ifbench', 'multi_if', 'multichallenge', 'structeval', 'humanevalplus']
    if args.include_long:
        names += [f'long_{int(length)}_{task}' for length in args.long_lengths.split(',') for task in ('niah_single_1','qa_1','rag','icl')]
    rows = []
    frozen = source / 'splits' / 'main_test.jsonl'
    if not frozen.is_file():
        raise ValueError('Diagnostic subsets must come from the frozen main_test split')
    by_task = {}
    for row in read_jsonl(frozen):
        by_task.setdefault(row['task'], []).append(row)
    for name in names:
        if name in by_task:
            # Short published prompts for inexpensive plumbing validation, not cherry-picked capability scores.
            row = min(by_task[name], key=lambda r: (sum(len(m['content']) for m in r['messages']), r['id']))
            row['split'] = 'main_test'
            rows.append(row)
    write_jsonl(output / 'splits' / 'main_test.jsonl', rows)
    write_json(output / 'manifest.json', {'complete_inventory': False, 'diagnostic': True,
                                         'origin': str(source), 'tasks': names, 'rows': len(rows),
                                         'splits': {'main_test': {'hash': digest(rows), 'rows':len(rows)}}})


if __name__ == '__main__': main()
