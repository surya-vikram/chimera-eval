"""Audit published APPS reference solutions inside the existing code sandbox.

Never execute dataset code in the host process. Failures are reported for review,
not silently converted into policy failures or automatic dataset edits.
"""
import argparse
import concurrent.futures
import json
import time
from pathlib import Path

from eval_stack.common import digest, read_jsonl, write_json
from eval_stack.graders import Grader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0, help='0 audits all train/val APPS rows')
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    jobs = [(split, row) for split in ('rl_val', 'rl_train')
            for row in read_jsonl(Path(args.data_dir) / 'splits' / f'{split}.jsonl') if row['task'] == 'apps']
    if args.limit:
        jobs = jobs[:args.limit]
    grader = Grader(code_concurrency=args.workers)

    def run(job):
        split, row = job
        path = output / (row['id'] + '.json')
        if path.exists():
            saved = json.loads(path.read_text())
            if saved['row_hash'] == digest(row):
                return saved
            raise ValueError('Audit input changed')
        result = dict(id=row['id'], split=split, row_hash=digest(row), passed=False, attempts=[])
        start = time.monotonic()
        try:
            for code in row['verification']['reference_solutions']:
                grade = grader.execute_code(row, code)
                result['attempts'].append(grade)
                if grade['passed']:
                    result['passed'] = True
                    break
            if result['passed']:
                # Deliberately wrong candidate must not pass a vacuous test suite.
                result['negative'] = grader.execute_code(row, "raise RuntimeError('negative audit control')")
                if result['negative']['passed']:
                    result['passed'] = False
        except Exception as exc:
            result['error'] = f'{type(exc).__name__}: {exc}'
        result['seconds'] = time.monotonic() - start
        write_json(path, result)
        return result

    results = []
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        for result in pool.map(run, jobs):
            results.append(result)
            if len(results) % 50 == 0:
                print(json.dumps({'completed': len(results), 'total': len(jobs),
                                  'passed': sum(r['passed'] for r in results)}), flush=True)
    summary = {'total': len(results), 'passed': sum(r['passed'] for r in results),
               'failures': [r['id'] for r in results if not r['passed']],
               'scope': 'reference and negative-control sandbox audit; no policy training'}
    write_json(output / 'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
