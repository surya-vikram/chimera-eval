"""Small opt-in live scorer smoke. Uses rl_val only; not a capability benchmark."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import time
import urllib.request

from eval_stack.common import digest, read_jsonl, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://127.0.0.1:8020')
    p.add_argument('--data-dir', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    def request(route, body=None):
        req = urllib.request.Request(args.url + route, data=None if body is None else json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=300) as response:
            return json.load(response)
    protocol = request('/health')['protocol_id']
    excluded = request('/admission')['excluded_rows']
    rows = list(read_jsonl(Path(args.data_dir) / 'splits' / 'rl_val.jsonl'))
    jobs = []
    for task in ('mcqa', 'gsm8k_train', 'openqa'):
        row = min((r for r in rows if r['task'] == task and r['id'] not in excluded), key=lambda r: len(json.dumps(r['messages'])))
        gold = str(row['verification']['answer'])
        boxed = row['verification'].get('output_regex', '').startswith(r'\\boxed')
        answer = 'Answer: ' + gold if task == 'mcqa' and not boxed else '\\boxed{' + gold + '}'
        for variant, text, finish, expected in [('correct', answer, 'stop', 1),
                                                ('wrong', 'Answer: Z' if task == 'mcqa' else '\\boxed{definitely unrelated answer}', 'stop', 0),
                                                ('capped', answer, 'length', 0)]:
            payload = {'protocol_id': protocol, 'request_id': f'live-smoke-v2:{task}:{variant}',
                       'split': 'rl_val', 'row_id': row['id'], 'row_hash': digest(row),
                       'response': {'text': text, 'finish_reason': finish}}
            jobs.append((task, variant, expected, payload))
    def run(job):
        task, variant, expected, payload = job
        start = time.monotonic()
        result = request('/score', payload)
        return {'task': task, 'variant': variant, 'expected': expected, 'result': result,
                'seconds': time.monotonic() - start, 'matches': result['grade']['score'] == expected}
    with concurrent.futures.ThreadPoolExecutor(3) as pool:
        results = list(pool.map(run, jobs))
    # Repeat immutable requests to check durable caching without a second judge call.
    repeated = request('/score', jobs[6][3])
    assert repeated == results[6]['result']
    report = {'protocol': protocol, 'cases': results, 'passed': sum(r['matches'] for r in results),
              'total': len(results), 'repeat_identical': True, 'quarantined_rows': len(excluded),
              'scope': 'integration smoke, not judge qualification'}
    write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'cases'}))
    if report['passed'] != report['total']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
