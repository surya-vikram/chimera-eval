"""Private-network, bounded scoring service reusing the evaluation graders.

No model hosting and no caller-supplied gold/rubrics. Run one instance per cache
directory. Only immutable rl_train/rl_val records are available to this service.
"""
import argparse
import fcntl
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
from pathlib import Path
from socketserver import ThreadingMixIn
import threading

from .client import Client
from .common import digest, read_jsonl, validate_record, write_json
from .graders import Grader
from .runner import settings


class Conflict(ValueError):
    pass


class RewardStore:
    def __init__(self, data_dir, cache_dir, grader, protocol):
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.lock_file = (self.cache / '.service.lock').open('a')
        fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.rows = {}
        self.excluded_rows = {}
        manifest = json.loads((Path(data_dir) / 'manifest.json').read_text())
        for split in ('rl_train', 'rl_val'):
            rows = [validate_record(r) for r in read_jsonl(Path(data_dir) / 'splits' / f'{split}.jsonl')]
            if digest(rows) != manifest['splits'][split]['hash']:
                raise ValueError(f'{split}: manifest mismatch')
            for row in rows:
                key = (split, row['id'])
                if key in self.rows:
                    raise ValueError('Duplicate row identity')
                self.rows[key] = row
                if row['verifier'] == 'choice':
                    # Validate extraction against canonical correct outputs before
                    # any policy receives a reward. Published v2 MCQA contains an
                    # over-unescaped \\boxed regex in part of its training inventory.
                    answer = str(row['verification']['answer'])
                    candidates = [answer, 'Answer: ' + answer, '\\boxed{' + answer + '}']
                    try:
                        valid = any(Grader().grade(row, {'text': text, 'finish_reason': 'stop'})['score'] == 1
                                    for text in candidates)
                    except Exception:
                        valid = False
                    if not valid:
                        self.excluded_rows[row['id']] = 'choice reference-format roundtrip failed'
        source_hash = digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(Path(__file__).parent.glob('*.py'))})
        self.protocol_id = digest({'manifest': manifest, 'protocol': protocol, 'code': source_hash,
                                   'excluded_rows': self.excluded_rows})
        self.grader = grader
        self.guard = threading.Lock()
        self.active = {}
        self.directory = self.cache / self.protocol_id
        self.directory.mkdir(exist_ok=True)
        write_json(self.directory / 'protocol.json', {'id': self.protocol_id, 'config': protocol,
                                                      'manifest': manifest, 'code': source_hash,
                                                      'excluded_rows': self.excluded_rows})

    def close(self):
        self.lock_file.close()

    def grade(self, request):
        if set(request) != {'request_id', 'protocol_id', 'split', 'row_id', 'row_hash', 'response'}:
            raise ValueError('Unexpected/missing request fields')
        if request['protocol_id'] != self.protocol_id:
            raise Conflict('Scoring protocol changed')
        ident = request['request_id']
        if not isinstance(ident, str) or not 1 <= len(ident) <= 512:
            raise ValueError('Invalid request identity')
        row = self.rows.get((request['split'], request['row_id']))
        if row is None or request['row_hash'] != digest(row):
            raise ValueError('Unknown or changed frozen row')
        if row['id'] in self.excluded_rows:
            raise ValueError('Quarantined verifier metadata: ' + self.excluded_rows[row['id']])
        if row.get('turns'):
            raise ValueError('Live multi-turn scoring requires a trajectory adapter; unsupported in v0')
        response = request['response']
        if not isinstance(response, dict) or not isinstance(response.get('text'), str):
            raise ValueError('Response text required')
        if response.get('finish_reason') not in ('stop', 'length'):
            raise ValueError('Infrastructure abort is not a gradeable completion')
        fingerprint = digest(request)
        path = self.directory / (digest(ident) + '.json')
        # Coalesce duplicate requests without serializing unrelated scores.
        while True:
            with self.guard:
                if path.exists():
                    saved = json.loads(path.read_text())
                    if saved['request_hash'] != fingerprint:
                        raise Conflict('Request identity reused with different input')
                    return saved['result']
                if ident not in self.active:
                    self.active[ident] = threading.Event()
                    break
                event = self.active[ident]
            if not event.wait(600):
                raise TimeoutError('Duplicate request still being scored')
        try:
            grade = self.grader.grade(row, response)
            score = grade.get('score')
            if grade.get('status') != 'valid' or type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('Grader did not return a valid normalized score')
            result = {'protocol_id': self.protocol_id, 'request_id': ident, 'grade': grade}
            write_json(path, {'request_hash': fingerprint, 'request': request, 'result': result})
            return result
        finally:
            with self.guard:
                self.active.pop(ident).set()


class RewardServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

    def __init__(self, address, store, workers=8):
        if workers < 1:
            raise ValueError('workers must be positive')
        self.slots = threading.BoundedSemaphore(workers)
        self.store = store
        super().__init__(address, Handler)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            request.sendall(b'HTTP/1.0 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n')
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def log_message(self, *args):
        pass  # Request bodies/golds never go into HTTP access logs.

    def reply(self, status, value):
        body = json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/admission':
            self.reply(200, {'protocol_id': self.server.store.protocol_id,
                             'excluded_rows': self.server.store.excluded_rows})
            return
        if self.path != '/health':
            self.reply(404, {'error': 'Unknown route'})
            return
        self.reply(200, {'protocol_id': self.server.store.protocol_id, 'status': 'ready'})

    def do_POST(self):
        if self.path != '/score':
            self.reply(404, {'error': 'Unknown route'})
            return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 4 * 1024 * 1024:
                raise ValueError('Invalid request size')
            payload = json.loads(self.rfile.read(size))
            self.reply(200, self.server.store.grade(payload))
        except Conflict as exc:
            self.reply(409, {'error': str(exc)})
        except (ValueError, KeyError, TypeError) as exc:
            self.reply(400, {'error': str(exc)})
        except Exception as exc:
            # A scoring fault is never a valid score of zero.
            self.reply(503, {'error': type(exc).__name__ + ': ' + str(exc)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8020)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--judge-revision', required=True, help='Immutable served weights/precision identity')
    args = parser.parse_args()
    c = settings()
    from .token_budget import TokenBudget
    capacity = c['JUDGE_KV_CACHE_NUM_TOKENS']
    judge = Client(c['JUDGE_URL'], c['JUDGE_NAME'], c['JUDGE_SAMPLING'],
                   c['MAX_PENDING'] if capacity else c['JUDGE_CONCURRENCY'],
                   timeout=c['REQUEST_TIMEOUT'], retries=c['REQUEST_RETRIES'],
                   token_budget=TokenBudget(capacity) if capacity else None)
    if c['JUDGE_NAME'] not in [m['id'] for m in judge.request('/models')['data']]:
        raise ValueError('Configured judge model is not served')
    judge_keys = {k: v for k, v in c.items() if k.startswith('JUDGE_') or k.startswith('CODE_')
                  or k.startswith('REQUEST_')}
    judge_keys['revision'] = args.judge_revision
    grader = Grader(judge, c['JUDGE_MAX_TOKENS'], c['JUDGE_CONTEXT'], c['CODE_IMAGE'],
                    c['CODE_TIMEOUT'], c['CODE_CONCURRENCY'],
                    judge_audit_dir=Path(args.cache_dir) / 'judge_attempts',
                    judge_attempts=c['JUDGE_ATTEMPTS'], judge_max_retry_tokens=c['JUDGE_MAX_RETRY_TOKENS'])
    store = RewardStore(args.data_dir, args.cache_dir, grader, judge_keys)
    server = RewardServer((args.host, args.port), store, args.workers)
    print(json.dumps({'address': server.server_address, 'protocol_id': store.protocol_id}), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        store.close()


if __name__ == '__main__':
    main()
