import concurrent.futures
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

from eval_stack.common import digest, write_json, write_jsonl
from eval_stack.graders import Grader
from eval_stack.reward_service import RewardStore, RewardServer, Conflict


class RewardServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.row = {'id': 'fixture', 'family_id': 'family', 'task': 'mcqa', 'domain': 'knowledge',
                    'messages': [{'role': 'user', 'content': 'Choose A or B.'}],
                    'verifier': 'choice', 'verification': {'answer': 'B', 'labels': ['A', 'B']},
                    'source': 'fixture', 'binary': True}
        rows = [self.row]
        for split in ('rl_train', 'rl_val'):
            write_jsonl(self.root / 'splits' / f'{split}.jsonl', rows)
        write_json(self.root / 'manifest.json', {'splits': {s: {'hash': digest(rows)} for s in ('rl_train', 'rl_val')}})
        self.store = RewardStore(self.root, self.root / 'cache', Grader(), {'fixture': True})
        self.payload = {'request_id': 'run:0:0', 'protocol_id': self.store.protocol_id,
                        'split': 'rl_train', 'row_id': 'fixture', 'row_hash': digest(self.row),
                        'response': {'text': 'B', 'finish_reason': 'stop'}}

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_direct_grader_parity_and_cache(self):
        direct = Grader().grade(self.row, self.payload['response'])
        self.assertEqual(self.store.grade(self.payload)['grade'], direct)
        self.store.grader = None
        self.assertEqual(self.store.grade(self.payload)['grade'], direct)
        different = dict(self.payload, response={'text': 'A', 'finish_reason': 'stop'})
        with self.assertRaises(Conflict):
            self.store.grade(different)

    def test_capped_is_zero_and_abort_is_error(self):
        capped = dict(self.payload, response={'text': 'B', 'finish_reason': 'length'})
        self.assertEqual(self.store.grade(capped)['grade']['score'], 0)
        with self.assertRaises(ValueError):
            self.store.grade(dict(self.payload, response={'text': 'B', 'finish_reason': 'abort'}))

    def test_cannot_supply_gold_or_main_test(self):
        with self.assertRaises(ValueError):
            self.store.grade(dict(self.payload, verification={'answer': 'A'}))
        with self.assertRaises(ValueError):
            self.store.grade(dict(self.payload, split='main_test'))

    def test_duplicates_coalesce_and_independent_work_overlaps(self):
        class Slow:
            calls = 0
            active = 0
            maximum = 0
            lock = threading.Lock()
            def grade(s, row, response):
                with s.lock:
                    s.calls += 1
                    s.active += 1
                    s.maximum = max(s.maximum, s.active)
                time.sleep(.05)
                with s.lock:
                    s.active -= 1
                return {'status': 'valid', 'score': .75, 'passed': None}
        slow = Slow()
        self.store.grader = slow
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            futures = [pool.submit(self.store.grade, self.payload) for _ in range(3)]
            futures.append(pool.submit(self.store.grade, dict(self.payload, request_id='other')))
            results = [f.result() for f in futures]
        self.assertEqual(slow.calls, 2)
        self.assertEqual(slow.maximum, 2)
        self.assertEqual(results[0]['grade']['score'], .75)

    def test_http_roundtrip(self):
        server = RewardServer(('127.0.0.1', 0), self.store, workers=2)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f'http://127.0.0.1:{server.server_port}'
        try:
            with urllib.request.urlopen(url + '/health') as response:
                self.assertEqual(json.load(response)['protocol_id'], self.store.protocol_id)
            req = urllib.request.Request(url + '/score', data=json.dumps(self.payload).encode())
            with urllib.request.urlopen(req) as response:
                self.assertEqual(json.load(response)['grade']['score'], 1.)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
