import concurrent.futures
import os
import io
import threading
import time
import unittest
import urllib.error
from unittest.mock import patch

from eval_stack.client import Client, EndpointError
from eval_stack.runner import settings
from eval_stack.token_budget import TokenBudget, interleave_domains


class TokenBudgetTests(unittest.TestCase):
    def wait_for_queue(self, budget, count):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            with budget.condition:
                if len(budget.waiters) == count:
                    return
            time.sleep(.005)
        self.fail('Waiter did not enter queue')

    def test_mixed_requests_never_exceed_capacity_and_release_on_error(self):
        budget = TokenBudget(100)
        barrier = threading.Barrier(8)
        def work(i):
            barrier.wait(timeout=3)
            try:
                with budget.reserve(20 + i * 10, timeout=3):
                    time.sleep(.01)
                    if i == 3:
                        raise RuntimeError('failed generation')
            except RuntimeError:
                pass
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            list(pool.map(work, range(8)))
        self.assertLessEqual(budget.peak, 100)
        self.assertEqual(budget.used, 0)
        self.assertEqual(budget.waiters, [])

    def test_backfill_then_drain_for_long_request(self):
        budget = TokenBudget(100, max_bypasses=1)
        admitted = []
        release_long = threading.Event()
        def work(name, tokens):
            with budget.reserve(tokens, timeout=3):
                admitted.append(name)
                if name == 'long':
                    release_long.wait(timeout=3)
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            with budget.reserve(60):
                long = pool.submit(work, 'long', 90)
                self.wait_for_queue(budget, 1)
                # A short request fills the remaining space once.
                with budget.reserve(30, timeout=1):
                    self.assertEqual(budget.used, 90)
                short = pool.submit(work, 'short', 20)
                self.wait_for_queue(budget, 2)
                self.assertEqual(admitted, [])
            deadline = time.monotonic() + 3
            while not admitted and time.monotonic() < deadline:
                time.sleep(.005)
            self.assertEqual(admitted, ['long'])
            release_long.set()
            long.result(timeout=3)
            short.result(timeout=3)
        self.assertEqual(admitted, ['long', 'short'])

    def test_oversize_and_timeout_do_not_leak(self):
        budget = TokenBudget(10)
        with self.assertRaises(ValueError):
            with budget.reserve(11):
                pass
        with budget.reserve(10):
            with self.assertRaises(TimeoutError):
                with budget.reserve(1, timeout=.01):
                    pass
            self.assertEqual(budget.waiters, [])
        self.assertEqual(budget.used, 0)

    def test_target_and_judge_share_budget_and_release_on_transport_error(self):
        budget = TokenBudget(100)
        clients = [Client('http://fixture/v1', role, {}, token_budget=budget)
                   for role in ('target', 'judge')]
        entered, release = threading.Event(), threading.Event()
        def request(route, payload=None):
            if route == '/tokenize':
                return {'count': 20}
            entered.set()
            release.wait(timeout=3)
            raise EndpointError('transport failed')
        for client in clients:
            client.request = request
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            first = pool.submit(clients[0].complete, [], 60)
            self.assertTrue(entered.wait(timeout=3))
            second = pool.submit(clients[1].complete, [], 60)
            self.wait_for_queue(budget, 1)
            self.assertEqual(budget.used, 80)
            release.set()
            for future in (first, second):
                with self.assertRaises(EndpointError):
                    future.result(timeout=3)
        self.assertEqual(budget.peak, 80)
        self.assertEqual(budget.used, 0)

    def test_domain_interleave_keeps_every_job_and_within_domain_order(self):
        jobs = [({'domain': domain}, i) for i, domain in enumerate(['math']*3 + ['code']*2 + ['long'])]
        self.assertEqual([j[1] for j in interleave_domains(jobs)], [0, 3, 5, 1, 4, 2])

    def test_independent_role_budgets_allow_judge_while_target_is_full(self):
        target, judge = TokenBudget(100), TokenBudget(40)
        admitted = threading.Event()
        with target.reserve(100):
            def work():
                with judge.reserve(40, timeout=.5):
                    admitted.set()
                    self.assertEqual(target.used + judge.used, 140)
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                pool.submit(work).result(timeout=2)
            self.assertTrue(admitted.is_set())
        self.assertEqual((target.used, judge.used), (0, 0))

    def test_equal_weight_waiters_stress_no_leak_or_oversubscription(self):
        budget = TokenBudget(131072)
        def work(i):
            tokens = [128, 4096, 32768, 131072][i % 4]
            with budget.reserve(tokens, timeout=10):
                time.sleep(.001)
                with budget.condition:
                    self.assertLessEqual(budget.used, budget.capacity)
            return i
        with concurrent.futures.ThreadPoolExecutor(32) as pool:
            self.assertEqual(sorted(pool.map(work, range(256))), list(range(256)))
        self.assertEqual(budget.used, 0)
        self.assertEqual(budget.waiters, [])

    def test_transport_retries_hold_reservation_and_release_on_exhaustion(self):
        budget = TokenBudget(100)
        client = Client('http://fixture/v1', 'target', {}, retries=2, token_budget=budget)
        client.token_count = lambda messages: 20
        def fail(*args, **kwargs):
            self.assertEqual(budget.used, 80)
            raise urllib.error.HTTPError('http://fixture', 503, 'busy', {}, io.BytesIO(b'busy'))
        with patch('urllib.request.urlopen', side_effect=fail) as request, patch('time.sleep'):
            with self.assertRaises(EndpointError):
                client.complete([], 60)
            self.assertEqual(request.call_count, 3)
        self.assertEqual(budget.used, 0)

    def test_oversize_never_sends_completion(self):
        client = Client('http://fixture/v1', 'target', {}, token_budget=TokenBudget(100))
        client.token_count = lambda messages: 50
        with patch.object(client, 'request') as request:
            with self.assertRaisesRegex(ValueError, 'budget is 100'):
                client.complete([], 60)
            request.assert_not_called()

    def test_configuration_defaults_and_invalid_capacity(self):
        for role in ('MODEL', 'JUDGE'):
            key = role + '_KV_CACHE_NUM_TOKENS'
            with patch.dict(os.environ, {key: '100000'}, clear=True):
                self.assertEqual(settings()['MAX_PENDING'], 128)
                self.assertEqual(settings()[key], 100000)
            with patch.dict(os.environ, {key: '-1'}, clear=True):
                with self.assertRaises(ValueError):
                    settings()
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(settings()['MODEL_KV_CACHE_NUM_TOKENS'], 0)
            self.assertEqual(settings()['JUDGE_KV_CACHE_NUM_TOKENS'], 0)
            self.assertEqual(settings()['MAX_PENDING'], 16)
        with patch.dict(os.environ, {'KV_CACHE_NUM_TOKENS': '100000'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'Use MODEL_KV'):
                settings()
