import io
import unittest

from eval_stack.service_log import ServiceLog


class Judge:
    def take_stats(self):
        return {'calls': 4, 'in_flight': 7, 'latencies': [1.0, 2.0, 3.0, 9.0], 'output_tokens': [200, 240, 220, 260],
                'cut_off': 1, 'retried': 2, 'failed': 0}


class Budget:
    capacity, used, waiters = 1000, 250, [object(), object()]


class ServiceLogTests(unittest.TestCase):
    def test_stats_line_covers_requests_judge_and_kv_budget(self):
        now = [100.0]
        log = ServiceLog(out=io.StringIO(), clock=lambda: now[0])
        for status in (200, 200, 503):
            log.begin()
            log.end(status, started=now[0] - 2.0, error=None if status == 200 else 'EndpointError: judge down')
        log.busy()
        line = log.stats_line(Judge(), Budget())
        for part in ('scored: 2 ', 'failed: 1', 'busy rejects: 1', 'score p50/p95: 2.0s/2.0s', 'judge calls: 4',
                     'judge in flight: 7', 'judge p50/p95: 2.0s/3.0s', 'judge out tokens: 230', 'cut off: 1',
                     'judge retries: 2', 'KV budget: 25% used, 2 waiting'):
            self.assertIn(part, line)

    def test_a_repeated_failure_is_written_once_a_minute_with_its_count(self):
        now, out = [0.0], io.StringIO()
        log = ServiceLog(out=out, clock=lambda: now[0])
        for _ in range(5):
            log.error(503, 'EndpointError: judge down')
        now[0] = 61.0
        log.error(503, 'EndpointError: judge down')
        lines = out.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn('(and 4 more like it)', lines[1])


if __name__ == '__main__':
    unittest.main()
