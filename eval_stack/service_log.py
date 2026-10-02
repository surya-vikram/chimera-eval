"""Reward-service log: the settings it started with, one stats line a minute while it works, and every
request it could not score. Written to stdout (docker logs) and appended to <cache-dir>/reward_service.log.
Request bodies and gold answers are never logged."""
import collections
import statistics
import sys
import threading
import time


def seconds(values, q):
    if not values:
        return '-'
    ordered = sorted(values)
    return f'{ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))]:.1f}s'


class ServiceLog:
    def __init__(self, path=None, interval=60.0, out=None, clock=time.time):
        self.file = open(path, 'a') if path else None
        self.out, self.interval, self.clock = out or sys.stdout, interval, clock
        self.lock = threading.Lock()
        self.counts, self.latencies = collections.Counter(), []
        self.errors = {}  # (status, message) -> (last written, repeats since)
        self.in_flight = 0

    def write(self, text):
        line = f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.clock()))}] {text}'
        with self.lock:
            print(line, file=self.out, flush=True)
            if self.file:
                self.file.write(line + '\n')
                self.file.flush()

    # ---- request accounting (called from the HTTP handler threads) ------------------------------------
    def begin(self):
        with self.lock:
            self.in_flight += 1

    def end(self, status, started, error=None):
        with self.lock:
            self.in_flight -= 1
            self.counts[status] += 1
            if status == 200:
                self.latencies.append(self.clock() - started)
        if error is not None:
            self.error(status, error)

    def busy(self):
        """Every worker thread was taken; the request got a 503 without being read."""
        with self.lock:
            self.counts['busy'] += 1

    def error(self, status, message):
        """One line per distinct failure, at most once a minute (a judge outage fails every request)."""
        key, now = (status, str(message)[:120]), self.clock()
        with self.lock:
            last, repeats = self.errors.get(key, (None, 0))
            if last is not None and now - last < self.interval:
                self.errors[key] = (last, repeats + 1)
                return
            self.errors[key] = (now, 0)
        more = f' (and {repeats} more like it)' if repeats else ''
        self.write(f'reward error | status: {status} | {str(message)[:400]}{more}')

    # ---- periodic stats ---------------------------------------------------------------------------------
    def stats_line(self, judge=None, budget=None):
        with self.lock:
            counts, latencies, self.counts, self.latencies = self.counts, self.latencies, collections.Counter(), []
            in_flight = self.in_flight
        scored = counts.get(200, 0)
        failed = sum(v for k, v in counts.items() if k not in (200, 'busy'))
        if not (scored or failed or counts.get('busy') or in_flight):
            return None
        parts = ['reward stats', f'scored: {scored} ({scored / self.interval:.1f}/s)', f'in flight: {in_flight}',
                 f'score p50/p95: {seconds(latencies, .5)}/{seconds(latencies, .95)}', f'failed: {failed}',
                 f'busy rejects: {counts.get("busy", 0)}']
        if judge is not None:
            j = judge.take_stats()
            parts += [f'judge calls: {j["calls"]}', f'judge in flight: {j["in_flight"]}',
                      f'judge p50/p95: {seconds(j["latencies"], .5)}/{seconds(j["latencies"], .95)}',
                      f'judge out tokens: {statistics.mean(j["output_tokens"]):.0f}' if j['output_tokens'] else 'judge out tokens: -',
                      f'cut off: {j["cut_off"]}', f'judge retries: {j["retried"]}', f'judge failures: {j["failed"]}']
        if budget is not None:
            parts.append(f'KV budget: {budget.used / max(1, budget.capacity):.0%} used, {len(budget.waiters)} waiting')
        return ' | '.join(parts)

    def run(self, judge=None, budget=None):
        def loop():
            while True:
                time.sleep(self.interval)
                try:
                    line = self.stats_line(judge, budget)
                    if line:
                        self.write(line)
                except Exception as exc:  # the log must never take the service down
                    self.write(f'reward stats unavailable: {type(exc).__name__}: {exc}')
        threading.Thread(target=loop, daemon=True, name='reward-stats').start()
