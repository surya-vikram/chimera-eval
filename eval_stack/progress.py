"""One progress line with status counts, printed at most every PROGRESS_SECONDS, instead of a line per sample."""
from collections import Counter
import time

from .common import env

OUTCOMES = ('passed', 'failed', 'scored', 'truncated', 'error')
MAX_ERROR_LINES = 10


def outcome(record):
    """passed/failed for binary grades, scored for continuous ones; truncated and error apart."""
    grade = record.get('grade', {})
    if grade.get('status') != 'valid':
        return 'error'
    if any(t.get('response', {}).get('finish_reason') == 'length' for t in record.get('turns', [])):
        return 'truncated'
    return {True: 'passed', False: 'failed'}.get(grade.get('passed'), 'scored')


def clock(seconds):
    seconds = int(seconds)
    return f'{seconds // 3600}h{seconds // 60 % 60:02d}m' if seconds >= 3600 else f'{seconds // 60}m{seconds % 60:02d}s'


class Progress:
    def __init__(self, total, done=(), interval=None):
        self.total, self.counts, self.errors = total, Counter(), Counter()
        self.interval = env('PROGRESS_SECONDS', 30, float) if interval is None else interval
        for record in done:
            self.counts[outcome(record)] += 1
        self.resumed = sum(self.counts.values())
        self.start = self.last = time.monotonic()
        print(f'Evaluating {total} samples ({self.resumed} already graded, {total - self.resumed} to run); '
              f'progress every {self.interval:g}s.', flush=True)

    def update(self, record):
        kind = outcome(record)
        self.counts[kind] += 1
        if kind == 'error':
            # Each distinct error is shown once (with the first task it hit), so a broken
            # dependency is visible without flooding.
            message = str(record['grade'].get('error', 'unspecified'))[:300]
            self.errors[message] += 1
            if self.errors[message] == 1 and len(self.errors) <= MAX_ERROR_LINES:
                print(f"[error] {record.get('task', '?')}: {message}", flush=True)
                if len(self.errors) == MAX_ERROR_LINES:
                    print('[error] further distinct errors are counted only; see audit/report.json', flush=True)
        self.tick(force=self.done() >= self.total)

    def tick(self, force=False):
        """Print when the interval has passed, also while long requests are still running."""
        now = time.monotonic()
        if force or now - self.last >= self.interval:
            self.last = now
            print(self.line(now), flush=True)

    def done(self):
        return sum(self.counts.values())

    def line(self, now=None):
        now = time.monotonic() if now is None else now
        done, elapsed = self.done(), now - self.start
        fresh = done - self.resumed
        rate = fresh / elapsed if elapsed > 0 else 0
        eta = clock((self.total - done) / rate) if rate and done < self.total else '—'
        width = 20
        filled = width * done // self.total if self.total else width
        counts = ' · '.join(f'{name} {self.counts[name]}' for name in OUTCOMES if self.counts[name] or name in ('passed', 'failed', 'error'))
        return (f"[{'#' * filled}{'.' * (width - filled)}] {done}/{self.total} ({100 * done / max(self.total, 1):.0f}%) | "
                f'{counts} | {rate:.2f}/s | elapsed {clock(elapsed)}' + (f' | eta {eta}' if done < self.total else ''))
