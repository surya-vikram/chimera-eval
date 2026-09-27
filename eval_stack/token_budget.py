"""Client-side token reservations; not a measurement of server GPU memory."""
from collections import deque
from contextlib import contextmanager
import threading
import time


class TokenBudget:
    def __init__(self, capacity, max_bypasses=8):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('KV token capacity must be a positive integer')
        if type(max_bypasses) is not int or max_bypasses < 0:
            raise ValueError('max_bypasses must be a nonnegative integer')
        self.capacity, self.max_bypasses = capacity, max_bypasses
        self.used = self.peak = 0
        self.waiters = []
        self.condition = threading.Condition()

    def _eligible(self):
        # Backfill with the oldest fitting request, until an older request has
        # been bypassed max_bypasses times. Then drain enough space for it.
        free = self.capacity - self.used
        for item in self.waiters:
            if item['tokens'] <= free:
                return item
            if item['bypasses'] >= self.max_bypasses:
                break
        return None

    @contextmanager
    def reserve(self, tokens, timeout=1800):
        if type(tokens) is not int or not 0 < tokens <= self.capacity:
            raise ValueError(f'Request requires {tokens} KV tokens; budget is {self.capacity}. '
                             'Increase the relevant MODEL_KV_CACHE_NUM_TOKENS or JUDGE_KV_CACHE_NUM_TOKENS; prompts/caps are never shrunk.')
        item = {'tokens': tokens, 'bypasses': 0}
        start = time.monotonic()
        deadline = start + timeout
        with self.condition:
            self.waiters.append(item)
            try:
                while self._eligible() is not item:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError('Timed out waiting for KV token budget')
                    self.condition.wait(remaining)
                for older in self.waiters:
                    if older is item:
                        break
                    older['bypasses'] += 1
                self.waiters = [w for w in self.waiters if w is not item]
                self.used += tokens
                self.peak = max(self.peak, self.used)
                self.condition.notify_all()
            except BaseException:
                self.waiters = [w for w in self.waiters if w is not item]
                self.condition.notify_all()
                raise
        try:
            yield {'reserved_tokens': tokens, 'wait_seconds': time.monotonic() - start}
        finally:
            with self.condition:
                self.used -= tokens
                self.condition.notify_all()


def interleave_domains(jobs):
    """Round robin domains without changing selected rows or sample seeds."""
    groups = {}
    for job in jobs:
        groups.setdefault(job[0]['domain'], deque()).append(job)
    active = deque(groups.values())
    while active:
        group = active.popleft()
        yield group.popleft()
        if group:
            active.append(group)
