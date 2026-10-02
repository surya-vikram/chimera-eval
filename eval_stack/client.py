"""Bounded stdlib HTTP client. No SDK-shaped extra_body on the wire."""
import json
from contextlib import nullcontext
import threading
import time
import urllib.error
import urllib.request


class EndpointError(RuntimeError):
    pass


class Client:
    def __init__(self, url, model, sampling, concurrency=4, shared=None, timeout=180, retries=2, token_budget=None):
        self.url, self.model, self.sampling = url.rstrip("/"), model, sampling
        self.slot = threading.BoundedSemaphore(concurrency)
        self.shared = shared or threading.BoundedSemaphore(concurrency)
        self.timeout, self.retries = timeout, retries
        self.calls = 0
        self.token_budget = token_budget
        # For the reward service's stats line: judge calls finished since the last take_stats().
        self.stats_lock = threading.Lock()
        self.in_flight = self.retried = self.failed = self.cut_off = 0
        self.latencies, self.output_tokens = [], []

    def request(self, route, payload=None):
        url = (self.url[:-3] if self.url.endswith("/v1") and route == "/tokenize" else self.url) + route
        body = None if payload is None else json.dumps(payload).encode()
        for attempt in range(self.retries + 1):
            try:
                with self.slot, self.shared:
                    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=self.timeout) as response:
                        result = json.load(response)
                    self.calls += 1
                    return result
            except urllib.error.HTTPError as e:
                message = e.read(4096).decode(errors="replace")
                if e.code not in (408, 429, 500, 502, 503, 504):
                    raise EndpointError(f"HTTP {e.code}: {message}") from e
                error = f"HTTP {e.code}: {message}"
            except (OSError, ValueError) as e:
                error = str(e)
            if attempt < self.retries:
                with self.stats_lock:
                    self.retried += 1
                time.sleep(min(2 ** attempt, 8))
        with self.stats_lock:
            self.failed += 1
        raise EndpointError(error)

    def take_stats(self):
        """Judge calls finished since the previous call, and the ones still running."""
        with self.stats_lock:
            stats = {"calls": len(self.latencies), "in_flight": self.in_flight, "latencies": self.latencies,
                     "output_tokens": self.output_tokens, "cut_off": self.cut_off, "retried": self.retried,
                     "failed": self.failed}
            self.latencies, self.output_tokens, self.cut_off, self.retried, self.failed = [], [], 0, 0, 0
        return stats

    def token_count(self, messages):
        payload = {"model": self.model, "messages": messages, "add_generation_prompt": True}
        if "chat_template_kwargs" in self.sampling:
            payload["chat_template_kwargs"] = self.sampling["chat_template_kwargs"]
        result = self.request("/tokenize", payload)
        if 'count' not in result and 'tokens' not in result:
            raise EndpointError('Tokenization response has no count/tokens')
        return result.get("count", len(result.get("tokens", [])))

    def complete(self, messages, max_tokens, seed=None, response_format=None):
        payload = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                   "stream": False, **self.sampling}
        if seed is not None:
            payload["seed"] = seed
        if response_format:
            payload["response_format"] = response_format
        start = time.monotonic()
        with self.stats_lock:
            self.in_flight += 1
        try:
            reservation = (self.token_budget.reserve(self.token_count(messages) + max_tokens, self.timeout)
                           if self.token_budget else nullcontext(None))
            # Tokenization never reserves KV capacity. Hold reservations across
            # transport retries; release before the next turn or judge request.
            with reservation as admission:
                result = self.request("/chat/completions", payload)
        finally:
            with self.stats_lock:
                self.in_flight -= 1
        if not result.get("choices"):
            raise EndpointError("Missing choices")
        choice = result["choices"][0]
        with self.stats_lock:
            self.latencies.append(time.monotonic() - start)
            self.output_tokens.append((result.get("usage") or {}).get("completion_tokens", 0))
            self.cut_off += choice.get("finish_reason") == "length"
        return {"text": choice["message"].get("content") or "",
                "reasoning": choice["message"].get("reasoning_content", choice["message"].get("reasoning")),
                "finish_reason": choice.get("finish_reason"), "usage": result.get("usage", {}),
                "latency": time.monotonic() - start, "request": payload,
                "raw_response": result,
                "kv_admission": admission,
                "server_model": result.get("model"), "response_id": result.get("id")}
