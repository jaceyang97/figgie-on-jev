"""Minimal client for TypeSafe AI's Jev decision model, plus an offline stand-in.

Jev takes a JSON "state" and a set of typed questions and returns numeric
answers. Two routes speak the same request shape; OpenRouter is used when
OPENROUTER_API_KEY is set, otherwise TypeSafe directly (--provider openrouter
with no key sends no Authorization header, for a proxy that injects it):

    POST https://openrouter.ai/api/alpha/decisions   model typesafe/jev-1.13  key $OPENROUTER_API_KEY
    POST https://api.typesafe.ai/v1/systemone        model jev-latest         key $TYPESAFE_API_KEY

    Authorization: Bearer <key>
    {"model": "...", "state": {...},
     "questions": {"<id>": {"type": "choice"|"score"|"noul",
                            "instructions": "...", "criteria": ...}}}

and each answer comes back under data["answers"]["<id>"], for example
{"type": "choice", "choice": "pass", "confidence": 0.9, "probabilities": {...}}.
Several questions in one request are evaluated in parallel.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
import urllib.error
import urllib.request

ROUTES = {
    "openrouter": {"key": "OPENROUTER_API_KEY", "url": "https://openrouter.ai/api/alpha/decisions", "model": "typesafe/jev-1.13"},
    "typesafe": {"key": "TYPESAFE_API_KEY", "url": "https://api.typesafe.ai/v1/systemone", "model": "jev-latest"},
}
PRICE_PER_INPUT_TOKEN = 0.042 / 1_000_000  # USD; output tokens are free
MAX_RPS = 18.0  # Jev allows 1200 requests/minute; stay a little under it


class RateLimiter:
    """Spaces request starts at least 1/rps seconds apart, across threads."""

    def __init__(self, rps: float):
        self.interval = 1.0 / rps if rps > 0 else 0.0
        self.next_start = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next_start)
            self.next_start = start + self.interval
        if start > now:
            time.sleep(start - now)


class JevError(RuntimeError):
    pass


class CallBudget:
    """A hard cap on Jev calls shared by every client in a run (the run's spending limit)."""

    def __init__(self, max_calls: int | None):
        self.max_calls = max_calls
        self.calls = 0
        self.lock = threading.Lock()

    def take(self) -> None:
        with self.lock:
            if self.max_calls is not None and self.calls >= self.max_calls:
                raise JevError(f"run budget of {self.max_calls} Jev calls reached")
            self.calls += 1


def write_log(path: str | None, lock: threading.Lock, record: dict) -> None:
    if path:
        with lock, open(path, "a") as f:
            f.write(json.dumps(record) + "\n")


def load_env_file(path: str = ".env.local") -> None:
    """Read NAME=VALUE lines into os.environ. Variables already set in the process win."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            os.environ.setdefault(name.strip(), value.strip())


def pick_route(provider: str | None = None) -> tuple[str, dict, str]:
    """(provider, route, key). With no provider, OpenRouter wins if its key is set.

    Naming provider="openrouter" explicitly works without a local key: the
    request then goes out with no Authorization header, for setups where a
    proxy adds the credential (e.g. a cloud sandbox's egress gateway).
    """
    names = [provider] if provider else list(ROUTES)
    for name in names:
        if name not in ROUTES:
            raise JevError(f"unknown provider {name!r}; choose from {sorted(ROUTES)}")
        key = os.environ.get(ROUTES[name]["key"], "").strip()
        if key:
            return name, ROUTES[name], key
    if provider == "openrouter":
        return provider, ROUTES[provider], ""
    wanted = " or ".join(ROUTES[n]["key"] for n in names)
    raise JevError(f"Set {wanted} (in the environment or .env.local), or pass --backend mock to run offline.")


class JevClient:
    """Calls the real Jev API. Every call is appended to `log_path` (JSONL) if given, with the caller's `meta`
    (run, condition, game, seat, time, decision number) so each call can be joined to its game.

    Safe to share between threads: requests are paced by one rate limiter, and
    `last_latency` is per thread, so each game sees its own round-trip times.
    """

    backend = "jev"

    def __init__(self, provider: str | None = None, model: str | None = None, log_path: str | None = None,
                 timeout: float = 20.0, max_calls: int | None = None, env_file: str = ".env.local",
                 retries: int = 3, max_rps: float = MAX_RPS, limiter: RateLimiter | None = None,
                 budget: CallBudget | None = None):
        load_env_file(env_file)
        self.provider, route, self.api_key = pick_route(provider)
        self.url = route["url"]
        self.model = model or os.environ.get("JEV_MODEL") or route["model"]
        self.backend = f"jev ({self.provider}, {self.model})"
        self.log_path = log_path
        self.timeout = timeout
        self.retries = retries  # for timeouts, connection errors, 429 and 5xx
        self.max_calls = max_calls
        self.calls = 0
        self.input_tokens = 0
        self.limiter = limiter or RateLimiter(max_rps)
        self.budget = budget
        self.lock = threading.Lock()
        self.log_lock = threading.Lock()
        self.local = threading.local()

    @property
    def last_latency(self) -> float:
        return getattr(self.local, "latency", 0.0)

    @property
    def cost_usd(self) -> float:
        return self.input_tokens * PRICE_PER_INPUT_TOKEN

    def ask(self, state, questions: dict, meta: dict | None = None) -> dict:
        with self.lock:
            if self.max_calls is not None and self.calls >= self.max_calls:
                raise JevError(f"max_calls={self.max_calls} reached")
            self.calls += 1
        if self.budget:
            self.budget.take()
        payload = {"model": self.model, "state": state, "questions": questions}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.url, data=json.dumps(payload).encode(), method="POST", headers=headers)
        for attempt in range(self.retries + 1):
            self.limiter.wait()
            start = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read())
                break
            except urllib.error.HTTPError as e:
                if e.code < 500 and e.code != 429 or attempt == self.retries:
                    raise JevError(f"Jev HTTP {e.code}: {e.read()[:500]!r}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt == self.retries:
                    raise JevError(f"Jev request failed: {getattr(e, 'reason', e)}") from None
            time.sleep(2 ** attempt)
        latency = self.local.latency = time.perf_counter() - start
        with self.lock:
            self.input_tokens += int(data.get("usage", {}).get("input_tokens", 0))
        write_log(self.log_path, self.log_lock, {"meta": meta or {}, "wall_time": time.time(), "latency": latency,
                                                 "request": payload, "response": data})
        return data["answers"]


class MockJev:
    """Offline stand-in with the same interface. It is NOT Jev.

    It answers every question with random probabilities, so results produced
    with it only show that the pipeline runs; they say nothing about Jev. It
    writes the same log records as the real client (with an estimated token
    count), so the logging and cost paths can be checked for free.
    """

    backend = "mock"

    def __init__(self, seed: int = 0, latency: float = 0.3, log_path: str | None = None, max_calls: int | None = None,
                 budget: CallBudget | None = None, **_):
        self.rng = random.Random(seed)
        self.lock = threading.Lock()
        self.log_lock = threading.Lock()
        self.calls = 0
        self.input_tokens = 0
        self.last_latency = latency
        self.log_path = log_path
        self.max_calls = max_calls
        self.budget = budget
        self.model = "mock"

    @property
    def cost_usd(self) -> float:
        return self.input_tokens * PRICE_PER_INPUT_TOKEN

    def ask(self, state, questions: dict, meta: dict | None = None) -> dict:
        payload = {"model": self.model, "state": state, "questions": questions}
        tokens = len(json.dumps(payload)) // 4  # rough: about 4 characters per token
        with self.lock:
            if self.max_calls is not None and self.calls >= self.max_calls:
                raise JevError(f"max_calls={self.max_calls} reached")
            self.calls += 1
            self.input_tokens += tokens
            answers = self._answer(questions)
        if self.budget:
            self.budget.take()
        write_log(self.log_path, self.log_lock, {"meta": meta or {}, "wall_time": time.time(), "latency": self.last_latency,
                                                 "request": payload,
                                                 "response": {"answers": answers, "usage": {"input_tokens": tokens}}})
        return answers

    def _answer(self, questions: dict) -> dict:
        answers = {}
        for qid, q in questions.items():
            if q["type"] == "choice":
                labels = list(q["criteria"])
                w = [self.rng.expovariate(1.0) for _ in labels]
                total = sum(w)
                probs = {k: x / total for k, x in zip(labels, w)}
                choice = max(probs, key=probs.get)
                answers[qid] = {"type": "choice", "choice": choice, "confidence": probs[choice], "probabilities": probs}
            elif q["type"] == "score":
                n = len(q["criteria"])
                answers[qid] = {"type": "score", "score": self.rng.uniform(0, n - 1)}
            else:
                answers[qid] = {"type": "noul", "noul": self.rng.random()}
        return answers


def make_client(backend: str, **kw):
    if backend == "jev":
        return JevClient(**{k: v for k, v in kw.items()
                            if k in ("provider", "log_path", "max_calls", "max_rps", "limiter", "budget")})
    if backend == "mock":
        return MockJev(**kw)
    raise ValueError(f"unknown backend {backend!r}")
