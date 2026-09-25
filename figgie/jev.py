"""Minimal client for TypeSafe AI's Jev decision model, plus an offline stand-in.

Jev takes a JSON "state" and a set of typed questions and returns numeric
answers. Two routes speak the same request shape; OpenRouter is used when
OPENROUTER_API_KEY is set, otherwise TypeSafe directly:

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
import time
import urllib.error
import urllib.request

ROUTES = {
    "openrouter": {"key": "OPENROUTER_API_KEY", "url": "https://openrouter.ai/api/alpha/decisions", "model": "typesafe/jev-1.13"},
    "typesafe": {"key": "TYPESAFE_API_KEY", "url": "https://api.typesafe.ai/v1/systemone", "model": "jev-latest"},
}
PRICE_PER_INPUT_TOKEN = 0.042 / 1_000_000  # USD; output tokens are free


class JevError(RuntimeError):
    pass


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
    """(provider, route, key). With no provider, OpenRouter wins if its key is set."""
    names = [provider] if provider else list(ROUTES)
    for name in names:
        if name not in ROUTES:
            raise JevError(f"unknown provider {name!r}; choose from {sorted(ROUTES)}")
        key = os.environ.get(ROUTES[name]["key"], "").strip()
        if key:
            return name, ROUTES[name], key
    wanted = " or ".join(ROUTES[n]["key"] for n in names)
    raise JevError(f"Set {wanted} (in the environment or .env.local), or pass --backend mock to run offline.")


class JevClient:
    """Calls the real Jev API. Every call is appended to `log_path` (JSONL) if given."""

    backend = "jev"

    def __init__(self, provider: str | None = None, model: str | None = None, log_path: str | None = None,
                 timeout: float = 20.0, max_calls: int | None = None, env_file: str = ".env.local"):
        load_env_file(env_file)
        self.provider, route, self.api_key = pick_route(provider)
        self.url = route["url"]
        self.model = model or os.environ.get("JEV_MODEL") or route["model"]
        self.backend = f"jev ({self.provider}, {self.model})"
        self.log_path = log_path
        self.timeout = timeout
        self.max_calls = max_calls
        self.calls = 0
        self.input_tokens = 0
        self.last_latency = 0.0

    @property
    def cost_usd(self) -> float:
        return self.input_tokens * PRICE_PER_INPUT_TOKEN

    def ask(self, state, questions: dict) -> dict:
        if self.max_calls is not None and self.calls >= self.max_calls:
            raise JevError(f"max_calls={self.max_calls} reached")
        payload = {"model": self.model, "state": state, "questions": questions}
        req = urllib.request.Request(
            self.url, data=json.dumps(payload).encode(), method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise JevError(f"Jev HTTP {e.code}: {e.read()[:500]!r}") from None
        except urllib.error.URLError as e:
            raise JevError(f"Jev request failed: {e.reason}") from None
        self.last_latency = time.perf_counter() - start
        self.calls += 1
        self.input_tokens += int(data.get("usage", {}).get("input_tokens", 0))
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps({"latency": self.last_latency, "request": payload, "response": data}) + "\n")
        return data["answers"]


class MockJev:
    """Offline stand-in with the same interface. It is NOT Jev.

    It answers every question with random probabilities, so results produced
    with it only show that the pipeline runs; they say nothing about Jev.
    """

    backend = "mock"

    def __init__(self, seed: int = 0, latency: float = 0.3, **_):
        self.rng = random.Random(seed)
        self.calls = 0
        self.input_tokens = 0
        self.cost_usd = 0.0
        self.last_latency = latency

    def ask(self, state, questions: dict) -> dict:
        self.calls += 1
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
        return JevClient(**{k: v for k, v in kw.items() if k in ("provider", "log_path", "max_calls")})
    if backend == "mock":
        return MockJev(**kw)
    raise ValueError(f"unknown backend {backend!r}")
