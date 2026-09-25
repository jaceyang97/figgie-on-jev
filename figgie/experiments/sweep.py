"""Run every Jev personality against the same field and print one table.

Each personality takes one seat against the same three opponents, so the rows
are directly comparable. Each rule-based twin (and any other baseline) takes
the same seat too. All seats and games run at once; one rate limiter keeps
the Jev calls under the API's request limit. Market columns show how that one participant changes
the game for everyone.

Usage:
  python -m figgie.experiments.sweep --backend mock --games 10
  python -m figgie.experiments.sweep --backend jev --games 20 --field fundamentalist,bottom_feeder,chartist
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from ..agents import CLASSICAL
from ..jev import MAX_RPS, RateLimiter, make_client
from ..personalities import PERSONALITIES
from .tournament import run as run_tournament


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["mock", "jev"], default="mock")
    ap.add_argument("--provider", choices=["openrouter", "typesafe"], default=None,
                    help="Jev route (default: OpenRouter if OPENROUTER_API_KEY is set, else TypeSafe)")
    ap.add_argument("--field", default="fundamentalist,bottom_feeder,noise", help="the three fixed opponents")
    ap.add_argument("--personalities", default=",".join(PERSONALITIES))
    ap.add_argument("--baselines", default=",".join(CLASSICAL),
                    help="rule-based agents to put in the same seat for reference ('' for none)")
    ap.add_argument("--assist", action="store_true", help="give each Jev agent the exact goal probabilities")
    ap.add_argument("--context", default="", help="comma list of extra context parts: log, summary, known, assist")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--duration", type=float, default=240.0)
    ap.add_argument("--jev-latency", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-calls", type=int, default=5000, help="hard cap on Jev API calls per personality")
    ap.add_argument("--max-rps", type=float, default=MAX_RPS, help="Jev requests per second, shared by all seats")
    ap.add_argument("--log-dir", help="write each Jev seat's requests and responses to <dir>/<personality>.jsonl")
    ap.add_argument("--out", help="write sweep.json here")
    args = ap.parse_args(argv)

    field = args.field.split(",")
    if len(field) != 3:
        raise SystemExit("--field needs exactly 3 agents")
    parts = [p for p in args.context.split(",") if p] + (["assist"] if args.assist else [])
    flags = "".join(f"+{p}" for p in parts)
    seats = [(p, f"jev:{p}{flags}") for p in args.personalities.split(",") if p]
    seats += [(f"[{b}]", b) for b in args.baselines.split(",") if b]
    limiter = RateLimiter(args.max_rps)
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)

    def play(seat):
        p, spec = seat
        client = None
        if spec.startswith("jev"):
            log = os.path.join(args.log_dir, f"{p}.jsonl") if args.log_dir else None
            client = make_client(args.backend, seed=args.seed, log_path=log, max_calls=args.max_calls,
                                 provider=args.provider, limiter=limiter)
        t = SimpleNamespace(backend=args.backend, provider=args.provider, lineup=",".join([spec] + field),
                            games=args.games, duration=args.duration, jev_latency=args.jev_latency, seed=args.seed,
                            max_calls=args.max_calls, log=None, out=None, workers=None)
        s = run_tournament(t, client)
        key = spec if spec in s["agents"] else f"{spec}#0"
        return p, {"jev": s["agents"][key], "field": {k: v["mean_pnl"] for k, v in s["agents"].items() if k != key},
                   "market": s["market"], "calls": s.get("jev_calls", 0), "cost_usd": s.get("cost_usd", 0.0)}

    # Every seat and every game runs at once; the shared limiter paces the Jev calls.
    with ThreadPoolExecutor(max_workers=len(seats)) as pool:
        rows = dict(pool.map(play, seats))

    if args.backend == "mock":
        print("NOTE: the mock backend answers at random and ignores personalities; Jev rows say nothing about Jev.\n")
    print(f"{'agent in seat':<18}{'P&L':>9}{'95% CI':>20}{'trades':>9}{'mkt trades':>12}{'mispricing':>12}")
    for p, r in rows.items():
        lo, hi = r["jev"]["ci95"]
        print(f"{p:<18}{r['jev']['mean_pnl']:>9.1f}{f'[{lo:.0f}, {hi:.0f}]':>20}{r['jev']['trades_per_game']:>9.1f}"
              f"{r['market']['trades_per_game']:>12.1f}{r['market']['mean_mispricing_chips']:>12.2f}")
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "sweep.json"), "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
