"""Run every Jev personality against the same field and print one table.

Each personality takes one seat against the same three opponents, so the rows
are directly comparable. Market columns show how that one participant changes
the game for everyone.

Usage:
  python -m figgie.experiments.sweep --backend mock --games 10
  python -m figgie.experiments.sweep --backend jev --games 20 --field fundamentalist,bottom_feeder,chartist
"""

from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace

from ..personalities import PERSONALITIES
from .tournament import run as run_tournament


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["mock", "jev"], default="mock")
    ap.add_argument("--provider", choices=["openrouter", "typesafe"], default=None,
                    help="Jev route (default: OpenRouter if OPENROUTER_API_KEY is set, else TypeSafe)")
    ap.add_argument("--field", default="fundamentalist,bottom_feeder,noise", help="the three fixed opponents")
    ap.add_argument("--personalities", default=",".join(PERSONALITIES))
    ap.add_argument("--baselines", default="fundamentalist,noise",
                    help="classical agents to put in the same seat for reference ('' for none)")
    ap.add_argument("--assist", action="store_true", help="give each Jev agent the exact goal probabilities")
    ap.add_argument("--history", action="store_true", help="add a compressed summary of the whole game log to Jev's state")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--duration", type=float, default=240.0)
    ap.add_argument("--jev-latency", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-calls", type=int, default=5000, help="hard cap on Jev API calls per personality")
    ap.add_argument("--log")
    ap.add_argument("--out", help="write sweep.json here")
    args = ap.parse_args(argv)

    field = args.field.split(",")
    if len(field) != 3:
        raise SystemExit("--field needs exactly 3 agents")
    rows = {}
    flags = ("+history" if args.history else "") + ("+assist" if args.assist else "")
    seats = [(p, f"jev:{p}{flags}") for p in args.personalities.split(",") if p]
    seats += [(f"[{b}]", b) for b in args.baselines.split(",") if b]
    for p, spec in seats:
        t = SimpleNamespace(backend=args.backend, provider=args.provider, lineup=",".join([spec] + field), games=args.games,
                            duration=args.duration, jev_latency=args.jev_latency, seed=args.seed,
                            max_calls=args.max_calls, log=args.log, out=None)
        s = run_tournament(t)
        key = spec if spec in s["agents"] else f"{spec}#0"
        rows[p] = {"jev": s["agents"][key], "field": {k: v["mean_pnl"] for k, v in s["agents"].items() if k != key},
                   "market": s["market"], "cost_usd": s.get("cost_usd", 0.0)}

    if args.backend == "mock":
        print("NOTE: the mock backend answers at random and ignores personalities, so Jev rows are identical.\n")
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
