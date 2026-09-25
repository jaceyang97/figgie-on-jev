"""Experiment 2: games between Jev personalities and classical agents.

Plays N games with a fixed line-up of four agents and reports, per seat, the
mean profit with a bootstrap 95% CI (resampling whole games, as in the paper),
plus how active each seat was. Market-level metrics show how the mix of
personalities changes the game: trades per game, and mean mispricing (|trade
price - what the card turned out to be worth|, where a goal card is worth 10
chips and any other card 0).

Seats rotate each game so no personality keeps a seat advantage.

Usage:
  python -m figgie.experiments.tournament --backend mock --games 20 \\
      --lineup jev:hoarder,jev:market_maker,fundamentalist,bottom_feeder
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

from ..agents import make_agent
from ..engine import play_game
from ..jev import make_client
from ..stats import bootstrap_ci, mean


def run(args) -> dict:
    rng = random.Random(args.seed)
    specs = args.lineup.split(",")
    if len(specs) != 4:
        raise SystemExit("--lineup needs exactly 4 agents")
    client = make_client(args.backend, seed=args.seed, log_path=args.log, max_calls=args.max_calls) if any(s.startswith("jev") for s in specs) else None
    pnl = defaultdict(list)
    trades_by = defaultdict(list)
    rejected_by = defaultdict(list)
    n_trades, mispricing = [], []
    games = []
    for g in range(args.games):
        order = specs[g % 4 :] + specs[: g % 4]
        agents = [make_agent(s, client, jev_latency=args.jev_latency) for s in order]
        res = play_game(agents, rng, duration=args.duration)
        goal = res.deck.goal
        for seat, spec in enumerate(order):
            key = f"{spec}#{specs.index(spec)}" if specs.count(spec) > 1 else spec
            pnl[key].append(res.pnl[seat])
            trades_by[key].append(sum(1 for t in res.trades if seat in (t.buyer, t.seller)))
            rejected_by[key].append(res.rejected[seat])
        n_trades.append(len(res.trades))
        mispricing.extend(abs(t.price - (10 if t.suit == goal else 0)) for t in res.trades)
        games.append({"game": g, "seats": order, "goal": goal, "pnl": res.pnl, "trades": len(res.trades)})

    summary = {
        "backend": client.backend if client else "none",
        "games": args.games,
        "lineup": specs,
        "agents": {
            k: {
                "mean_pnl": round(mean(v), 2),
                "ci95": [round(x, 2) for x in bootstrap_ci(v)],
                "trades_per_game": round(mean(trades_by[k]), 2),
                "rejected_orders_per_game": round(mean(rejected_by[k]), 2),
            }
            for k, v in pnl.items()
        },
        "market": {
            "trades_per_game": round(mean(n_trades), 2),
            "mean_mispricing_chips": round(mean(mispricing), 2),
        },
    }
    if client:
        summary["jev_calls"] = client.calls
        summary["cost_usd"] = round(client.cost_usd, 4)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        with open(os.path.join(args.out, "games.jsonl"), "w") as f:
            for row in games:
                f.write(json.dumps(row) + "\n")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["mock", "jev"], default="mock")
    ap.add_argument("--lineup", default="jev:neutral,fundamentalist,bottom_feeder,noise")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--duration", type=float, default=240.0)
    ap.add_argument("--jev-latency", type=float, default=None,
                    help="simulated Jev latency in seconds (default: the measured round-trip time)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-calls", type=int, default=5000, help="hard cap on Jev API calls")
    ap.add_argument("--log", help="append every Jev request/response to this JSONL file")
    ap.add_argument("--out", help="directory for summary.json and games.jsonl")
    print(json.dumps(run(ap.parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()
