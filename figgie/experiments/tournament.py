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
      --lineup jev:chartist,jev:market_maker,fundamentalist,bottom_feeder
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from ..agents import make_agent
from ..engine import play_game
from ..jev import make_client
from ..stats import bootstrap_ci, mean


def game_rng(seed: int, g: int) -> random.Random:
    """Each game gets its own seeded RNG, so games can run in any order or in parallel."""
    return random.Random(seed * 1_000_003 + g)


def run(args, client=None) -> dict:
    specs = args.lineup.split(",")
    if len(specs) != 4:
        raise SystemExit("--lineup needs exactly 4 agents")
    if client is None and any(s.startswith("jev") for s in specs):
        client = make_client(args.backend, seed=args.seed, log_path=args.log, max_calls=args.max_calls,
                             provider=args.provider)

    done = {}
    checkpoint = getattr(args, "checkpoint", None)
    if checkpoint and os.path.exists(checkpoint):
        with open(checkpoint) as f:
            for line in f:
                rec = json.loads(line)
                if rec["game"] < args.games and rec["seats"] == specs[rec["game"] % 4 :] + specs[: rec["game"] % 4]:
                    done[rec["game"]] = rec
    lock = threading.Lock()
    errors = []

    def play(g: int):
        order = specs[g % 4 :] + specs[: g % 4]
        agents = [make_agent(s, client, jev_latency=args.jev_latency) for s in order]
        try:
            res = play_game(agents, game_rng(args.seed, g), duration=args.duration)
        except Exception as e:  # e.g. the API refused a call; keep the games that finished
            with lock:
                errors.append(f"game {g}: {e}")
            return None
        rec = {
            "game": g, "seats": order, "goal": res.deck.goal, "pnl": res.pnl,
            "seat_trades": [sum(1 for t in res.trades if seat in (t.buyer, t.seller)) for seat in range(4)],
            "rejected": res.rejected, "trade_prices": [[t.suit, t.price] for t in res.trades],
        }
        if checkpoint:
            with lock, open(checkpoint, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return rec

    # Jev games spend their time waiting on the API, so run them side by side;
    # the client's rate limiter keeps the total under Jev's request limit.
    todo = [g for g in range(args.games) if g not in done]
    workers = getattr(args, "workers", None) or (max(1, len(todo)) if client else 1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(done.values()) + [r for r in pool.map(play, todo) if r is not None]
    records.sort(key=lambda r: r["game"])
    for e in errors:
        print(f"WARNING {args.lineup}: {e}", file=sys.stderr)

    pnl = defaultdict(list)
    trades_by = defaultdict(list)
    rejected_by = defaultdict(list)
    n_trades, mispricing = [], []
    games = []
    for rec in records:
        order, goal = rec["seats"], rec["goal"]
        for seat, spec in enumerate(order):
            key = f"{spec}#{specs.index(spec)}" if specs.count(spec) > 1 else spec
            pnl[key].append(rec["pnl"][seat])
            trades_by[key].append(rec["seat_trades"][seat])
            rejected_by[key].append(rec["rejected"][seat])
        n_trades.append(len(rec["trade_prices"]))
        mispricing.extend(abs(price - (10 if suit == goal else 0)) for suit, price in rec["trade_prices"])
        games.append({"game": rec["game"], "seats": order, "goal": goal, "pnl": rec["pnl"], "trades": len(rec["trade_prices"])})
    if not records:
        raise RuntimeError(f"no games finished for {args.lineup}: {errors[:1]}")

    summary = {
        "backend": client.backend if client else "none",
        "games": len(records),
        "games_failed": len(errors),
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
    ap.add_argument("--provider", choices=["openrouter", "typesafe"], default=None,
                    help="Jev route (default: OpenRouter if OPENROUTER_API_KEY is set, else TypeSafe)")
    ap.add_argument("--lineup", default="jev:neutral,fundamentalist,bottom_feeder,noise")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--duration", type=float, default=240.0)
    ap.add_argument("--jev-latency", type=float, default=None,
                    help="simulated Jev latency in seconds (default: the measured round-trip time)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None, help="games played at once (default: all, when Jev plays)")
    ap.add_argument("--max-calls", type=int, default=5000, help="hard cap on Jev API calls")
    ap.add_argument("--log", help="append every Jev request/response to this JSONL file")
    ap.add_argument("--out", help="directory for summary.json and games.jsonl")
    print(json.dumps(run(ap.parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()
