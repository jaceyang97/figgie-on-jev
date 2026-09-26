"""Play N games with a fixed line-up of four agents.

Reports, per seat, the mean profit with a bootstrap 95% CI (resampling whole
games, as in the paper), how active each seat was, and market metrics: trades
per game and mean mispricing (|trade price - card value|, where a goal-suit
card is worth 10 chips and any other card 0).

Seats rotate each game so no agent keeps a seat advantage. Game g always uses
seed `game_seed(seed, g)`, so the same g has the same deal in every line-up
and in both rule sets. Every finished game is appended to the checkpoint file
as a full record (figgie.records), and a rerun resumes from it.

Usage:
  python -m figgie.experiments.tournament --games 100 --mechanism A \\
      --lineup fundamentalist,bottom_feeder,noise,noise
  python -m figgie.experiments.tournament --backend mock --games 4 \\
      --lineup jev:neutral,fundamentalist,bottom_feeder,noise --tested 0 --out /tmp/t
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

from ..agents import SPEEDS, make_agent, parse_speed
from ..engine import play_game
from ..jev import make_client
from ..records import game_record, write_run_header
from ..stats import bootstrap_ci, mean


def game_seed(seed: int, g: int) -> int:
    return seed * 1_000_003 + g


def game_rng(seed: int, g: int) -> random.Random:
    """Each game gets its own seeded RNG, so games can run in any order or in parallel."""
    return random.Random(game_seed(seed, g))


def load_checkpoint(path: str | None, games: int, rotate) -> dict[int, dict]:
    done = {}
    if path and os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:  # a line cut off when a run was stopped
                    continue
                if rec["game"] < games and rec["seats"] == rotate(rec["game"]):
                    done[rec["game"]] = rec
    return done


def run(args, client=None) -> dict:
    specs = args.lineup.split(",")
    if len(specs) != 4:
        raise SystemExit("--lineup needs exactly 4 agents")
    mechanism = getattr(args, "mechanism", "B")
    speed = parse_speed(getattr(args, "speed", None), SPEEDS["equal"])
    tested = getattr(args, "tested", None)  # index into the line-up of the seat under test
    condition = getattr(args, "condition", None) or args.lineup
    run_id = getattr(args, "run_id", None)
    if client is None and any(s.startswith("jev") for s in specs):
        client = make_client(args.backend, seed=args.seed, log_path=args.log, max_calls=args.max_calls,
                             provider=args.provider)

    def rotate(g: int) -> list[str]:
        return specs[g % 4:] + specs[: g % 4]

    checkpoint = getattr(args, "checkpoint", None)
    if checkpoint is None and getattr(args, "out", None):
        os.makedirs(args.out, exist_ok=True)
        checkpoint = os.path.join(args.out, "games.jsonl")
        write_run_header(args.out, "tournament", {k: v for k, v in vars(args).items()})
    done = load_checkpoint(checkpoint, args.games, rotate)
    lock = threading.Lock()
    errors = []

    def play(g: int):
        order = rotate(g)
        shift = g % 4
        tested_seat = None if tested is None else (tested - shift) % 4
        meta = {"run": run_id, "condition": condition, "game": g, "seed": game_seed(args.seed, g)}
        agents = [make_agent(s, client, speed=speed, decode=getattr(args, "decode", "sample"), meta=meta) for s in order]
        try:
            res = play_game(agents, game_rng(args.seed, g), duration=args.duration, mechanism=mechanism,
                            max_events=getattr(args, "max_events", None), tested_seat=tested_seat)
        except Exception as e:  # e.g. the API refused a call; keep the games that finished
            with lock:
                errors.append(f"game {g}: {e}")
            return None
        rec = game_record(res, g, game_seed(args.seed, g), order, condition=condition, agents=agents,
                          extra={"tested_seat": tested_seat, "run": run_id})
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
    if not records:
        raise RuntimeError(f"no games finished for {args.lineup}: {errors[:1]}")
    summary = summarise(records, specs)
    summary.update({"backend": client.backend if client else "none", "games_failed": len(errors),
                    "mechanism": mechanism, "speed": list(speed)})
    if client:
        summary["jev_calls"] = client.calls
        summary["cost_usd"] = round(client.cost_usd, 4)
    if getattr(args, "out", None):
        with open(os.path.join(args.out, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
    return summary


def seat_key(spec: str, specs: list[str], lineup_index: int) -> str:
    return f"{spec}#{lineup_index}" if specs.count(spec) > 1 else spec


def summarise(records: list[dict], specs: list[str]) -> dict:
    pnl, cash, payout = defaultdict(list), defaultdict(list), defaultdict(list)
    trades_by, rejected_by, decisions_by, passes_by = (defaultdict(list) for _ in range(4))
    n_trades, mispricing = [], []
    for rec in records:
        order, goal = rec["seats"], rec["goal"]
        shift = rec["game"] % 4  # order is specs rotated by the game number
        for seat, spec in enumerate(order):
            key = seat_key(spec, specs, (seat + shift) % 4)
            pnl[key].append(rec["pnl"][seat])
            cash[key].append(rec["final_chips"][seat])
            payout[key].append(rec["payout"][seat])
            trades_by[key].append(rec["seat_trades"][seat])
            rejected_by[key].append(rec["rejected"][seat])
            decisions_by[key].append(rec["decisions"][seat])
            passes_by[key].append(rec["passes"][seat])
        n_trades.append(len(rec["trades"]))
        mispricing.extend(abs(price - (10 if suit == goal else 0)) for _, suit, price, *_ in rec["trades"])

    def sd(xs):
        m = mean(xs)
        return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else float("nan")

    return {
        "games": len(records),
        "lineup": specs,
        "agents": {
            k: {
                "mean_pnl": round(mean(v), 2),
                "ci95": [round(x, 2) for x in bootstrap_ci(v)],
                "sd_pnl": round(sd(v), 2),
                "sd_cash": round(sd(cash[k]), 2),
                "sd_payout": round(sd(payout[k]), 2),
                "trades_per_game": round(mean(trades_by[k]), 2),
                "decisions_per_game": round(mean(decisions_by[k]), 2),
                "passes_per_game": round(mean(passes_by[k]), 2),
                "rejected_orders_per_game": round(mean(rejected_by[k]), 2),
            }
            for k, v in pnl.items()
        },
        "market": {
            "trades_per_game": round(mean(n_trades), 2),
            "mean_mispricing_chips": round(mean(mispricing), 2),
        },
    }


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--backend", choices=["mock", "jev"], default="mock")
    ap.add_argument("--provider", choices=["openrouter", "typesafe"], default=None,
                    help="Jev route (default: OpenRouter if OPENROUTER_API_KEY is set, else TypeSafe)")
    ap.add_argument("--duration", type=float, default=240.0)
    ap.add_argument("--max-events", type=int, default=None, help="end games after this many events (the paper: 10000)")
    ap.add_argument("--speed", default="equal", help="'equal' (0.5/s, 0.3 s), 'paper' (1/s, 0 s) or '<rate>/<latency>'")
    ap.add_argument("--decode", choices=["sample", "argmax", "hierarchical"], default="sample")
    ap.add_argument("--seed", type=int, default=0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--lineup", default="jev:neutral,fundamentalist,bottom_feeder,noise")
    ap.add_argument("--mechanism", choices=["A", "B"], default="B")
    ap.add_argument("--tested", type=int, default=None, help="line-up index of the seat under test (hidden from prey choice)")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--workers", type=int, default=None, help="games played at once (default: all, when Jev plays)")
    ap.add_argument("--max-calls", type=int, default=5000, help="hard cap on Jev API calls")
    ap.add_argument("--log", help="append every Jev request/response to this JSONL file")
    ap.add_argument("--out", help="directory for run.json, games.jsonl (full records) and summary.json")
    print(json.dumps(run(ap.parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()
