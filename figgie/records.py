"""What each game and each run writes to disk.

One JSON line per game holds everything the planned figures and statistics
need, so no analysis has to replay a game: the seed and deal, every trade with
its time and both sides, every order and cancel, each seat's decisions,
passes and rejected orders, cash and payout, and, for Jev seats, one line per
decision (time, chosen action, its probability, the goal-suit belief). Jev's
full probability maps stay in the client log, joined by run, condition, game,
seat and decision number.
"""

from __future__ import annotations

import datetime
import json
import os
import platform
import subprocess
import sys

from .cards import SUITS
from .engine import GameResult


def git_state() -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=here, capture_output=True, text=True,
                                    timeout=5).stdout.strip())
    except Exception:  # not a git checkout
        sha, dirty = "", None
    return {"git_sha": sha, "git_dirty": dirty}


def run_header(kind: str, args: dict) -> dict:
    """Written once per run as run.json: what ran, with which code and settings."""
    return {
        "kind": kind,
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        **git_state(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "argv": sys.argv,
        "args": args,
    }


def write_run_header(out_dir: str, kind: str, args: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "run.json"), "w") as f:
        json.dump(run_header(kind, args), f, indent=2)


def game_record(res: GameResult, game: int, seed: int, seats: list[str], condition: str | None = None,
                agents=None, extra: dict | None = None) -> dict:
    """The full record of one game (see the module docstring)."""
    rec = {
        "game": game, "seed": seed, "condition": condition, "mechanism": res.mechanism,
        "seats": seats, "labels": res.labels, "speeds": [list(s) for s in res.speeds],
        "deck": {"common": res.deck.common, "eight": res.deck.eight, "goal": res.deck.goal,
                 "goal_cards": res.deck.count(res.deck.goal)},
        "goal": res.deck.goal,
        "initial_hands": res.initial_hands, "final_hands": res.final_hands,
        "pnl": res.pnl, "final_chips": res.final_chips, "payout": res.payout,
        "decisions": res.decisions, "passes": res.passes, "rejected": res.rejected,
        "seat_trades": [sum(1 for t in res.trades if seat in (t.buyer, t.seller)) for seat in range(len(seats))],
        "events": res.events, "t_end": round(res.t_end, 3),
        # trades: [t, suit, price, buyer, seller, aggressor]
        "trades": [[round(t.t, 3), t.suit, t.price, t.buyer, t.seller, t.aggressor] for t in res.trades],
        # orders accepted onto the book: [t, player, side, suit, price, oid]
        "orders": [[round(o.t, 3), o.player, o.side, o.suit, o.price, o.oid] for o in res.orders],
        # orders that left the book without trading: [t, player, oid, suit, side, price, reason]
        "cancels": [[round(c.t, 3), c.player, c.oid, c.suit, c.side, c.price, c.reason] for c in res.cancels],
        "trade_prices": [[t.suit, t.price] for t in res.trades],  # kept for older analysis code
    }
    if agents is not None:
        traces = {str(p): a.trace for p, a in enumerate(agents) if getattr(a, "trace", None) is not None}
        if traces:
            rec["jev_trace"] = traces
    if extra:
        rec.update(extra)
    return rec


def starting_goal_cards(rec: dict, seat: int) -> int:
    return rec["initial_hands"][seat][rec["goal"]]


__all__ = ["game_record", "run_header", "write_run_header", "git_state", "starting_goal_cards", "SUITS"]
