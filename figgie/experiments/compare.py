"""Legacy (the first run, 2026-09-25): Jev's decisions vs card counting, rule set B only.

Superseded by figgie.experiments.stage1 (frozen moments in both rule sets, persona wording,
state versions, repeatability). Kept so the first run can be reproduced.

Plays games between classical agents and, at random decision points of one
seat, freezes the view and asks Jev two questions in one call:

  goal   - which suit is the goal suit? (compared with card counting)
  action - which action would you take? (compared with the fundamentalist)

Metrics
  Brier score and log loss of the goal-suit probabilities against the true goal
  suit, for Jev, card counting and a uniform 25% guess; mean total
  variation distance between Jev and the posterior; top-1 agreement.
  For actions: how often Jev picks the same action as the fundamentalist, and
  Jev's regret in chips: the best immediate edge available from taking a price
  minus the edge of Jev's choice (quotes and passing count as 0 edge, taking a
  bad price counts negative).

Usage:
  python -m figgie.experiments.compare --backend mock --games 20
  python -m figgie.experiments.compare --backend jev --games 20 --out results/compare
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor

from ..agents import Fundamentalist, make_agent
from ..agents.classical import take_edges
from ..agents.jev_agent import action_menu, action_question, add_context, goal_question, hierarchical_choice
from ..cards import SUITS
from ..engine import View, play_game
from ..jev import make_client
from ..posterior import card_values
from ..stats import bootstrap_ci, mean


class Probe(Fundamentalist):
    """A fundamentalist that snapshots some of its decision points for later questioning."""

    def __init__(self, p_snapshot: float, **kw):
        super().__init__(**kw)
        self.p_snapshot = p_snapshot
        self.snapshots = []

    def decide(self, view: View):
        action = super().decide(view)
        if view.trades and self.rng.random() < self.p_snapshot:
            known = self.counter.known()
            self.snapshots.append({
                "view": view, "known": known, "posterior": self.counter.goal_probabilities(),
                "values": card_values(known, view.hand), "classical_action": action,
            })
        return action


def brier(probs: dict[str, float], goal: str) -> float:
    return sum((probs.get(s, 0.0) - (1.0 if s == goal else 0.0)) ** 2 for s in SUITS)


def log_loss(probs: dict[str, float], goal: str) -> float:
    return -math.log(max(probs.get(goal, 0.0), 1e-9))


def run(args) -> dict:
    rng = random.Random(args.seed)
    client = make_client(args.backend, seed=args.seed, log_path=args.log, max_calls=args.max_calls, provider=args.provider)
    points = []
    for g in range(args.games):
        probe = Probe(p_snapshot=args.p_snapshot)
        opponents = [make_agent(s) for s in args.opponents.split(",")]
        result = play_game([probe] + opponents, rng, duration=args.duration)
        # Sample across the whole game, not just the first snapshots, which all fall in the opening seconds.
        snaps = rng.sample(probe.snapshots, min(args.per_game, len(probe.snapshots)))
        points += [(g, result.deck.goal, snap) for snap in sorted(snaps, key=lambda s: s["view"].t)]
    parts = set(filter(None, args.context.split(",")))

    def question(point):
        g, goal, snap = point
        view = snap["view"]
        menu = action_menu(view)
        state = add_context(view, snap["known"] if "known" in parts else None,
                            snap["posterior"] if "assist" in parts or args.assist else None,
                            log="log" in parts, summary="summary" in parts)
        answers = client.ask(state, {
            "goal": goal_question(),
            "action": action_question(menu, args.personality),
        })
        jev_goal = answers["goal"]["probabilities"]
        jev_label = hierarchical_choice(answers["action"].get("probabilities") or {answers["action"]["choice"]: 1.0})
        jev_action = menu.get(jev_label, menu["pass"])[0]
        edges = take_edges(view, snap["values"])
        best_edge = max([0.0] + list(edges.values()))
        jev_edge = edges.get(jev_action, 0.0)
        return {
            "game": g, "t": round(view.t, 2), "true_goal": goal,
            **{f"posterior_{s}": round(snap["posterior"][s], 4) for s in SUITS},
            **{f"jev_{s}": round(jev_goal.get(s, 0.0), 4) for s in SUITS},
            "brier_posterior": brier(snap["posterior"], goal),
            "brier_jev": brier(jev_goal, goal),
            "logloss_posterior": log_loss(snap["posterior"], goal),
            "logloss_jev": log_loss(jev_goal, goal),
            "tv_distance": 0.5 * sum(abs(jev_goal.get(s, 0.0) - snap["posterior"][s]) for s in SUITS),
            "top1_agree": max(jev_goal, key=jev_goal.get) == max(snap["posterior"], key=snap["posterior"].get),
            "classical_action": snap["classical_action"].label(),
            "jev_action": jev_label,
            "action_agree": jev_action == snap["classical_action"],
            "regret": best_edge - jev_edge,
        }

    # The frozen states are independent, so ask about all of them at once (the client paces requests).
    with ThreadPoolExecutor(max_workers=32) as pool:
        rows = list(pool.map(question, points))
    summary = summarise(rows, client)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "decisions.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        with open(os.path.join(args.out, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
    return summary


def summarise(rows, client) -> dict:
    def col(k):
        return [float(r[k]) for r in rows]

    out = {"backend": client.backend, "decision_points": len(rows), "jev_calls": client.calls, "cost_usd": round(client.cost_usd, 4)}
    for k in ("brier_posterior", "brier_jev", "logloss_posterior", "logloss_jev", "tv_distance", "top1_agree", "action_agree", "regret"):
        xs = col(k)
        lo, hi = bootstrap_ci(xs, groups=[r["game"] for r in rows])  # points in one game are not independent
        out[k] = {"mean": round(mean(xs), 4), "ci95": [round(lo, 4), round(hi, 4)]}
    out["brier_uniform"] = 0.75  # 4 suits at 25%: 3 * 0.0625 + 0.5625
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["mock", "jev"], default="mock")
    ap.add_argument("--provider", choices=["openrouter", "typesafe"], default=None,
                    help="Jev route (default: OpenRouter if OPENROUTER_API_KEY is set, else TypeSafe)")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--per-game", type=int, default=5, help="max decision points questioned per game")
    ap.add_argument("--p-snapshot", type=float, default=0.3)
    ap.add_argument("--opponents", default="fundamentalist,bottom_feeder,noise")
    ap.add_argument("--personality", default="neutral")
    ap.add_argument("--assist", action="store_true", help="give Jev the card-counting goal probabilities in the state")
    ap.add_argument("--context", default="", help="comma list of extra context parts: log, summary, known, assist "
                    "(log and summary are observed facts; known and assist are derived by code)")
    ap.add_argument("--duration", type=float, default=240.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-calls", type=int, default=500, help="hard cap on Jev API calls")
    ap.add_argument("--log", help="append every Jev request/response to this JSONL file")
    ap.add_argument("--out", help="directory for decisions.csv and summary.json")
    print(json.dumps(run(ap.parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()
