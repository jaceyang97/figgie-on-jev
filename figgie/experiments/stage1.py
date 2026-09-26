"""Stage 1, the decision layer: ask Jev about frozen moments of rule-based games.

Three steps, each a subcommand:

  moments  Play rule-based games in the stage 2 line-up (the tested seat against
           fundamentalist, bottom-feeder, noise trader) and freeze moments of the
           tested seat: for each rule set and trader type, `--per-type` moments at
           random times over the whole game, at most `--per-game` from one game,
           and only moments where the twin can act. For each moment we store the
           view, the true goal suit, the card-counting probabilities and the twin's
           action distribution (Monte Carlo over its random choices; an order the
           market would reject counts as pass; a price outside Jev's price ladder
           moves to the nearest ladder price).
  ask      Ask Jev about the moments. 1a: each type's moments with the algorithm
           persona, the behaviour persona and no persona. 1b: the fundamentalist
           seat's moments, no persona, five state versions, action and goal suit.
           1c: 50 moments asked twice. Every row keeps Jev's full probabilities;
           the client log keeps every request. Reruns skip rows already done.
  analyse  Similarity to the twin (overlap, all actions and actions only), buy /
           sell / take / pass shares, paired persona-minus-neutral differences with
           a game-level bootstrap and Holm correction (1a); log loss, Brier score,
           top-1 accuracy and calibration against card counting (1b);
           repeatability (1c).

Usage:
  python -m figgie.experiments.stage1 moments --out results/stage1/moments.jsonl
  python -m figgie.experiments.stage1 ask --study 1a --backend mock --moments results/stage1/moments.jsonl \\
      --out results/stage1/1a.jsonl
  python -m figgie.experiments.stage1 analyse --moments results/stage1/moments.jsonl \\
      --rows results/stage1/1a.jsonl results/stage1/1b.jsonl results/stage1/1c.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from ..agents import CLASSICAL, SPEEDS, TYPES, make_agent
from ..agents.jev_agent import PRICE_LADDER, action_menu, action_question, add_context, goal_question
from ..cards import SUITS
from ..engine import View, play_game
from ..jev import MAX_RPS, CallBudget, RateLimiter, make_client
from ..market import Action, Order, Quote, Trade
from ..records import write_run_header
from ..stats import bootstrap_ci, mean
from .tournament import game_rng, game_seed

FIELD = ("fundamentalist", "bottom_feeder", "noise")
STATE_VERSIONS = ("", "summary", "log,summary", "known", "assist")


# --- views to and from JSON -------------------------------------------------------------------------------


def view_to_json(v: View) -> dict:
    q = lambda x: None if x is None else [x.price, x.player]  # noqa: E731
    return {
        "t": v.t, "t_end": v.t_end, "me": v.me, "hand": v.hand, "chips": v.chips,
        "bids": {s: q(v.bids[s]) for s in SUITS}, "asks": {s: q(v.asks[s]) for s in SUITS},
        "trades": [[t.t, t.suit, t.price, t.buyer, t.seller, t.aggressor] for t in v.trades],
        "orders": [[o.t, o.player, o.side, o.suit, o.price, o.oid] for o in v.orders],
        "mechanism": v.mechanism,
        "depth": None if v.depth is None else {s: {k: [q(x) for x in xs] for k, xs in d.items()} for s, d in v.depth.items()},
        "my_orders": [[o.t, o.player, o.side, o.suit, o.price, o.oid] for o in v.my_orders],
        "progress": v.progress,
    }


def view_from_json(d: dict) -> View:
    q = lambda x: None if x is None else Quote(*x)  # noqa: E731
    return View(
        t=d["t"], t_end=d["t_end"], me=d["me"], hand=d["hand"], chips=d["chips"],
        bids={s: q(d["bids"][s]) for s in SUITS}, asks={s: q(d["asks"][s]) for s in SUITS},
        trades=tuple(Trade(*x) for x in d["trades"]), orders=tuple(Order(*x) for x in d["orders"]),
        mechanism=d["mechanism"],
        depth=None if d["depth"] is None else {s: {k: [q(x) for x in xs] for k, xs in dd.items()} for s, dd in d["depth"].items()},
        my_orders=tuple(Order(*x) for x in d["my_orders"]), progress=d["progress"],
    )


# --- the twin's action distribution -----------------------------------------------------------------------


def to_menu_label(action: Action, menu: dict) -> tuple[str, bool]:
    """The menu label for a twin action, and whether it was moved. Rejected or impossible actions are pass."""
    label = action.label()
    if label in menu:
        return label, False
    if action.kind in ("bid", "ask") and action.price not in PRICE_LADDER:
        nearest = min(PRICE_LADDER, key=lambda p: (abs(p - action.price), p))
        moved = Action(action.kind, action.suit, nearest).label()
        if moved in menu:
            return moved, True
    return "pass", False


def twin_distribution(view: View, twin, n: int) -> tuple[dict[str, float], float]:
    menu = action_menu(view)
    counts, moved = Counter(), 0
    for i in range(n):
        t = copy.copy(twin)
        t.rng = random.Random(i)
        label, m = to_menu_label(t.decide(view), menu)
        counts[label] += 1
        moved += m
    return {k: c / n for k, c in counts.items()}, moved / n


class Freezer:
    """Wraps the tested seat's agent and keeps copies of it at some of its wake-ups."""

    def __init__(self, inner, p: float, rng: random.Random):
        self.inner, self.p, self.rng, self.snaps = inner, p, rng, []

    def __getattr__(self, k):
        return getattr(self.inner, k)

    def decide(self, view):
        if self.rng.random() < self.p:
            self.snaps.append((view, copy.deepcopy(self.inner)))
        return self.inner.decide(view)


def make_moments(args) -> None:
    rng = random.Random(args.seed)
    speed = SPEEDS[args.speed]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n_out = 0
    with open(args.out, "w") as f:
        for mech in args.mechanisms.split(","):
            for typ in args.types.split(","):
                got, g = 0, 0
                while got < args.per_type and g < args.max_games:
                    specs = [typ, *FIELD]
                    order = specs[g % 4:] + specs[: g % 4]
                    seat = (0 - g % 4) % 4
                    agents = [make_agent(s, speed=speed) for s in order]
                    agents[seat] = Freezer(agents[seat], args.p_freeze, random.Random(rng.random()))
                    res = play_game(agents, game_rng(args.seed, g), mechanism=mech, tested_seat=seat)
                    snaps = agents[seat].snaps
                    rng.shuffle(snaps)
                    kept = 0
                    for view, twin in snaps:
                        if kept >= args.per_game or got >= args.per_type:
                            break
                        dist, moved = twin_distribution(view, twin, args.mc)
                        if dist.get("pass", 0.0) >= 1.0:
                            continue  # the twin cannot act here
                        f.write(json.dumps({
                            "id": f"{mech}/{typ}/{g}/{round(view.t, 3)}", "mechanism": mech, "type": typ, "game": g,
                            "seed": game_seed(args.seed, g), "seat": seat, "t": view.t, "goal": res.deck.goal,
                            "known": twin.counter.known(), "card_counting": twin.counter.goal_probabilities(),
                            "twin_dist": dist, "twin_moved_share": moved, "menu_size": len(action_menu(view)),
                            "view": view_to_json(view),
                        }) + "\n")
                        kept += 1
                        got += 1
                        n_out += 1
                    g += 1
                print(f"{mech} {typ}: {got} moments from {g} games")
    write_run_header(os.path.dirname(os.path.abspath(args.out)), "stage1-moments", vars(args))
    print(f"wrote {n_out} moments to {args.out}")


# --- asking Jev -------------------------------------------------------------------------------------------


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def plan_jobs(moments: list[dict], study: str, context: str, n_repeat: int) -> list[dict]:
    jobs = []
    if study == "1a":
        for m in moments:
            for persona, wording in ((m["type"], "algorithm"), (m["type"] + "-desc", "behaviour"), ("neutral", "none")):
                jobs.append({"moment": m["id"], "persona": persona, "wording": wording, "context": context, "repeat": 0,
                             "goal": False})
    elif study == "1b":
        for m in moments:
            if m["type"] != "fundamentalist":
                continue
            for version in STATE_VERSIONS:
                jobs.append({"moment": m["id"], "persona": "neutral", "wording": "none", "context": version,
                             "repeat": 0, "goal": True})
    elif study == "1c":
        chosen = [m for m in moments if m["type"] == "fundamentalist"][:n_repeat]
        for m in chosen:
            for r in (0, 1):
                jobs.append({"moment": m["id"], "persona": "neutral", "wording": "none", "context": context, "repeat": r,
                             "goal": True})
    else:
        raise SystemExit(f"unknown study {study!r}")
    return jobs


def ask(args) -> None:
    moments = {m["id"]: m for m in load_jsonl(args.moments)}
    jobs = plan_jobs(list(moments.values()), args.study, args.context, args.repeat_moments)
    key = lambda j: (j["moment"], j["persona"], j["context"], j["repeat"])  # noqa: E731
    done = {key(r) for r in load_jsonl(args.out)} if os.path.exists(args.out) else set()
    jobs = [j for j in jobs if key(j) not in done]
    print(f"{args.study}: {len(jobs)} requests to make ({len(done)} already done)")
    if not jobs:
        return
    out_dir = os.path.dirname(os.path.abspath(args.out))
    write_run_header(out_dir, f"stage1-{args.study}", vars(args))
    client = make_client(args.backend, provider=args.provider, limiter=RateLimiter(args.max_rps),
                         budget=CallBudget(args.max_calls), log_path=args.log or args.out.replace(".jsonl", ".log.jsonl"))
    lock = __import__("threading").Lock()

    def run(job):
        m = moments[job["moment"]]
        view = view_from_json(m["view"])
        parts = set(filter(None, job["context"].split(",")))
        state = add_context(view, m["known"] if "known" in parts else None,
                            m["card_counting"] if "assist" in parts else None,
                            log="log" in parts, summary="summary" in parts)
        menu = action_menu(view)
        questions = {"action": action_question(menu, job["persona"], view.mechanism)}
        if job["goal"]:
            questions["goal"] = goal_question(view.mechanism)
        meta = {"study": args.study, **job, "mechanism": m["mechanism"], "type": m["type"], "game": m["game"]}
        answers = client.ask(state, questions, meta=meta)
        a = answers["action"]
        row = {**job, "study": args.study, "mechanism": m["mechanism"], "type": m["type"], "game": m["game"],
               "action_probs": a.get("probabilities") or {a["choice"]: 1.0}}
        if "goal" in answers:
            g = answers["goal"]
            row["goal_probs"] = g.get("probabilities") or {g["choice"]: 1.0}
        with lock, open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run, jobs))
    print(f"calls {client.calls}, estimated cost US${client.cost_usd:.3f}")


# --- analysis ---------------------------------------------------------------------------------------------


def overlap(p: dict, q: dict) -> float:
    return sum(min(p.get(k, 0.0), q.get(k, 0.0)) for k in set(p) | set(q))


def acting(d: dict) -> dict:
    d = {k: v for k, v in d.items() if k != "pass"}
    z = sum(d.values())
    return {k: v / z for k, v in d.items()} if z > 0 else {}


def shares(d: dict) -> dict:
    out = {"buy": 0.0, "sell": 0.0, "take": 0.0, "pass": 0.0, "cancel": 0.0}
    for k, v in d.items():
        kind = k.split("_")[0]
        if kind in ("bid", "buy"):
            out["buy"] += v
        if kind in ("ask", "sell"):
            out["sell"] += v
        if kind in ("buy", "sell"):
            out["take"] += v
        if kind in ("pass", "cancel"):
            out[kind] += v
    return out


def boot_p(xs, groups, n_boot: int = 4000, seed: int = 0) -> float:
    """Two-sided bootstrap p-value for mean(xs) = 0, resampling whole groups."""
    rng = random.Random(seed)
    clusters = defaultdict(list)
    for x, g in zip(xs, groups):
        clusters[g].append(x)
    cl = list(clusters.values())
    below = 0
    for _ in range(n_boot):
        draw = [x for _ in cl for x in cl[rng.randrange(len(cl))]]
        below += (sum(draw) / len(draw)) <= 0
    frac = below / n_boot
    return min(1.0, 2 * min(frac, 1 - frac))


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def summary_ci(xs, groups):
    lo, hi = bootstrap_ci(xs, n_boot=4000, groups=groups)
    return {"mean": round(mean(xs), 4), "ci95": [round(lo, 4), round(hi, 4)], "n": len(xs)}


def analyse_1a(rows, moments) -> dict:
    by = defaultdict(dict)  # (mech, type, moment) -> wording -> row
    for r in rows:
        by[(r["mechanism"], r["type"], r["moment"])][r["wording"]] = r
    cells = defaultdict(lambda: defaultdict(list))
    for (mech, typ, mid), w in by.items():
        if "none" not in w:
            continue
        twin = moments[mid]["twin_dist"]
        base_all, base_act = overlap(w["none"]["action_probs"], twin), overlap(acting(w["none"]["action_probs"]), acting(twin))
        for wording, r in w.items():
            c = cells[(mech, typ, wording)]
            c["overlap"].append(overlap(r["action_probs"], twin))
            c["overlap_acting"].append(overlap(acting(r["action_probs"]), acting(twin)))
            c["game"].append(moments[mid]["game"])
            for k, v in shares(r["action_probs"]).items():
                c[f"share_{k}"].append(v)
            if wording != "none":
                c["diff"].append(c["overlap"][-1] - base_all)
                c["diff_acting"].append(c["overlap_acting"][-1] - base_act)
    out, pvals = {}, {}
    for (mech, typ, wording), c in sorted(cells.items()):
        key = f"{mech}/{typ}/{wording}"
        res = {k: summary_ci(v, c["game"]) for k, v in c.items() if k != "game" and v}
        twin_shares = [shares(moments[m]["twin_dist"]) for (mm, tt, m) in by if mm == mech and tt == typ]
        res["twin_share"] = {k: round(mean([s[k] for s in twin_shares]), 4) for k in twin_shares[0]} if twin_shares else {}
        if c["diff"]:
            pvals[key] = boot_p(c["diff"], c["game"])
            res["p_diff"] = round(pvals[key], 5)
        out[key] = res
    for k, p in holm(pvals).items():
        out[k]["p_diff_holm"] = round(p, 5)
    return out


def brier(p, goal):
    return sum((p.get(s, 0.0) - (s == goal)) ** 2 for s in SUITS)


def logloss(p, goal):
    return -math.log(max(p.get(goal, 0.0), 1e-9))


def analyse_1b(rows, moments) -> dict:
    cells = defaultdict(lambda: defaultdict(list))
    calib = defaultdict(lambda: [[0, 0.0, 0] for _ in range(10)])  # per cell: bins of [n, sum p, hits]
    for r in rows:
        if "goal_probs" not in r:
            continue
        m = moments[r["moment"]]
        goal, cc = m["goal"], m["card_counting"]
        c = cells[(r["mechanism"], r["context"] or "basic")]
        jl, cl = logloss(r["goal_probs"], goal), logloss(cc, goal)
        c["logloss_jev"].append(jl)
        c["logloss_card_counting"].append(cl)
        c["logloss_diff"].append(jl - cl)
        c["brier_jev"].append(brier(r["goal_probs"], goal))
        c["brier_card_counting"].append(brier(cc, goal))
        c["top1_jev"].append(float(max(r["goal_probs"], key=r["goal_probs"].get) == goal))
        c["top1_card_counting"].append(float(max(cc, key=cc.get) == goal))
        c["t"].append(m["t"])
        c["game"].append(m["game"])
        for s in SUITS:
            p = r["goal_probs"].get(s, 0.0)
            b = calib[(r["mechanism"], r["context"] or "basic")][min(9, int(p * 10))]
            b[0] += 1
            b[1] += p
            b[2] += s == goal
    out = {}
    for (mech, ctx), c in sorted(cells.items()):
        res = {k: summary_ci(v, c["game"]) for k, v in c.items() if k not in ("game", "t")}
        res["logloss_uniform"] = round(math.log(4), 4)
        res["brier_uniform"] = 0.75
        res["calibration"] = [{"bin": i / 10, "n": n, "mean_p": round(sp / n, 4), "hit_rate": round(h / n, 4)}
                              for i, (n, sp, h) in enumerate(calib[(mech, ctx)]) if n]
        # H1c: does Jev's log loss fall later in the game? Split at the median time.
        ts = sorted(c["t"])
        mid = ts[len(ts) // 2] if ts else 0
        early = [x for x, t in zip(c["logloss_jev"], c["t"]) if t < mid]
        late = [x for x, t in zip(c["logloss_jev"], c["t"]) if t >= mid]
        res["logloss_jev_early_late"] = [round(mean(early), 4), round(mean(late), 4)]
        out[f"{mech}/{ctx}"] = res
    return out


def tv(p, q):
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in set(p) | set(q))


def analyse_1c(rows) -> dict:
    pairs = defaultdict(dict)
    for r in rows:
        pairs[(r["moment"], r["context"])][r["repeat"]] = r
    act, goal, same = [], [], []
    for d in pairs.values():
        if 0 in d and 1 in d:
            a0, a1 = d[0]["action_probs"], d[1]["action_probs"]
            act.append(tv(a0, a1))
            same.append(float(max(a0, key=a0.get) == max(a1, key=a1.get)))
            if "goal_probs" in d[0] and "goal_probs" in d[1]:
                goal.append(tv(d[0]["goal_probs"], d[1]["goal_probs"]))
    return {"pairs": len(act), "action_tv_mean": round(mean(act), 5) if act else None,
            "action_top_same": round(mean(same), 4) if same else None,
            "goal_tv_mean": round(mean(goal), 5) if goal else None}


def analyse(args) -> None:
    moments = {m["id"]: m for m in load_jsonl(args.moments)}
    rows = [r for path in args.rows for r in load_jsonl(path)]
    report = {
        "1a": analyse_1a([r for r in rows if r["study"] == "1a"], moments),
        "1b": analyse_1b([r for r in rows if r["study"] == "1b"], moments),
        "1c": analyse_1c([r for r in rows if r["study"] == "1c"]),
        "twin_moved_share": round(mean([m["twin_moved_share"] for m in moments.values()]), 5),
    }
    text = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    print(text)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("moments")
    m.add_argument("--mechanisms", default="A,B")
    m.add_argument("--types", default=",".join(TYPES))
    m.add_argument("--per-type", type=int, default=100)
    m.add_argument("--per-game", type=int, default=3)
    m.add_argument("--p-freeze", type=float, default=0.1, help="chance of keeping a candidate at each wake-up")
    m.add_argument("--mc", type=int, default=4000, help="Monte Carlo draws for the twin's distribution")
    m.add_argument("--max-games", type=int, default=2000)
    m.add_argument("--speed", default="equal", choices=sorted(SPEEDS))
    m.add_argument("--seed", type=int, default=1)
    m.add_argument("--out", required=True)
    a = sub.add_parser("ask")
    a.add_argument("--study", choices=["1a", "1b", "1c"], required=True)
    a.add_argument("--moments", required=True)
    a.add_argument("--context", default="log,summary", help="state version for 1a and 1c")
    a.add_argument("--repeat-moments", type=int, default=50)
    a.add_argument("--backend", choices=["mock", "jev"], default="mock")
    a.add_argument("--provider", choices=["openrouter", "typesafe"], default=None)
    a.add_argument("--max-calls", type=int, default=5000)
    a.add_argument("--max-rps", type=float, default=MAX_RPS)
    a.add_argument("--workers", type=int, default=16)
    a.add_argument("--log", default=None, help="client log (default: <out without .jsonl>.log.jsonl)")
    a.add_argument("--out", required=True)
    n = sub.add_parser("analyse")
    n.add_argument("--moments", required=True)
    n.add_argument("--rows", nargs="+", required=True)
    n.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    {"moments": make_moments, "ask": ask, "analyse": analyse}[args.cmd](args)


if __name__ == "__main__":
    main()
