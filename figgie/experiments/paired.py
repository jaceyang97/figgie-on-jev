"""Stage 2 analysis: Jev minus its rule-based twin, in the same seat and the same games.

Reads the output of figgie.experiments.sweep. For each rule set and condition,
it pairs game g of the Jev arm with game g of the twin arm (same deal, same
seat, same opponent wake-up times) and computes d_g = profit(Jev) - profit(twin)
at the tested seat. Only games that both arms finished are used.

Reports (95% CIs from a cluster bootstrap over games, 4,000 resamples; one
game number is one cluster in every condition, because all conditions share
the seeds):
  - mean d for each condition (RQ4 per condition)
  - mean d over all conditions in each rule set (RQ4)
  - the rule-set effect A - B and the wording effect (behaviour - algorithm)
  - the regression d = b0 + bA*[A] + bD*[behaviour wording] + sum_k g_k*[type k]
    + bAD*[A]*[behaviour wording], and the same with the number of goal cards in
    the starting hand as a covariate (secondary)
  - secondary measures per condition, for the Jev arm and the twin arm: trades
    per game, the tested seat's trades and rejected orders, mispricing (goal
    card 10, other cards 0), and Jev's goal-suit belief (log loss and Brier
    score next to card counting at the same moments)

Usage:
  python -m figgie.experiments.paired --run results/stage2 --out results/stage2/paired.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict

from ..cards import SUITS
from ..records import starting_goal_cards
from ..stats import mean

N_BOOT = 4000


def load_games(path: str) -> dict[int, dict]:
    out = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    out[rec["game"]] = rec
    return out


def pairs(run_dir: str) -> list[dict]:
    """One row per (rule set, condition, game) with d and the covariates."""
    with open(os.path.join(run_dir, "run.json")) as f:
        args = json.load(f)["args"]
    rows = []
    for mech in args["mechanisms"].split(","):
        games_dir = os.path.join(run_dir, mech, "games")
        twins = {t: load_games(os.path.join(games_dir, f"[{t}].jsonl")) for t in args["twins"]}
        for persona in args["personas"]:
            jev = load_games(os.path.join(games_dir, f"{persona}.jsonl"))
            twin = twins[args["twin_of"][persona]]
            for g in sorted(set(jev) & set(twin)):
                a, b = jev[g], twin[g]
                seat = a["tested_seat"]
                if b["tested_seat"] != seat or a["seed"] != b["seed"]:
                    raise ValueError(f"{mech}/{persona} game {g}: the two arms are not paired")
                base = persona.removesuffix("-desc")
                rows.append({
                    "mechanism": mech, "condition": persona, "type": base,
                    "wording": "none" if base == "neutral" else ("behaviour" if persona.endswith("-desc") else "algorithm"),
                    "game": g, "seat": seat, "d": a["pnl"][seat] - b["pnl"][seat],
                    "pnl_jev": a["pnl"][seat], "pnl_twin": b["pnl"][seat],
                    "goal_cards_start": starting_goal_cards(a, seat), "jev": a, "twin": b,
                })
    return rows


def cluster_boot(rows: list[dict], stat, n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, list[float]]:
    """stat(rows) and its 95% percentile CI, resampling whole game numbers."""
    by_game = defaultdict(list)
    for r in rows:
        by_game[r["game"]].append(r)
    games = sorted(by_game)
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        sample = [r for _ in games for r in by_game[games[rng.randrange(len(games))]]]
        try:
            draws.append(stat(sample))
        except (ZeroDivisionError, ValueError):
            continue
    draws.sort()
    if not draws:
        return stat(rows), [float("nan"), float("nan")]
    return stat(rows), [draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]]


def fmt(est, ci, nd=2):
    return {"estimate": round(est, nd), "ci95": [round(x, nd) for x in ci]}


# --- least squares without dependencies ---------------------------------------------------------------------


def solve(a: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[p][c]) < 1e-12:
            raise ValueError("singular design")
        m[c], m[p] = m[p], m[c]
        for r in range(n):
            if r != c:
                f = m[r][c] / m[c][c]
                m[r] = [x - f * y for x, y in zip(m[r], m[c])]
    return [m[i][n] / m[i][i] for i in range(n)]


def ols(x: list[list[float]], y: list[float]) -> list[float]:
    k = len(x[0])
    xtx = [[sum(r[i] * r[j] for r in x) for j in range(k)] for i in range(k)]
    xty = [sum(r[i] * yy for r, yy in zip(x, y)) for i in range(k)]
    return solve(xtx, xty)


def design(rows, types, covariate: bool):
    names = ["intercept", "rule_set_A", "behaviour_wording", "A_x_behaviour"] + [f"type_{t}" for t in types]
    if covariate:
        names.append("goal_cards_start")
    x = []
    for r in rows:
        a, d = float(r["mechanism"] == "A"), float(r["wording"] == "behaviour")
        row = [1.0, a, d, a * d] + [float(r["type"] == t) for t in types]
        if covariate:
            row.append(float(r["goal_cards_start"]))
        x.append(row)
    return names, x


def regression(rows, covariate: bool) -> dict:
    types = sorted({r["type"] for r in rows} - {"neutral"})  # neutral is the reference
    names, _ = design(rows, types, covariate)
    out = {}
    for i, name in enumerate(names):
        def stat(rs, i=i):
            _, x = design(rs, types, covariate)
            return ols(x, [r["d"] for r in rs])[i]
        out[name] = fmt(*cluster_boot(rows, stat, n_boot=1000))
    return out


# --- secondary measures -------------------------------------------------------------------------------------


def mispricing(rec) -> float:
    xs = [abs(price - (10 if suit == rec["goal"] else 0)) for _, suit, price, *_ in rec["trades"]]
    return mean(xs) if xs else float("nan")


def belief_scores(rec, seat) -> dict | None:
    trace = rec.get("jev_trace", {}).get(str(seat))
    if not trace:
        return None
    goal = rec["goal"]
    ll, br, ll_cc, br_cc = [], [], [], []
    for t in trace:
        if "goal" not in t:
            continue
        p = t["goal"]
        ll.append(-math.log(max(p.get(goal, 0.0), 1e-9)))
        br.append(sum((p.get(s, 0.0) - (s == goal)) ** 2 for s in SUITS))
        if "card_counting" in t:
            c = t["card_counting"]
            ll_cc.append(-math.log(max(c.get(goal, 0.0), 1e-9)))
            br_cc.append(sum((c.get(s, 0.0) - (s == goal)) ** 2 for s in SUITS))
    if not ll:
        return None
    return {"logloss": mean(ll), "brier": mean(br), "logloss_cc": mean(ll_cc) if ll_cc else None,
            "brier_cc": mean(br_cc) if br_cc else None}


def secondary(rows) -> dict:
    out = {}
    for arm in ("jev", "twin"):
        recs = [(r[arm], r["seat"]) for r in rows]
        out[arm] = {
            "trades_per_game": round(mean([len(rec["trades"]) for rec, _ in recs]), 2),
            "seat_trades": round(mean([rec["seat_trades"][s] for rec, s in recs]), 2),
            "seat_rejected": round(mean([rec["rejected"][s] for rec, s in recs]), 2),
            "seat_passes": round(mean([rec["passes"][s] for rec, s in recs]), 2),
            "mispricing": round(mean([m for m in (mispricing(rec) for rec, _ in recs) if not math.isnan(m)]), 3),
        }
    beliefs = [b for b in (belief_scores(r["jev"], r["seat"]) for r in rows) if b]
    if beliefs:
        out["jev_belief"] = {k: round(mean([b[k] for b in beliefs if b[k] is not None]), 4)
                             for k in ("logloss", "brier", "logloss_cc", "brier_cc")}
        out["jev_belief"].update({"logloss_uniform": round(math.log(4), 4), "brier_uniform": 0.75})
    return out


def analyse(rows) -> dict:
    mean_d = lambda rs: mean([r["d"] for r in rs])  # noqa: E731
    report = {"n_pairs": len(rows), "conditions": {}, "rule_sets": {}, "effects": {}}
    by_cond = defaultdict(list)
    for r in rows:
        by_cond[(r["mechanism"], r["condition"])].append(r)
    for (mech, cond), rs in sorted(by_cond.items()):
        report["conditions"][f"{mech}/{cond}"] = {
            "games": len(rs), "mean_d": fmt(*cluster_boot(rs, mean_d)),
            "mean_pnl_jev": round(mean([r["pnl_jev"] for r in rs]), 2),
            "mean_pnl_twin": round(mean([r["pnl_twin"] for r in rs]), 2),
            "sd_d": round((sum((r["d"] - mean_d(rs)) ** 2 for r in rs) / max(1, len(rs) - 1)) ** 0.5, 2),
            **secondary(rs),
        }
    for mech in sorted({r["mechanism"] for r in rows}):
        rs = [r for r in rows if r["mechanism"] == mech]
        report["rule_sets"][mech] = {"pairs": len(rs), "mean_d": fmt(*cluster_boot(rs, mean_d))}
    if len(report["rule_sets"]) == 2:
        report["effects"]["A_minus_B"] = fmt(*cluster_boot(rows, lambda rs: mean_d([r for r in rs if r["mechanism"] == "A"])
                                                           - mean_d([r for r in rs if r["mechanism"] == "B"])))
    typed = [r for r in rows if r["wording"] != "none"]
    if {r["wording"] for r in typed} == {"algorithm", "behaviour"}:
        report["effects"]["behaviour_minus_algorithm"] = fmt(*cluster_boot(
            typed, lambda rs: mean_d([r for r in rs if r["wording"] == "behaviour"])
            - mean_d([r for r in rs if r["wording"] == "algorithm"])))
    try:
        report["regression"] = regression(rows, covariate=False)
        report["regression_with_goal_cards"] = regression(rows, covariate=True)
    except (ValueError, IndexError) as e:  # too few conditions for the full model
        report["regression"] = {"error": str(e)}
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="the sweep output directory")
    ap.add_argument("--out", default=None, help="write the report here as JSON (default: <run>/paired.json)")
    ap.add_argument("--pairs-csv", default=None, help="also write one row per pair (default: <run>/pairs.csv)")
    args = ap.parse_args(argv)
    rows = pairs(args.run)
    if not rows:
        raise SystemExit("no paired games found")
    report = analyse(rows)
    out = args.out or os.path.join(args.run, "paired.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    csv_path = args.pairs_csv or os.path.join(args.run, "pairs.csv")
    cols = ["mechanism", "condition", "type", "wording", "game", "seat", "d", "pnl_jev", "pnl_twin", "goal_cards_start"]
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"{'condition':<32}{'games':>6}{'mean d':>9}{'95% CI':>18}")
    for k, v in report["conditions"].items():
        lo, hi = v["mean_d"]["ci95"]
        print(f"{k:<32}{v['games']:>6}{v['mean_d']['estimate']:>9.1f}{f'[{lo:.0f}, {hi:.0f}]':>18}")
    for k, v in {**report["rule_sets"], **report["effects"]}.items():
        est = v["mean_d"] if "mean_d" in v else v
        lo, hi = est["ci95"]
        print(f"{k:<32}{'':>6}{est['estimate']:>9.1f}{f'[{lo:.0f}, {hi:.0f}]':>18}")
    print(f"wrote {out} and {csv_path}")


if __name__ == "__main__":
    main()
