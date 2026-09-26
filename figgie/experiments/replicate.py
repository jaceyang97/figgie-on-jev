"""Stage 0: replicate the paper with rule-based traders only, and measure the profit noise.

No Jev calls; everything here is free.

Replication. The paper's line-ups (Ozerov et al. 2021, figures 3-5 and tables
2-3), 100 games each, in three settings:
  A-10k   rule set A (order book), the game ends after 10,000 events (the paper)
  A-240s  rule set A, 240-second games
  B-240s  rule set B (open outcry), 240-second games
Each setting runs two times: with the paper's speeds (1 decision per second,
no latency) and with the equal speed that stage 2 uses (0.5 per second,
0.3 s). The table 3 line-up has its own speeds (one fundamentalist with
latency 0 and three with the paper's latency of 100, which is a delay of
2 x 100 = 200 in this engine), so it runs once, in A-10k, as in the paper.

The report gives, per line-up and seat, the mean profit, the paper's value
(mean cash + payout = profit + 400, because the paper's players start with
350 after the ante and the pot pays 50 on average), and the standard
deviations of wealth, cash and payout (the paper's table 2). For table 3 it
gives bootstrap 95% CIs of each slow trader minus the fast trader, for wealth,
cash and payout, as in the paper.

Noise check. For each rule set and each trader type, the stage 2 line-up
(type, fundamentalist, bottom-feeder, noise trader) with the type in the tested
seat, 100 games at the equal speed. It reports the standard deviation of the
tested seat's profit and of the paired difference to the fundamentalist in the
same seat and games, and the minimum detectable effect for 40 games.

Output: <out>/<setting>/<line-up>/games.jsonl and summary.json for every run,
and <out>/report.json and <out>/report.md with all tables.

Usage:
  python -m figgie.experiments.replicate --out results/stage0
  python -m figgie.experiments.replicate --games 10 --out /tmp/stage0   # quick check
"""

from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

from ..agents import TYPES
from ..records import write_run_header
from ..stats import bootstrap_ci, mean
from .tournament import run as run_tournament

F, B, N = "fundamentalist", "bottom_feeder", "noise"
FAST, SLOW = "fundamentalist@1/0", "fundamentalist@1/200"

LINEUPS = {
    "fig3_f_3n": [F, N, N, N],
    "fig3_2f_2n": [F, F, N, N],
    "fig3_3f_n": [F, F, F, N],
    "fig4_f_b_2n": [F, B, N, N],
    "fig4_2f_b_n": [F, F, B, N],
    "fig4_3f_b": [F, F, F, B],
    "fig5_f_2b_n": [F, B, B, N],
    "fig5_2f_2b": [F, F, B, B],
    "tab2_4n": [N, N, N, N],
    "tab2_4f": [F, F, F, F],
    "tab2_4b": [B, B, B, B],
}
TABLE3 = [FAST, SLOW, SLOW, SLOW]
SETTINGS = {  # name -> (rule set, duration, max events)
    "A-10k": ("A", 240.0, 10_000),
    "A-240s": ("A", 240.0, None),
    "B-240s": ("B", 240.0, None),
}
SPEED_NAMES = ("paper", "equal")
PAPER_VALUE_OFFSET = 400


def _run(job: dict) -> tuple[str, dict]:
    args = SimpleNamespace(backend="mock", provider=None, lineup=",".join(job["lineup"]), games=job["games"],
                           duration=job["duration"], max_events=job["max_events"], speed=job["speed"],
                           decode="sample", seed=job["seed"], mechanism=job["mechanism"], tested=job.get("tested"),
                           condition=job["name"], run_id=job["run_id"], max_calls=None, log=None, workers=1,
                           out=job["out"], checkpoint=None)
    return job["name"], run_tournament(args)


def jobs_for(args) -> list[dict]:
    jobs = []
    for setting, (mech, duration, max_events) in SETTINGS.items():
        for speed in SPEED_NAMES:
            for name, lineup in LINEUPS.items():
                jobs.append({"name": f"replication/{setting}-{speed}/{name}", "lineup": lineup, "mechanism": mech,
                             "duration": duration, "max_events": max_events, "speed": speed})
    mech, duration, max_events = SETTINGS["A-10k"]
    jobs.append({"name": "replication/A-10k-own/tab3_fast_3slow", "lineup": TABLE3, "mechanism": mech,
                 "duration": duration, "max_events": max_events, "speed": "paper"})
    for mech in ("A", "B"):
        for t in TYPES:
            jobs.append({"name": f"noise/{mech}/{t}", "lineup": [t, F, B, N], "mechanism": mech, "duration": 240.0,
                         "max_events": None, "speed": "equal", "tested": 0})
    for j in jobs:
        j.update(games=args.noise_games if j["name"].startswith("noise/") else args.games, seed=args.seed,
                 run_id=os.path.basename(os.path.normpath(args.out)), out=os.path.join(args.out, j["name"]))
    return jobs


def load_records(path: str) -> list[dict]:
    with open(os.path.join(path, "games.jsonl")) as f:
        return sorted((json.loads(line) for line in f if line.strip()), key=lambda r: r["game"])


def sd(xs) -> float:
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else float("nan")


def lineup_index(rec: dict, seat: int) -> int:
    return (seat + rec["game"] % 4) % 4


def table3(records: list[dict]) -> dict:
    """Slow minus fast, per slow seat (the paper's table 3), with bootstrap 95% CIs over games."""
    out = {}
    for slow_idx in (1, 2, 3):
        diffs = {"wealth": [], "cash": [], "payout": []}
        for rec in records:
            by_idx = {lineup_index(rec, s): s for s in range(4)}
            fast, slow = by_idx[0], by_idx[slow_idx]
            diffs["wealth"].append(rec["pnl"][slow] - rec["pnl"][fast])
            diffs["cash"].append(rec["final_chips"][slow] - rec["final_chips"][fast])
            diffs["payout"].append(rec["payout"][slow] - rec["payout"][fast])
        out[f"slow{slow_idx - 1}"] = {k: {"mean": round(mean(v), 2), "ci95": [round(x, 2) for x in bootstrap_ci(v, 4000)]}
                                      for k, v in diffs.items()}
    return out


def noise_check(out_dir: str, games_for_mde: int) -> dict:
    res = {}
    for mech in ("A", "B"):
        tested = {t: {r["game"]: r["pnl"][r["tested_seat"]] for r in load_records(os.path.join(out_dir, "noise", mech, t))}
                  for t in TYPES}
        for t in TYPES:
            row = {"sd_pnl": round(sd(list(tested[t].values())), 1), "mean_pnl": round(mean(list(tested[t].values())), 1)}
            if t != F:
                common = sorted(set(tested[t]) & set(tested[F]))
                d = [tested[t][g] - tested[F][g] for g in common]
                row["sd_diff_vs_fundamentalist"] = round(sd(d), 1)
                row[f"mde_{games_for_mde}_games"] = round(2.8 * sd(d) / math.sqrt(games_for_mde), 1)
            res[f"{mech}/{t}"] = row
    return res


def markdown(report: dict) -> str:
    lines = ["# Stage 0 report", "", "Profit = chips at the end minus chips at the start. Paper value = profit + 400 "
             "(mean cash + payout, as in the paper's figures). SDs are over games.", ""]
    for setting, runs in report["replication"].items():
        lines += [f"## {setting}", "", "| line-up | seat | profit | 95% CI | paper value | SD wealth | SD cash | SD payout "
                  "| trades/game |", "|---|---|---|---|---|---|---|---|---|"]
        for name, s in runs.items():
            for seat, a in s["agents"].items():
                lo, hi = a["ci95"]
                lines.append(f"| {name} | {seat} | {a['mean_pnl']:.1f} | [{lo:.0f}, {hi:.0f}] | "
                             f"{a['mean_pnl'] + PAPER_VALUE_OFFSET:.0f} | {a['sd_pnl']:.1f} | {a['sd_cash']:.1f} | "
                             f"{a['sd_payout']:.1f} | {s['market']['trades_per_game']:.1f} |")
        lines.append("")
    lines += ["## Table 3: slow minus fast (A-10k)", "", "| competitor | wealth | cash | payout |", "|---|---|---|---|"]
    for k, v in report["table3"].items():
        cell = lambda x: f"{x['mean']:.1f} [{x['ci95'][0]:.1f}, {x['ci95'][1]:.1f}]"  # noqa: E731
        lines.append(f"| {k} | {cell(v['wealth'])} | {cell(v['cash'])} | {cell(v['payout'])} |")
    lines += ["", "## Noise check (stage 2 line-up, tested seat, equal speed)", "",
              "| rule set / type | mean profit | SD profit | SD of difference to fundamentalist | MDE |", "|---|---|---|---|---|"]
    for k, v in report["noise"].items():
        mde = next((x for kk, x in v.items() if kk.startswith("mde_")), "")
        lines.append(f"| {k} | {v['mean_pnl']} | {v['sd_pnl']} | {v.get('sd_diff_vs_fundamentalist', '')} | {mde} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=100, help="games per replication line-up (the paper: 100)")
    ap.add_argument("--noise-games", type=int, default=100, help="games per noise-check run")
    ap.add_argument("--mde-games", type=int, default=40, help="games per condition in stage 2, for the MDE")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    write_run_header(args.out, "stage0", vars(args))
    jobs = jobs_for(args)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        summaries = dict(pool.map(_run, jobs))

    report = {"replication": {}, "table3": {}, "noise": {}}
    for name, s in summaries.items():
        if name.startswith("replication/"):
            _, setting, lineup = name.split("/")
            report["replication"].setdefault(setting, {})[lineup] = s
    report["table3"] = table3(load_records(os.path.join(args.out, "replication", "A-10k-own", "tab3_fast_3slow")))
    report["noise"] = noise_check(args.out, args.mde_games)
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    text = markdown(report)
    with open(os.path.join(args.out, "report.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
