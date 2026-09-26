"""Figures and tables for the final paper, from results/final.

  python paper/final/make_figures.py

Reads results/final/stage0/report.json, results/final/stage1/{report.json,moments.jsonl,1a.jsonl,1b.jsonl},
results/final/stage2/{paired.json,pairs.csv} and the stage 2 game records. Writes PDFs to paper/final/figures and
LaTeX table bodies to paper/final/tables.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from figgie.cards import SUITS  # noqa: E402
from figgie.stats import bootstrap_ci, mean  # noqa: E402

RES = os.path.join(ROOT, "results", "final")
OUT = os.path.join(ROOT, "paper", "final", "figures")
TAB = os.path.join(ROOT, "paper", "final", "tables")
os.makedirs(OUT, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})
# Okabe-Ito colours (safe for colour-blind readers)
C = {"f": "#0072B2", "b": "#E69F00", "n": "#009E73", "jev": "#D55E00", "twin": "#555555", "cc": "#0072B2",
     "alg": "#D55E00", "desc": "#CC79A7", "none": "#999999", "A": "#0072B2", "B": "#E69F00"}
TYPES = ["fundamentalist", "bottom_feeder", "chartist", "noise", "market_maker_gm", "market_maker_as"]
SHORT = {"fundamentalist": "Fund.", "bottom_feeder": "Bottom-f.", "chartist": "Chartist", "noise": "Noise",
         "market_maker_gm": "MM (GM)", "market_maker_as": "MM (AS)", "neutral": "Neutral"}


def load_json(*p):
    path = os.path.join(RES, *p)
    return json.load(open(path)) if os.path.exists(path) else None


def load_jsonl(*p):
    path = os.path.join(RES, *p)
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    if os.environ.get("PREVIEW"):  # PNG copies for a quick look
        fig.savefig(os.path.join(os.environ["PREVIEW"], name.replace(".pdf", ".png")), dpi=160)
    plt.close(fig)
    print("wrote", name)


# --- stage 0 --------------------------------------------------------------------------------------------------

# Mean cash + payout per seat, read by eye from the paper's figures 3-5 (so approximate, +-10).
PAPER = {
    "fig3_f_3n": [660, 310, 320, 300], "fig3_2f_2n": [560, 560, 235, 235], "fig3_3f_n": [480, 490, 490, 130],
    "fig4_f_b_2n": [630, 540, 250, 160], "fig4_2f_b_n": [500, 510, 450, 140], "fig4_3f_b": [430, 430, 425, 315],
    "fig5_f_2b_n": [620, 420, 425, 125], "fig5_2f_2b": [490, 490, 310, 305],
}
LINEUP_LABEL = {"fig3_f_3n": "f+3n", "fig3_2f_2n": "2f+2n", "fig3_3f_n": "3f+n", "fig4_f_b_2n": "f+b+2n",
                "fig4_2f_b_n": "2f+b+n", "fig4_3f_b": "3f+b", "fig5_f_2b_n": "f+2b+n", "fig5_2f_2b": "2f+2b"}


def seat_type(key):
    return {"fundamentalist": "f", "bottom_feeder": "b", "noise": "n"}[key.split("#")[0].split("@")[0]]


def stage0():
    rep = load_json("stage0", "report.json")
    if rep is None:
        return
    settings = [("A-10k-paper", "A, 10,000 events", "o"), ("A-240s-paper", "A, 240 s", "s"),
                ("B-240s-paper", "B, 240 s", "^")]
    fig, axes = plt.subplots(1, 8, figsize=(7.0, 2.1), sharey=True)
    for ax, (lu, paper) in zip(axes, PAPER.items()):
        for j, (setting, label, marker) in enumerate(settings):
            s = rep["replication"].get(setting, {}).get(lu)
            if not s:
                continue
            keys = list(s["agents"])
            for i, k in enumerate(keys):
                a = s["agents"][k]
                se = a["sd_pnl"] / math.sqrt(s["games"])
                ax.errorbar(i + (j - 1) * 0.18, a["mean_pnl"] + 400, yerr=2 * se, fmt=marker, ms=3, lw=0.7,
                            color=C[seat_type(k)], label=label if (i == 0 and lu == "fig3_f_3n") else None,
                            mfc="white" if j else C[seat_type(k)])
        ax.scatter(range(4), paper, marker="_", s=120, color="black", lw=1.2, zorder=5,
                   label="paper (read from figures)" if lu == "fig3_f_3n" else None)
        ax.axhline(400, color="#bbbbbb", lw=0.6, ls="--", zorder=0)
        types = [seat_type(k) for k in rep["replication"]["A-10k-paper"][lu]["agents"]]
        ax.set_xticks(range(4), types)
        ax.set_title(LINEUP_LABEL[lu])
    axes[0].set_ylabel("mean cash + payout (chips)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.12))
    save(fig, "stage0_replication.pdf")

    # table: homogeneous SDs (paper table 2) and table 3
    lines = []
    for setting in ("A-10k-paper", "A-240s-paper", "B-240s-paper", "A-240s-equal", "B-240s-equal"):
        for lu, name in (("tab2_4n", "Noise"), ("tab2_4f", "Fundamentalist"), ("tab2_4b", "Bottom-feeder")):
            s = rep["replication"].get(setting, {}).get(lu)
            if not s:
                continue
            a = list(s["agents"].values())
            sd = lambda k: mean([x[k] for x in a])  # noqa: E731
            lines.append(f"{setting} & {name} & {sd('sd_pnl'):.1f} & {sd('sd_cash'):.1f} & {sd('sd_payout'):.1f} & "
                         f"{s['market']['trades_per_game']:.0f} \\\\")
    open(os.path.join(TAB, "table2.tex"), "w").write("\n".join(lines) + "\n")
    t3 = rep["table3"]
    lines = []
    for k, v in t3.items():
        cell = lambda x: f"[{x['ci95'][0]:.1f}, {x['ci95'][1]:.1f}]"  # noqa: E731
        lines.append(f"{k} & {cell(v['wealth'])} & {cell(v['cash'])} & {cell(v['payout'])} \\\\")
    open(os.path.join(TAB, "table3.tex"), "w").write("\n".join(lines) + "\n")
    lines = []
    for k, v in rep["noise"].items():
        mech, t = k.split("/")
        lines.append(f"{mech} & {SHORT[t]} & {v['mean_pnl']:.0f} & {v['sd_pnl']:.0f} & "
                     f"{v.get('sd_diff_vs_fundamentalist', float('nan')):.0f} \\\\".replace("nan", "--"))
    open(os.path.join(TAB, "noise.tex"), "w").write("\n".join(lines) + "\n")


# --- stage 1 --------------------------------------------------------------------------------------------------


def stage1():
    rep = load_json("stage1", "report.json")
    if rep is None:
        return
    # 1a: overlap with the twin
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2), sharey=True)
    for ax, mech in zip(axes, "AB"):
        for j, (w, lab) in enumerate((("algorithm", "algorithm persona"), ("behaviour", "behaviour persona"),
                                      ("none", "no persona"))):
            key = {"algorithm": "alg", "behaviour": "desc", "none": "none"}[w]
            xs, ys, lo, hi = [], [], [], []
            for i, t in enumerate(TYPES):
                v = rep["1a"].get(f"{mech}/{t}/{w}")
                if not v:
                    continue
                xs.append(i + (j - 1) * 0.22)
                ys.append(v["overlap"]["mean"])
                lo.append(v["overlap"]["mean"] - v["overlap"]["ci95"][0])
                hi.append(v["overlap"]["ci95"][1] - v["overlap"]["mean"])
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", ms=3.2, lw=0.8, color=C[key], label=lab)
        ax.set_xticks(range(len(TYPES)), [SHORT[t] for t in TYPES], rotation=25, ha="right")
        ax.set_title(f"Rule set {mech}")
        ax.set_ylim(0, 0.7)
        ax.grid(axis="y", lw=0.3, alpha=0.5)
    axes[0].set_ylabel("overlap with the twin\n(1 = same action distribution)")
    axes[0].legend(frameon=False, loc="upper left")
    save(fig, "stage1_overlap.pdf")

    # 1a: action shares, Jev (algorithm persona) vs twin
    fig, axes = plt.subplots(2, 6, figsize=(7.0, 2.6), sharey=True)
    kinds = ["buy", "sell", "take", "pass"]
    for r, mech in enumerate("AB"):
        for c, t in enumerate(TYPES):
            ax = axes[r, c]
            v = rep["1a"].get(f"{mech}/{t}/algorithm")
            vd = rep["1a"].get(f"{mech}/{t}/behaviour")
            if not v:
                continue
            x = range(len(kinds))
            ax.bar([i - 0.27 for i in x], [v["twin_share"][k] for k in kinds], 0.27, color=C["twin"], label="twin")
            ax.bar(list(x), [v[f"share_{k}"]["mean"] for k in kinds], 0.27, color=C["alg"], label="Jev, algorithm")
            ax.bar([i + 0.27 for i in x], [vd[f"share_{k}"]["mean"] for k in kinds], 0.27, color=C["desc"],
                   label="Jev, behaviour")
            ax.set_xticks(list(x), kinds if r else [""] * 4, rotation=90 if r else 0)
            if r == 0:
                ax.set_title(SHORT[t])
            if c == 0:
                ax.set_ylabel(f"{mech}: share")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.06))
    save(fig, "stage1_shares.pdf")

    # 1b: log loss per state version
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.0), sharey=True)
    versions = [("basic", "basic"), ("summary", "+summary"), ("log,summary", "+log+summary"), ("known", "+known"),
                ("assist", "+assist")]
    for ax, mech in zip(axes, "AB"):
        for i, (k, lab) in enumerate(versions):
            v = rep["1b"].get(f"{mech}/{k}")
            if not v:
                continue
            m, (lo, hi) = v["logloss_jev"]["mean"], v["logloss_jev"]["ci95"]
            ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="o", ms=3.5, color=C["jev"] if k in ("basic", "summary", "log,summary") else "#999999")
        cc = rep["1b"][f"{mech}/summary"]["logloss_card_counting"]
        ax.axhspan(cc["ci95"][0], cc["ci95"][1], color=C["cc"], alpha=0.15, lw=0)
        ax.axhline(cc["mean"], color=C["cc"], lw=0.8, label="card counting")
        ax.axhline(math.log(4), color="black", lw=0.7, ls="--", label="uniform guess")
        ax.set_xticks(range(len(versions)), [lab for _, lab in versions], rotation=20, ha="right")
        ax.set_title(f"Rule set {mech}")
    axes[0].set_ylabel("log loss of the goal-suit belief\n(lower is better)")
    axes[0].legend(frameon=False, loc="upper left")
    save(fig, "stage1_logloss.pdf")

    # calibration, +summary, A and B pooled
    moments = {m["id"]: m for m in load_jsonl("stage1", "moments.jsonl")}
    rows = [r for r in load_jsonl("stage1", "1b.jsonl") if r["context"] == "summary"]
    fig, ax = plt.subplots(figsize=(2.6, 2.4))
    for name, get, col in (("Jev (+summary)", lambda r: r["goal_probs"], C["jev"]),
                           ("card counting", lambda r: moments[r["moment"]]["card_counting"], C["cc"])):
        bins = [[0, 0.0, 0] for _ in range(10)]
        for r in rows:
            goal = moments[r["moment"]]["goal"]
            for s in SUITS:
                p = get(r).get(s, 0.0)
                b = bins[min(9, int(p * 10))]
                b[0] += 1
                b[1] += p
                b[2] += s == goal
        pts = [(sp / n, h / n, n) for n, sp, h in bins if n >= 10]
        ax.plot([x for x, _, _ in pts], [y for _, y, _ in pts], "o-", ms=3, lw=0.9, color=col, label=name)
    ax.plot([0, 1], [0, 1], color="black", lw=0.6, ls="--")
    ax.set_xlabel("stated probability")
    ax.set_ylabel("how often the suit was the goal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper left")
    save(fig, "stage1_calibration.pdf")

    lines = []
    for mech in "AB":
        for k, lab in versions:
            v = rep["1b"].get(f"{mech}/{k}")
            if not v:
                continue
            f = lambda x: f"{x['mean']:.3f} [{x['ci95'][0]:.2f}, {x['ci95'][1]:.2f}]"  # noqa: E731
            lines.append(f"{mech} & {lab} & {f(v['logloss_jev'])} & {v['brier_jev']['mean']:.3f} & "
                         f"{v['top1_jev']['mean']:.2f} & {v['logloss_card_counting']['mean']:.3f} & "
                         f"{v['top1_card_counting']['mean']:.2f} \\\\")
    open(os.path.join(TAB, "stage1b.tex"), "w").write("\n".join(lines) + "\n")
    lines = []
    for mech in "AB":
        for t in TYPES:
            cells = []
            for w in ("algorithm", "behaviour", "none"):
                v = rep["1a"][f"{mech}/{t}/{w}"]
                cells.append(f"{v['overlap']['mean']:.2f}")
            da = rep["1a"][f"{mech}/{t}/algorithm"]["diff"]
            dd = rep["1a"][f"{mech}/{t}/behaviour"]["diff"]
            pa = rep["1a"][f"{mech}/{t}/algorithm"]["p_diff_holm"]
            pd = rep["1a"][f"{mech}/{t}/behaviour"]["p_diff_holm"]
            fmt = lambda d, p: f"{d['mean']:+.3f}" + ("$^*$" if p < 0.05 else "")  # noqa: E731
            lines.append(f"{mech} & {SHORT[t]} & {' & '.join(cells)} & {fmt(da, pa)} & {fmt(dd, pd)} \\\\")
    open(os.path.join(TAB, "stage1a.tex"), "w").write("\n".join(lines) + "\n")


# --- stage 2 --------------------------------------------------------------------------------------------------


def stage2():
    rep = load_json("stage2", "paired.json")
    if rep is None:
        return
    conds = ["neutral"] + [f"{t}{s}" for t in TYPES for s in ("", "-desc")]
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    for j, mech in enumerate("AB"):
        for i, c in enumerate(conds):
            v = rep["conditions"].get(f"{mech}/{c}")
            if not v:
                continue
            m, (lo, hi) = v["mean_d"]["estimate"], v["mean_d"]["ci95"]
            ax.errorbar(i + (j - 0.5) * 0.3, m, yerr=[[m - lo], [hi - m]], fmt="o" if not c.endswith("-desc") else "s",
                        ms=3.2, lw=0.8, color=C[mech], mfc=C[mech] if not c.endswith("-desc") else "white",
                        label=f"rule set {mech}" if i == 0 else None)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(range(len(conds)), [SHORT[c.removesuffix("-desc")] + (" (beh.)" if c.endswith("-desc") else "")
                                      for c in conds], rotation=35, ha="right")
    ax.set_ylabel("Jev profit minus twin profit\n(chips per game)")
    ax.grid(axis="y", lw=0.3, alpha=0.5)
    ax.legend(frameon=False, loc="lower left", ncol=2)
    save(fig, "stage2_paired.pdf")

    # Jev's belief over game time, all conditions pooled, vs card counting at the same moments
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.0), sharey=True)
    for ax, mech in zip(axes, "AB"):
        bins = defaultdict(lambda: {"jev": [], "cc": []})
        games_dir = os.path.join(RES, "stage2", mech, "games")
        for fn in os.listdir(games_dir):
            if fn.startswith("["):
                continue
            for line in open(os.path.join(games_dir, fn)):
                rec = json.loads(line)
                trace = rec.get("jev_trace", {}).get(str(rec["tested_seat"]), [])
                for t in trace:
                    if "goal" not in t or "card_counting" not in t:
                        continue
                    b = min(7, int(t["t"] // 30))
                    bins[b]["jev"].append(-math.log(max(t["goal"].get(rec["goal"], 0.0), 1e-9)))
                    bins[b]["cc"].append(-math.log(max(t["card_counting"].get(rec["goal"], 0.0), 1e-9)))
        xs = sorted(bins)
        for k, lab in (("jev", "Jev"), ("cc", "card counting")):
            ax.plot([x * 30 + 15 for x in xs], [mean(bins[x][k]) for x in xs], "o-", ms=3, lw=0.9, color=C[k if k == "jev" else "cc"],
                    label=lab)
        ax.axhline(math.log(4), color="black", lw=0.6, ls="--", label="uniform guess")
        ax.set_xlabel("game time (s)")
        ax.set_title(f"Rule set {mech}")
    axes[0].set_ylabel("mean log loss\nof the goal-suit belief")
    axes[0].legend(frameon=False)
    save(fig, "stage2_belief_time.pdf")

    lines = []
    for c in conds:
        cells = []
        for mech in "AB":
            v = rep["conditions"].get(f"{mech}/{c}")
            if not v:
                cells += ["--"] * 3
                continue
            d = v["mean_d"]
            sig = "$^*$" if d["ci95"][0] > 0 or d["ci95"][1] < 0 else ""
            cells += [f"{v['mean_pnl_jev']:.0f}", f"{v['mean_pnl_twin']:.0f}",
                      f"{d['estimate']:+.0f} [{d['ci95'][0]:.0f}, {d['ci95'][1]:.0f}]{sig}"]
        name = SHORT[c.removesuffix("-desc")] + (" (behaviour)" if c.endswith("-desc") else (" (algorithm)" if c != "neutral" else ""))
        lines.append(f"{name} & {' & '.join(cells)} \\\\")
    open(os.path.join(TAB, "stage2.tex"), "w").write("\n".join(lines) + "\n")

    lines = []
    for c in conds:
        for mech in "AB":
            v = rep["conditions"].get(f"{mech}/{c}")
            if not v:
                continue
            j, t = v["jev"], v["twin"]
            b = v.get("jev_belief", {})
            name = SHORT[c.removesuffix("-desc")] + (" (beh.)" if c.endswith("-desc") else "")
            lines.append(f"{mech} & {name} & {j['seat_trades']:.1f} & {t['seat_trades']:.1f} & {j['seat_passes']:.0f} & "
                         f"{t['seat_passes']:.0f} & {j['mispricing']:.2f} & {t['mispricing']:.2f} & "
                         f"{b.get('logloss', float('nan')):.2f} & {b.get('logloss_cc', float('nan')):.2f} \\\\")
    open(os.path.join(TAB, "stage2_secondary.tex"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    stage0()
    stage1()
    stage2()
