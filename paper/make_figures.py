"""Build the paper's figures and numbers from results/.

Reads results/compare/{summary.json,decisions.csv}, results/sweep/*/sweep.json
and the Jev call logs in results/logs/ (if present; otherwise the action-mix
table committed in paper/data/action_mix.json is reused). Writes PDFs to
paper/figures/ and a small numbers file to paper/data/.

  python paper/make_figures.py
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "paper", "figures")
DATA = os.path.join(ROOT, "paper", "data")
SUITS = ["spades", "clubs", "hearts", "diamonds"]
PERSONALITIES = ["neutral", "value", "market_maker", "momentum", "hoarder", "timid", "gambler"]
KINDS = ["pass", "buy", "sell", "bid", "ask"]

BLUE, ORANGE, GREY, GREEN, RED = "#2a6fdb", "#e07b39", "#9a9a9a", "#3a9a5b", "#c8453c"
KIND_COLORS = {"pass": "#cfcfcf", "buy": "#2a6fdb", "sell": "#e07b39", "bid": "#8fb4ef", "ask": "#f2b48c"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    if os.environ.get("PREVIEW_DIR"):
        fig.savefig(os.path.join(os.environ["PREVIEW_DIR"], name.replace(".pdf", ".png")), bbox_inches="tight", dpi=130)
    plt.close(fig)


def kind(label: str) -> str:
    return label.split("_")[0]


def compare_figures():
    summary = json.load(open(os.path.join(RES, "compare", "summary.json")))
    rows = list(csv.DictReader(open(os.path.join(RES, "compare", "decisions.csv"))))
    n = len(rows)

    def top(r, who):
        return max(SUITS, key=lambda s: float(r[f"{who}_{s}"]))

    acc = {w: sum(top(r, w) == r["true_goal"] for r in rows) / n for w in ("posterior", "jev")}
    maxp = {w: sum(max(float(r[f"{w}_{s}"]) for s in SUITS) for r in rows) / n for w in ("posterior", "jev")}

    # Fig 1: scoring rules with CIs.
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.7))
    for ax, metric, uniform, title in [
        (axes[0], "brier", summary["brier_uniform"], "Brier score"),
        (axes[1], "logloss", math.log(4), "Log loss"),
    ]:
        labels, means, errs, colors = [], [], [], []
        for who, lab, c in [("posterior", "Exact\nposterior", BLUE), ("jev", "Jev", ORANGE)]:
            m = summary[f"{metric}_{who}"]
            labels.append(lab); means.append(m["mean"]); colors.append(c)
            errs.append([[m["mean"] - m["ci95"][0]], [m["ci95"][1] - m["mean"]]])
        x = range(len(labels))
        for i in x:
            ax.bar(i, means[i], color=colors[i], width=0.6, yerr=errs[i], capsize=2, error_kw={"lw": 0.8})
        ax.axhline(uniform, color=GREY, ls="--", lw=0.8)
        ax.text(1.45, uniform, "uniform\n25%", va="center", ha="left", fontsize=6, color="#555")
        ax.set_xticks(list(x)); ax.set_xticklabels(labels)
        ax.set_xlim(-0.5, 2.0)
        ax.set_title(f"{title} (lower is better)")
    save(fig, "compare_scores.pdf")

    # Fig 2: reliability diagram over all 4 suits x n decision points.
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0001]
    fig, ax = plt.subplots(figsize=(2.6, 2.4))
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.8, ls="--")
    for who, c, lab, mk in [("posterior", BLUE, "Exact posterior", "o"), ("jev", ORANGE, "Jev", "s")]:
        pts = [(float(r[f"{who}_{s}"]), 1.0 if r["true_goal"] == s else 0.0) for r in rows for s in SUITS]
        xs, ys, ns = [], [], []
        for lo, hi in zip(bins, bins[1:]):
            b = [p for p in pts if lo <= p[0] < hi]
            if len(b) >= 5:
                xs.append(sum(p for p, _ in b) / len(b)); ys.append(sum(y for _, y in b) / len(b)); ns.append(len(b))
        ax.plot(xs, ys, color=c, marker=mk, ms=3, lw=1, label=lab)
    ax.set_xlabel("Stated probability of goal suit"); ax.set_ylabel("Observed frequency")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "compare_calibration.pdf")

    # Fig 3: action mix at the same decision points.
    mix = {w: Counter(kind(r[f"{w}_action"]) for r in rows) for w in ("classical", "jev")}
    fig, ax = plt.subplots(figsize=(3.4, 1.1))
    for i, (w, lab) in enumerate([("jev", "Jev (neutral)"), ("classical", "Fundamentalist")]):
        left = 0
        for k in KINDS:
            v = mix[w][k] / n
            ax.barh(i, v, left=left, color=KIND_COLORS[k], height=0.6, label=k if i == 0 else None)
            if v >= 0.07:
                ax.text(left + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=6)
            left += v
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Jev (neutral)", "Fundamentalist"])
    ax.set_xlim(0, 1); ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(ncol=5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "compare_actions.pdf")

    return {
        "n": n, "calls": summary["jev_calls"], "cost_usd": summary["cost_usd"],
        "top1_acc_posterior": acc["posterior"], "top1_acc_jev": acc["jev"],
        "mean_maxp_posterior": maxp["posterior"], "mean_maxp_jev": maxp["jev"],
        "top1_agree": summary["top1_agree"]["mean"], "action_agree": summary["action_agree"]["mean"],
        "tv": summary["tv_distance"]["mean"], "regret": summary["regret"],
        "brier_posterior": summary["brier_posterior"], "brier_jev": summary["brier_jev"],
        "logloss_posterior": summary["logloss_posterior"], "logloss_jev": summary["logloss_jev"],
        "mix_jev": {k: mix["jev"][k] / n for k in KINDS}, "mix_classical": {k: mix["classical"][k] / n for k in KINDS},
    }


def action_mix():
    path = os.path.join(DATA, "action_mix.json")
    logs = {p: os.path.join(RES, "logs", f"sweep_{p}.jsonl") for p in PERSONALITIES}
    if not all(os.path.exists(f) for f in logs.values()):
        return json.load(open(path))
    out = {}
    for p, f in logs.items():
        c, conf, lat = Counter(), [], []
        for line in open(f):
            d = json.loads(line)
            a = d["response"]["answers"]["action"]
            c[kind(a["choice"])] += 1
            conf.append(a["confidence"]); lat.append(d["latency"])
        total = sum(c.values())
        lat.sort()
        out[p] = {"calls": total, "mix": {k: c[k] / total for k in KINDS},
                  "mean_confidence": sum(conf) / total, "median_latency": lat[total // 2]}
    json.dump(out, open(path, "w"), indent=2)
    return out


def sweep_figures():
    rows = {}
    for f in glob.glob(os.path.join(RES, "sweep", "*", "sweep.json")):
        rows.update(json.load(open(f)))
    mix = action_mix()

    order = sorted(PERSONALITIES, key=lambda p: rows[p]["jev"]["mean_pnl"])
    base = ["[noise]", "[fundamentalist]"]
    names = base[:1] + order + base[1:]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for i, p in enumerate(names):
        r = rows[p]["jev"]
        m, (lo, hi) = r["mean_pnl"], r["ci95"]
        c = GREY if p.startswith("[") else (GREEN if m > 0 else ORANGE)
        ax.barh(i, m, color=c, height=0.65, xerr=[[m - lo], [hi - m]], capsize=2, error_kw={"lw": 0.8})
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([p.strip("[]") + (" (classical)" if p.startswith("[") else "") for p in names])
    ax.set_xlabel("Mean P&L per game in the test seat (chips, 95% CI)")
    save(fig, "sweep_pnl.pdf")

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    order2 = sorted(PERSONALITIES, key=lambda p: mix[p]["mix"]["pass"])
    for i, p in enumerate(order2):
        left = 0
        for k in KINDS:
            v = mix[p]["mix"][k]
            ax.barh(i, v, left=left, color=KIND_COLORS[k], height=0.65, label=k if i == 0 else None)
            if v >= 0.08:
                ax.text(left + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=6)
            left += v
    ax.set_yticks(range(len(order2))); ax.set_yticklabels(order2)
    ax.set_xlim(0, 1); ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(ncol=5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_xlabel("Share of Jev decisions")
    save(fig, "sweep_actions.pdf")

    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    for p in PERSONALITIES:
        r = rows[p]
        ax.scatter(r["jev"]["trades_per_game"], r["jev"]["mean_pnl"], color=ORANGE, s=14, zorder=3)
        ax.annotate(p, (r["jev"]["trades_per_game"], r["jev"]["mean_pnl"]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")
    for p in base:
        r = rows[p]
        ax.scatter(r["jev"]["trades_per_game"], r["jev"]["mean_pnl"], color=GREY, marker="D", s=12, zorder=3)
        ax.annotate(p.strip("[]"), (r["jev"]["trades_per_game"], r["jev"]["mean_pnl"]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points", color="#555")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Trades per game by the test seat"); ax.set_ylabel("Mean P&L (chips)")
    save(fig, "sweep_trades_pnl.pdf")

    return {"rows": rows, "mix": mix}


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(DATA, exist_ok=True)
    numbers = {"compare": compare_figures(), "sweep": sweep_figures()}
    json.dump(numbers, open(os.path.join(DATA, "numbers.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in numbers["compare"].items() if not isinstance(v, dict)}, indent=1))
    for p, r in sorted(numbers["sweep"]["rows"].items(), key=lambda kv: kv[1]["jev"]["mean_pnl"]):
        m = numbers["sweep"]["mix"].get(p)
        print(f"{p:18s} pnl {r['jev']['mean_pnl']:8.1f} ci {r['jev']['ci95']} trades {r['jev']['trades_per_game']:5.1f} "
              f"mkt {r['market']['trades_per_game']:5.1f} misp {r['market']['mean_mispricing_chips']:.2f} field {r['field']}"
              + (f" pass {m['mix']['pass']:.2f} conf {m['mean_confidence']:.3f} calls {m['calls']}" if m else ""))


if __name__ == "__main__":
    main()
