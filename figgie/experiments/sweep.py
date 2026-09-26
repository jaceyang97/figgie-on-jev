"""Stage 2, the game layer: every Jev condition and its rule-based twin, in both rule sets.

A condition is one rule set (A or B) and one Jev persona: neutral (no persona)
or one of the six trader types in algorithm wording (`jev:<type>`) or
behaviour wording (`jev:<type>-desc`). Each condition takes the tested seat
against the same three opponents (the paper's line-up: fundamentalist,
bottom-feeder, noise trader). Each twin, the rule-based trader of the same
type, takes the same seat in the same games; the twin of neutral is the
fundamentalist. All conditions and twins use the same game seeds, so game g
has the same deal everywhere. All seats and games run at once; one rate
limiter and one call budget cover the whole run.

Output, per rule set: <out>/<mech>/games/<name>.jsonl (full game records,
also the checkpoint: a rerun with the same arguments resumes) and
<out>/<mech>/logs/<name>.jsonl (every Jev request and response with its game,
seat and decision number). <out>/run.json records the code version, the
settings and which twin belongs to which condition. Analyse with
figgie.experiments.paired.

Usage:
  python -m figgie.experiments.sweep --backend mock --games 2 --out /tmp/stage2
  python -m figgie.experiments.sweep --backend jev --provider openrouter --games 40 --out results/stage2
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from ..agents import TYPES
from ..jev import MAX_RPS, CallBudget, RateLimiter, make_client
from ..records import write_run_header
from .tournament import add_common_args
from .tournament import run as run_tournament


def conditions(types, wordings, neutral: bool = True) -> list[str]:
    """Jev personas: neutral, then each type in each wording."""
    out = ["neutral"] if neutral else []
    for t in types:
        for w in wordings:
            out.append(t if w == "alg" else f"{t}-desc")
    return out


def twin_of(persona: str) -> str:
    base = persona.removesuffix("-desc")
    return "fundamentalist" if base == "neutral" else base


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--mechanisms", default="A,B")
    ap.add_argument("--field", default="fundamentalist,bottom_feeder,noise", help="the three fixed opponents")
    ap.add_argument("--types", default=",".join(TYPES))
    ap.add_argument("--wordings", default="alg,desc", help="alg (algorithm persona), desc (behaviour persona)")
    ap.add_argument("--no-neutral", action="store_true")
    ap.add_argument("--context", default="summary", help="Jev state version: comma list of log, summary, known, assist")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--max-calls", type=int, default=140_000, help="hard cap on Jev calls for the whole run")
    ap.add_argument("--max-rps", type=float, default=MAX_RPS, help="Jev requests per second, shared by all seats")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    field = args.field.split(",")
    if len(field) != 3:
        raise SystemExit("--field needs exactly 3 agents")
    flags = "".join(f"+{p}" for p in args.context.split(",") if p)
    personas = conditions([t for t in args.types.split(",") if t], args.wordings.split(","), not args.no_neutral)
    twins = sorted({twin_of(p) for p in personas})
    mechanisms = args.mechanisms.split(",")
    run_id = os.path.basename(os.path.normpath(args.out))
    write_run_header(args.out, "stage2", {**vars(args), "personas": personas, "twins": twins,
                                          "twin_of": {p: twin_of(p) for p in personas}})

    seats = []
    for mech in mechanisms:
        for sub in ("games", "logs"):
            os.makedirs(os.path.join(args.out, mech, sub), exist_ok=True)
        seats += [(mech, p, f"jev:{p}{flags}") for p in personas]
        seats += [(mech, f"[{t}]", t) for t in twins]
    limiter, budget = RateLimiter(args.max_rps), CallBudget(args.max_calls)

    def play(seat, games):
        mech, name, spec = seat
        client = None
        if spec.startswith("jev"):
            client = make_client(args.backend, seed=args.seed, provider=args.provider, limiter=limiter, budget=budget,
                                 log_path=os.path.join(args.out, mech, "logs", f"{name}.jsonl"))
        t = SimpleNamespace(backend=args.backend, provider=args.provider, lineup=",".join([spec] + field),
                            games=games, duration=args.duration, max_events=args.max_events, speed=args.speed,
                            decode=args.decode, seed=args.seed, mechanism=mech, tested=0, condition=f"{mech}/{name}",
                            run_id=run_id, max_calls=None, log=None, out=None, workers=None,
                            checkpoint=os.path.join(args.out, mech, "games", f"{name}.jsonl"))
        try:
            s = run_tournament(t, client)
        except RuntimeError as e:
            print(f"WARNING {mech}/{name}: {e}")
            return (mech, name), None
        key = spec if spec in s["agents"] else f"{spec}#0"
        return (mech, name), {"games": s["games"], "failed": s["games_failed"], "seat": s["agents"][key],
                              "market": s["market"], "calls": s.get("jev_calls", 0), "cost_usd": s.get("cost_usd", 0.0)}

    # The twins cost nothing: play all their games at once. The Jev arms play in waves, one game number
    # at a time for every condition, so the finished games stay balanced across conditions and a stop
    # (spending limit, API error) loses at most one game per condition.
    jev_seats = [x for x in seats if x[2].startswith("jev")]
    twin_seats = [x for x in seats if not x[2].startswith("jev")]
    rows, spent = {}, {}
    with ThreadPoolExecutor(max_workers=max(1, len(twin_seats))) as pool:
        rows.update({k: r for k, r in pool.map(lambda x: play(x, args.games), twin_seats) if r is not None})
    for wave in range(1, args.games + 1):
        with ThreadPoolExecutor(max_workers=max(1, len(jev_seats))) as pool:
            got = dict(pool.map(lambda x: play(x, wave), jev_seats))
        for k, r in got.items():
            if r is not None:
                c, u = spent.get(k, (0, 0.0))
                spent[k] = (c + r["calls"], u + r["cost_usd"])
                rows[k] = {**r, "calls": spent[k][0], "cost_usd": round(spent[k][1], 4)}
        failed = [k for k, r in got.items() if r is None or r["failed"]]
        done = min((r["games"] for r in got.values() if r is not None), default=0)
        print(f"wave {wave}: every condition has at least {done} games; Jev calls so far {budget.calls}", flush=True)
        if failed:
            print(f"STOP after wave {wave}: games failed in {sorted(failed)[:5]}... Fix the cause and rerun the "
                  "same command to continue.", flush=True)
            break

    if args.backend == "mock":
        print("NOTE: the mock backend answers at random and ignores personas; Jev rows say nothing about Jev.\n")
    print(f"{'condition':<28}{'games':>6}{'P&L':>9}{'95% CI':>18}{'trades':>8}{'calls':>8}{'US$':>8}")
    for (mech, name), r in sorted(rows.items()):
        lo, hi = r["seat"]["ci95"]
        print(f"{mech + '/' + name:<28}{r['games']:>6}{r['seat']['mean_pnl']:>9.1f}{f'[{lo:.0f}, {hi:.0f}]':>18}"
              f"{r['seat']['trades_per_game']:>8.1f}{r['calls']:>8}{r['cost_usd']:>8.3f}")
    print(f"\ntotal Jev calls {budget.calls}, estimated cost US${sum(r['cost_usd'] for r in rows.values()):.2f}")
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({f"{m}/{n}": r for (m, n), r in rows.items()}, f, indent=2)


if __name__ == "__main__":
    main()
