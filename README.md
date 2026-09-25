# figgie-on-jev

A Figgie market simulator for two questions:

1. **How do Jev's decisions compare with classical maths models?** Jev is TypeSafe AI's decision model. The classical baseline is exact Bayesian card counting and the hand-coded strategies from Ozerov, DiSilvio and Luo, [*Traders in a Strange Land*](https://arxiv.org/abs/2110.00879) (2021).
2. **How do different Jev "personalities" change a game?** For example, a value trader, a market maker, a momentum chaser, a majority hoarder, a timid player or a gambler.

Pure Python 3.10+, no dependencies (`pytest` for tests).

## Quick start

```bash
python -m pytest -q                                   # tests
python -m figgie.experiments.tournament --games 40 \
    --lineup fundamentalist,bottom_feeder,chartist,noise   # classical only, no API key needed
```

Every experiment runs offline with `--backend mock`, which is the default. The mock is **not Jev**: it answers at random so you can check the pipeline without an API key.

To use the real model, add a key to `.env.local` in the repo root. That file is gitignored, and variables already set in your shell take precedence over it:

```dotenv
OPENROUTER_API_KEY=sk-or-v1-...
TYPESAFE_API_KEY=...
```

Then run:

```bash
python -m figgie.experiments.compare --backend jev --games 20 --log jev.jsonl --out results/compare
python -m figgie.experiments.sweep   --backend jev --games 20 --out results/sweep
```

The client uses the same routing as Jace's other Jev code. If `OPENROUTER_API_KEY` is set, it calls OpenRouter (`POST https://openrouter.ai/api/alpha/decisions`, model `typesafe/jev-1.13`). Otherwise it calls TypeSafe directly (`POST https://api.typesafe.ai/v1/systemone`, model `jev-latest`). Use `--provider openrouter|typesafe` to force a route and `JEV_MODEL` to override the model. With `--provider openrouter` and no `OPENROUTER_API_KEY`, requests go out without an `Authorization` header, for environments whose network proxy injects the key (such as a Claude cloud sandbox with an openrouter.ai credential).

## Experiments

| Command | Question | Output |
|---|---|---|
| `figgie.experiments.compare` | At the same frozen game states, how do Jev's goal-suit beliefs and trade choices compare with the exact posterior and the fundamentalist? | Brier score and log loss (Jev, posterior, 25% baseline), distance between Jev and the posterior, top-1 agreement, action agreement, regret in chips. `decisions.csv` has one row per decision point. |
| `figgie.experiments.sweep` | Each personality in the same seat against the same three opponents, plus classical agents in that seat for reference. | A table of P&L with bootstrap 95% CI, trades, and market-wide trades and mispricing. |
| `figgie.experiments.tournament` | Any four-agent line-up, for example four Jev personalities playing each other. | P&L per agent with CI, activity, rejected orders and market metrics. Seats rotate each game. |

Agent specs: `fundamentalist`, `bottom_feeder`, `chartist`, `noise`, `jev:<personality>`, and `jev:<personality>+assist`. The `+assist` version is a hybrid: the exact card-counting probabilities are added to Jev's state, so code does the maths and Jev makes the decision. Personalities live in `figgie/personalities.py`; add your own there.

Useful flags: `--jev-latency` sets the simulated delay in seconds. By default the simulation uses each call's measured round-trip time, so Jev's speed is priced in. `--max-calls` is a hard cap on API calls. `--log` records every request and response.

## How it works

- `figgie/cards.py` and `figgie/posterior.py` hold the 12 possible decks, dealing, and exact Bayesian goal-suit probabilities from the cards known to exist. Known cards are your hand plus cards other players have shown they held by selling them. Card values include a share of the majority bonus.
- `figgie/market.py` has one best bid and one best ask per suit. Crossing orders trade, and every trade clears all quotes, as in the real game.
- `figgie/engine.py` is the discrete-event game. An agent decides on what it sees now, and its order lands `latency` seconds later against the book as it is *then*, so slow agents lose races.
- `figgie/agents/` has the paper's fundamentalist, bottom-feeder, chartist and a noise trader. `JevAgent` sends Jev the JSON state and one Choice question listing every legal action (at most 73, well under Jev's limit of 255).
- `figgie/jev.py` is a stdlib HTTP client for Jev through OpenRouter or TypeSafe directly (the same request body on both routes), plus the mock. It tracks calls, tokens and cost ($0.042 per million input tokens).

A Jev agent wakes about every 2 seconds, so a 4-minute game makes about 100 calls per Jev seat.

## Sanity check against the paper

Classical line-up, 40 games (`tournament --lineup fundamentalist,bottom_feeder,chartist,noise`): the fundamentalist and bottom-feeder both make about +160 chips a game. The chartist loses about 275 chips, and nearly all of its orders are rejected because it chases stale prices. This matches the paper's qualitative findings: fundamentalists win, bottom-feeders cap their edge, and chartists fail.
