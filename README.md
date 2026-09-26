# figgie-on-jev

A Figgie market simulator for two questions:

1. **How do Jev's decisions compare with classical models?** Jev is TypeSafe AI's decision model. The classical baseline is card counting and the rule-based traders from Ozerov, DiSilvio and Luo, [*Traders in a Strange Land*](https://arxiv.org/abs/2110.00879) (2021).
2. **Can Jev play a given type of market participant?** Each Jev persona is a written version of one rule-based trader. We compare Jev with that trader (its "twin") in the same seat and the same games.

Pure Python 3.10+, no dependencies (`pytest` for tests). The experiment design is in the project proposal; this file says how to run it.

## Quick start

```bash
python -m pytest -q
python -m figgie.experiments.tournament --games 40 --mechanism A \
    --lineup fundamentalist,bottom_feeder,noise,noise        # rule-based only, no API key needed
```

Every experiment runs offline with `--backend mock`, which is the default. The mock is **not Jev**: it answers at random, so you can check the code and the logs without an API key and at no cost.

To use the real model, add a key to `.env.local` in the repo root (gitignored; variables already set in your shell take precedence):

```dotenv
OPENROUTER_API_KEY=sk-or-v1-...
TYPESAFE_API_KEY=...
```

If `OPENROUTER_API_KEY` is set, the client calls OpenRouter (`POST https://openrouter.ai/api/alpha/decisions`, model `typesafe/jev-1.13`). Otherwise it calls TypeSafe directly (`POST https://api.typesafe.ai/v1/systemone`, model `jev-latest`). `--provider openrouter|typesafe` forces a route. With `--provider openrouter` and no key, requests go out without an `Authorization` header, for sandboxes whose network proxy adds the key.

## Rule sets

- **A, order book** (the paper). Many orders can wait at each price. A new order trades with the best waiting order on the other side, at the waiting order's price (price-time priority). Orders stay after other trades. Each player keeps at most 5 orders per side per suit (the oldest goes). A player's own order on the other side is removed, not traded against. An order whose owner can no longer pay or deliver is removed when it is reached. Players can cancel their orders in a suit.
- **B, open outcry** (real Figgie). One best bid and one best ask per suit. A new bid must beat the best bid, a new ask must beat the best ask. Every trade removes all bids and asks in all suits.

`--mechanism A|B` picks one. Games last 240 seconds, or with `--max-events 10000` they end after 10,000 events, as in the paper.

## Traders

| Spec | What it is |
|---|---|
| `fundamentalist`, `bottom_feeder`, `chartist`, `noise` | The paper's four traders (section 2.3, Algorithms 2 to 4). The fundamentalist deletes its stale orders in rule set A (Algorithm 4). The bottom-feeder copies the fundamentalists; the seat under test is never its prey. |
| `market_maker_gm` | Glosten and Milgrom (1985): quotes are the card's expected value after a buy or a sell, from card counting and the direction of other players' trades. |
| `market_maker_as` | Avellaneda and Stoikov (2008): quotes around the market price, moved against the cards it holds, wider with price variance and time left. |
| `jev:<persona>` | Jev with the persona of that trader (`jev:fundamentalist`: the steps of the algorithm) or its behaviour description (`jev:fundamentalist-desc`). `jev:neutral` has no persona. |

Jev context flags: `+summary` and `+log` add facts from the game (the public trade and order record); `+known` and `+assist` add values that code calculates (cards known to exist, card-counting probabilities) and are ablations only. Example: `jev:chartist-desc+summary`.

Speed: `--speed equal` (default; every trader decides 0.5 times per second on average and its order arrives 0.3 s later), `--speed paper` (1 per second, no delay) or `--speed <rate>/<latency>`. A spec can carry its own speed: `fundamentalist@1/200`.

Jev's action is a sample from its probabilities over the whole menu (`--decode sample`, the default). The menu has pass, take the best price, cancel (A only) and bids and asks at the prices 1 to 20 and 22 to 40 in steps of 2: at most 253 options, below Jev's limit of 255.

## Experiments

| Stage | Command | What it does | Cost |
|---|---|---|---|
| 0 | `figgie.experiments.replicate --out results/stage0` | The paper's line-ups (figures 3 to 5, tables 2 and 3), 100 games each, in A with 10,000 events, A with 240 s and B with 240 s, at the paper's speed and the equal speed. Also the profit noise of each trader type in the stage 2 line-up. Writes `report.md`. | Free |
| 1 | `figgie.experiments.stage1 moments / ask / analyse` | Frozen moments from rule-based games. 1a: persona wording (algorithm, behaviour, none) against the twin. 1b: five state versions, goal-suit belief against card counting. 1c: repeatability. | About US$1.5 |
| 2 | `figgie.experiments.sweep --backend jev --out results/stage2` | 26 conditions (2 rule sets × (neutral + 6 types × 2 wordings)), 40 games each, and each condition's twin, against a fundamentalist, a bottom-feeder and a noise trader. All conditions use the same game seeds. `--max-calls` is a hard spending limit. | About US$28 |
| 2 | `figgie.experiments.paired --run results/stage2` | Jev minus twin in the same seat and games: per condition, per rule set, A minus B, behaviour minus algorithm wording, and the regression with and without the starting goal cards. | Free |
| any | `figgie.experiments.tournament` | Any four-trader line-up. Seats rotate each game. | Free with rule-based traders |

`figgie.experiments.compare` is the first run's decision study (rule set B only), kept to reproduce it.

## What is logged

- `run.json` (or `run_<step>.json`) in each output directory: the git commit, whether the tree had changes, Python version, command line and all settings.
- Game records, one JSON line per game (`games.jsonl`, or `<out>/<A|B>/games/<condition>.jsonl` in stage 2): seed, rule set, deck and goal suit, seats and speeds, starting and final hands, profit, cash and payout per seat, decisions, passes and rejected orders per seat, every trade `[t, suit, price, buyer, seller, aggressor]`, every order `[t, player, side, suit, price, oid]`, every cancel with its reason, and for Jev seats one line per decision (time, chosen action and its probability, the goal-suit belief, and card counting at the same moment for comparison). A rerun with the same arguments continues from these files.
- Jev client logs, one JSON line per request (`<out>/<A|B>/logs/<condition>.jsonl` in stage 2): the full request and response and `meta` with run, condition, game, seed, seat, decision number, time, rule set, persona and decoding, so each request joins to its game record.

## How it works

- `figgie/cards.py` and `figgie/posterior.py`: the 12 possible decks, dealing, and card counting (the paper's Algorithm 3 and its likelihood over the 12 decks).
- `figgie/market.py`: rule set B (`Market`) and rule set A (`BookMarket`).
- `figgie/engine.py`: the discrete-event game. A trader wakes after an exponential gap, decides on what it sees, and its order reaches the market `latency` seconds later. Each seat has its own random stream for wake-ups and decisions, so the same seed gives the same deal and the same opponent wake-up gaps in every condition.
- `figgie/agents/`: the rule-based traders and `JevAgent`; `figgie/personalities.py`: the personas.
- `figgie/jev.py`: the HTTP client for Jev, the mock, a shared rate limiter (18 requests per second) and a shared call budget.
- `figgie/records.py`: what each game and run writes to disk.
