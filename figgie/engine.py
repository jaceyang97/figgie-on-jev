"""Discrete-event Figgie game with per-agent speed.

Each agent wakes at random times (exponential gaps, rate `wake_rate`). On
waking it sees the market as it is and picks an action. The action reaches
the exchange `latency` seconds later and is checked against the book as it is
*then*, so a slow agent can lose a race. The agent wakes again only after its
order has landed. This is the paper's latency model with our latency equal to
the paper's 2 Lambda.

Randomness is split into streams so that games can be paired across
conditions: the game seed fixes the deal, then one decision stream and one
wake-time stream per seat. Changing the agent in one seat therefore does not
change the other seats' wake times or random draws; only their view of the
market changes, because the market changes.

A game ends after `duration` seconds or, in the paper's mode, after
`max_events` events (consideration and add-order events, as the paper counts
them).
"""

from __future__ import annotations

import heapq
import itertools
import random
from dataclasses import dataclass, field

from .cards import ANTE, GOAL_CARD_PAYOUT, N_PLAYERS, STARTING_CHIPS, SUITS, Deck, deal
from .market import Action, Cancel, Order, Quote, Trade, make_market


@dataclass(frozen=True)
class View:
    """What one seat can see when it decides."""

    t: float
    t_end: float
    me: int
    hand: dict[str, int]
    chips: int
    bids: dict[str, Quote | None]
    asks: dict[str, Quote | None]
    trades: tuple[Trade, ...]
    orders: tuple[Order, ...]
    mechanism: str = "B"
    depth: dict | None = None  # rule set A: suit -> {"bids": [Quote...], "asks": [...]}, best first
    my_orders: tuple[Order, ...] = ()  # this seat's resting orders
    progress: float = 0.0  # fraction of the game elapsed (by time, or by events in the paper's mode)

    @property
    def time_left(self) -> float:
        return max(0.0, self.t_end - self.t)


@dataclass
class GameResult:
    deck: Deck
    initial_hands: list[dict[str, int]]
    final_hands: list[dict[str, int]]
    pnl: list[float]
    trades: list[Trade]
    orders: list[Order]
    names: list[str]
    decisions: list[int]
    rejected: list[int]
    mechanism: str = "B"
    passes: list[int] = field(default_factory=list)
    cancels: list[Cancel] = field(default_factory=list)
    final_chips: list[int] = field(default_factory=list)
    payout: list[float] = field(default_factory=list)
    events: int = 0
    t_end: float = 0.0
    labels: list[str] = field(default_factory=list)
    speeds: list[tuple[float, float]] = field(default_factory=list)


def payouts(deck: Deck, hands: list[dict[str, int]]) -> list[float]:
    """Chips each player receives from the pot: 10 per goal card plus a share of the majority bonus."""
    goal = deck.goal
    pay = [float(GOAL_CARD_PAYOUT * h[goal]) for h in hands]
    top = max(h[goal] for h in hands)
    winners = [p for p, h in enumerate(hands) if h[goal] == top]
    for p in winners:
        pay[p] += deck.bonus / len(winners)
    return pay


def settle(deck: Deck, hands: list[dict[str, int]], chips: list[int]) -> list[float]:
    """Final chips minus starting chips, after goal payouts and the majority bonus."""
    return [c + p - STARTING_CHIPS for c, p in zip(chips, payouts(deck, hands))]


def _view(market, t: float, t_end: float, p: int, progress: float) -> View:
    depth = None
    if market.mechanism == "A":
        depth = {s: market.depth(s, 3) for s in SUITS}
    return View(
        t=t, t_end=t_end, me=p, hand=dict(market.hands[p]), chips=market.chips[p],
        bids=dict(market.bids), asks=dict(market.asks),
        trades=tuple(market.trades), orders=tuple(market.orders),
        mechanism=market.mechanism, depth=depth, my_orders=tuple(market.open_orders(p)), progress=progress,
    )


def play_game(agents, rng: random.Random, duration: float = 240.0, mechanism: str = "B",
              max_events: int | None = None, tested_seat: int | None = None) -> GameResult:
    """Play one game. `tested_seat` is hidden from the other agents' type labels (see agents.base.Agent.start)."""
    assert len(agents) == N_PLAYERS
    deck, hands = deal(rng)
    initial = [dict(h) for h in hands]
    market = make_market(mechanism, hands, [STARTING_CHIPS - ANTE] * N_PLAYERS)
    agent_seeds = [rng.random() for _ in agents]
    wake_rngs = [random.Random(rng.random()) for _ in agents]
    labels = [("tested" if p == tested_seat else a.kind) for p, a in enumerate(agents)]
    for p, agent in enumerate(agents):
        agent.start(p, dict(hands[p]), N_PLAYERS, random.Random(agent_seeds[p]), labels=labels)

    t_end = float("inf") if max_events else duration
    seq = itertools.count()
    events: list[tuple[float, int, str, int, Action | None]] = []
    for p, agent in enumerate(agents):
        heapq.heappush(events, (wake_rngs[p].expovariate(agent.wake_rate), next(seq), "wake", p, None))

    decisions = [0] * N_PLAYERS
    passes = [0] * N_PLAYERS
    rejected = [0] * N_PLAYERS
    n_events = 0
    t_last = 0.0
    while events:
        t, _, kind, p, action = heapq.heappop(events)
        if max_events is not None:
            if n_events >= max_events:
                break
        elif t >= duration:
            break
        n_events += 1
        t_last = t
        progress = n_events / max_events if max_events else t / duration
        agent = agents[p]
        if kind == "wake":
            stale = agent.cancel_now(_view(market, t, t_end, p, progress)) if market.mechanism == "A" else []
            if stale:
                market.cancel_orders(t, p, stale, "lazy")
            action = agent.decide(_view(market, t, t_end, p, progress))
            decisions[p] += 1
            latency = agent.latency()
            if action.kind == "pass":
                passes[p] += 1
                heapq.heappush(events, (t + wake_rngs[p].expovariate(agent.wake_rate), next(seq), "wake", p, None))
            else:
                heapq.heappush(events, (t + latency, next(seq), "arrive", p, action))
        else:
            result = market.apply(t, p, action)
            if result is None:
                rejected[p] += 1
            elif isinstance(result, Trade):
                for a in agents:
                    a.on_trade(result)
            heapq.heappush(events, (t + wake_rngs[p].expovariate(agent.wake_rate), next(seq), "wake", p, None))

    pay = payouts(deck, market.hands)
    pnl = [c + x - STARTING_CHIPS for c, x in zip(market.chips, pay)]
    return GameResult(
        deck=deck, initial_hands=initial, final_hands=[dict(h) for h in market.hands], pnl=pnl,
        trades=list(market.trades), orders=list(market.orders),
        names=[a.name for a in agents], decisions=decisions, rejected=rejected,
        mechanism=market.mechanism, passes=passes, cancels=list(market.cancels),
        final_chips=list(market.chips), payout=pay, events=n_events, t_end=t_last, labels=labels,
        speeds=[(a.wake_rate, a.latency()) for a in agents],
    )


__all__ = ["View", "GameResult", "play_game", "settle", "payouts", "SUITS"]
