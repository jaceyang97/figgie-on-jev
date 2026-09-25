"""Discrete-event Figgie game with per-agent latency.

Each agent wakes at random times (exponential gaps). On waking it sees the
market as it is and picks an action. The action reaches the exchange
`latency` seconds later and is checked against the book as it is *then*, so a
slow agent can lose a race: the ask it wanted to lift may be gone. The agent
wakes again only after its order has landed. This follows the paper's idea of
modelling latency as a delay between seeing and acting, without storing
market snapshots.
"""

from __future__ import annotations

import heapq
import itertools
import random
from dataclasses import dataclass

from .cards import ANTE, GOAL_CARD_PAYOUT, N_PLAYERS, STARTING_CHIPS, SUITS, Deck, deal
from .market import Action, Market, Order, Quote, Trade


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


def settle(deck: Deck, hands: list[dict[str, int]], chips: list[int]) -> list[float]:
    """Final chips minus starting chips, after goal payouts and the majority bonus."""
    goal = deck.goal
    final = [float(c) for c in chips]
    for p, h in enumerate(hands):
        final[p] += GOAL_CARD_PAYOUT * h[goal]
    top = max(h[goal] for h in hands)
    winners = [p for p, h in enumerate(hands) if h[goal] == top]
    for p in winners:
        final[p] += deck.bonus / len(winners)
    return [f - STARTING_CHIPS for f in final]


def play_game(agents, rng: random.Random, duration: float = 240.0) -> GameResult:
    assert len(agents) == N_PLAYERS
    deck, hands = deal(rng)
    initial = [dict(h) for h in hands]
    market = Market(hands=hands, chips=[STARTING_CHIPS - ANTE] * N_PLAYERS)
    for p, agent in enumerate(agents):
        agent.start(p, dict(hands[p]), N_PLAYERS, random.Random(rng.random()))

    seq = itertools.count()
    events: list[tuple[float, int, str, int, Action | None]] = []
    for p, agent in enumerate(agents):
        heapq.heappush(events, (rng.expovariate(agent.wake_rate), next(seq), "wake", p, None))

    decisions = [0] * N_PLAYERS
    rejected = [0] * N_PLAYERS
    while events:
        t, _, kind, p, action = heapq.heappop(events)
        if t >= duration:
            break
        agent = agents[p]
        if kind == "wake":
            view = View(
                t=t, t_end=duration, me=p, hand=dict(market.hands[p]), chips=market.chips[p],
                bids=dict(market.bids), asks=dict(market.asks),
                trades=tuple(market.trades), orders=tuple(market.orders),
            )
            action = agent.decide(view)
            decisions[p] += 1
            latency = agent.latency()
            if action.kind != "pass":
                heapq.heappush(events, (t + latency, next(seq), "arrive", p, action))
            heapq.heappush(events, (t + latency + rng.expovariate(agent.wake_rate), next(seq), "wake", p, None))
        else:
            result = market.apply(t, p, action)
            if result is None:
                rejected[p] += 1
            elif isinstance(result, Trade):
                for a in agents:
                    a.on_trade(result)

    pnl = settle(deck, market.hands, market.chips)
    return GameResult(
        deck=deck, initial_hands=initial, final_hands=[dict(h) for h in market.hands], pnl=pnl,
        trades=list(market.trades), orders=list(market.orders),
        names=[a.name for a in agents], decisions=decisions, rejected=rejected,
    )


__all__ = ["View", "GameResult", "play_game", "settle", "SUITS"]
