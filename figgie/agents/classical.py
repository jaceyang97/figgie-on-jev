"""Hand-coded strategies modelled on Ozerov, DiSilvio and Luo (2021)."""

from __future__ import annotations

import math
from collections import defaultdict

from ..cards import SUITS
from ..engine import View
from ..market import PASS, Action
from ..posterior import card_values
from .base import Agent


def take_edges(view: View, values: dict[str, tuple[float, float]]) -> dict[Action, float]:
    """Immediate expected gain, in chips, of every take (buy/sell) available right now."""
    edges = {}
    for s in SUITS:
        buy_val, sell_val = values[s]
        ask, bid = view.asks[s], view.bids[s]
        if ask is not None and ask.player != view.me and ask.price <= view.chips:
            edges[Action("buy", s)] = buy_val - ask.price
        if bid is not None and bid.player != view.me and view.hand[s] > 0:
            edges[Action("sell", s)] = bid.price - sell_val
    return edges


class Fundamentalist(Agent):
    """Counts cards, computes the exact posterior, and trades against fair value.

    Takes any price at least `edge` chips better than its value; otherwise posts a
    bid or ask `spread` chips away from value in a random suit.
    """

    name = "fundamentalist"

    def __init__(self, edge: float = 1.0, spread: float = 2.0, **kw):
        super().__init__(**kw)
        self.edge = edge
        self.spread = spread

    def values(self, view: View) -> dict[str, tuple[float, float]]:
        return card_values(self.counter.known(), view.hand)

    def decide(self, view: View) -> Action:
        values = self.values(view)
        edges = take_edges(view, values)
        if edges:
            best, gain = max(edges.items(), key=lambda kv: kv[1])
            if gain >= self.edge:
                return best
        return self.quote(view, values)

    def quote(self, view: View, values) -> Action:
        options = []
        for s in SUITS:
            buy_val, sell_val = values[s]
            bid, ask = view.bids[s], view.asks[s]
            p = math.floor(buy_val - self.spread)
            if p >= 1 and (bid is None or p > bid.price) and (ask is None or p < ask.price) and p <= view.chips:
                options.append(Action("bid", s, p))
            q = math.ceil(sell_val + self.spread)
            if view.hand[s] > 0 and (ask is None or q < ask.price) and (bid is None or q > bid.price):
                options.append(Action("ask", s, max(q, 1)))
        return self.rng.choice(options) if options else PASS


class BottomFeeder(Agent):
    """Infers value from other players' recent quotes and trades against outliers.

    For each suit it averages the last `window` bids and asks posted by each
    other player, then takes any price `edge` chips better than that estimate.
    """

    name = "bottom_feeder"

    def __init__(self, window: int = 4, edge: float = 1.0, **kw):
        super().__init__(**kw)
        self.window = window
        self.edge = edge

    def estimates(self, view: View) -> dict[str, float | None]:
        by_player = defaultdict(list)
        for o in view.orders:
            if o.player != view.me:
                by_player[(o.player, o.suit)].append(o.price)
        est = {}
        for s in SUITS:
            means = [sum(v[-self.window :]) / len(v[-self.window :]) for (p, suit), v in by_player.items() if suit == s]
            est[s] = sum(means) / len(means) if means else None
        return est

    def decide(self, view: View) -> Action:
        est = self.estimates(view)
        best, gain = PASS, self.edge
        for s in SUITS:
            v = est[s]
            if v is None:
                continue
            ask, bid = view.asks[s], view.bids[s]
            if ask is not None and ask.player != view.me and v - ask.price >= gain and ask.price <= view.chips:
                best, gain = Action("buy", s), v - ask.price
            if bid is not None and bid.player != view.me and view.hand[s] > 0 and bid.price - v >= gain:
                best, gain = Action("sell", s), bid.price - v
        return best


class Chartist(Agent):
    """Momentum trader: buys suits whose trade prices are rising, sells falling ones."""

    name = "chartist"

    def __init__(self, lookback: int = 3, **kw):
        super().__init__(**kw)
        self.lookback = lookback

    def decide(self, view: View) -> Action:
        for s in self.rng.sample(SUITS, len(SUITS)):
            prices = [t.price for t in view.trades if t.suit == s][-self.lookback - 1 :]
            if len(prices) < 2:
                continue
            trend = prices[-1] - prices[0]
            ask, bid = view.asks[s], view.bids[s]
            if trend > 0:
                if ask is not None and ask.player != view.me and ask.price <= view.chips:
                    return Action("buy", s)
                return Action("bid", s, prices[-1] + 1)
            if trend < 0 and view.hand[s] > 0:
                if bid is not None and bid.player != view.me:
                    return Action("sell", s)
                return Action("ask", s, max(1, prices[-1] - 1))
        return PASS


class Noise(Agent):
    """Random quotes and takes. A liquidity source and a sanity baseline."""

    name = "noise"

    def decide(self, view: View) -> Action:
        s = self.rng.choice(SUITS)
        r = self.rng.random()
        if r < 0.25:
            return Action("buy", s)
        if r < 0.5 and view.hand[s] > 0:
            return Action("sell", s)
        if r < 0.75:
            return Action("bid", s, self.rng.randint(1, 12))
        if view.hand[s] > 0:
            return Action("ask", s, self.rng.randint(3, 20))
        return PASS
