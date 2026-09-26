"""Rule-based strategies.

The core four follow Ozerov, DiSilvio and Luo (2021), "Traders in a Strange
Land", section 2.3. They share one order rule (the paper's Algorithm 2) and
differ only in how they value a card. The market maker is one extension, taken
from a published model (see EXTENSION_SOURCES).

Where the paper leaves a choice open we pick: the suit to act in is drawn at
random among suits the agent can value; prices are whole chips (buy prices are
rounded down, sell prices up); the chartist horizon is TAU trades.
"""

from __future__ import annotations

import math

from ..cards import SUITS
from ..engine import View
from ..market import PASS, Action
from ..posterior import card_values
from .base import Agent

TAU = 3  # chartist look-back, in trades
K_PREY = 4  # bottom-feeder: orders per side it averages (the paper's k)
MM_HALF_SPREAD = 2  # market maker: chips either side of the last trade price

EXTENSION_SOURCES = {
    "market_maker": "Avellaneda & Stoikov (2008), Quantitative Finance 8(3); Glosten & Milgrom (1985), JFE 14(1)",
}


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


def trade_prices(view: View, suit: str) -> list[int]:
    return [t.price for t in view.trades if t.suit == suit]


def buy_at(view: View, suit: str, price: float) -> Action:
    """Limit buy at `price`: take the standing ask if it is at or below, else bid."""
    ask = view.asks[suit]
    if ask is not None and ask.player != view.me and ask.price <= price:
        return Action("buy", suit) if ask.price <= view.chips else PASS
    p = math.floor(price)
    return Action("bid", suit, p) if 1 <= p <= view.chips else PASS


def sell_at(view: View, suit: str, price: float) -> Action:
    """Limit sell at `price`: take the standing bid if it is at or above, else offer."""
    if view.hand[suit] < 1:
        return PASS
    bid = view.bids[suit]
    if bid is not None and bid.player != view.me and bid.price >= price:
        return Action("sell", suit)
    return Action("ask", suit, max(1, math.ceil(price)))


class PaperAgent(Agent):
    """The paper's Algorithm 2 on top of a per-suit value (pb, ps)."""

    def values(self, view: View) -> dict[str, tuple[float, float] | None]:
        raise NotImplementedError

    def decide(self, view: View) -> Action:
        values = {s: v for s, v in self.values(view).items() if v is not None}
        if not values:
            return PASS
        s = self.rng.choice(sorted(values))
        pb, ps = values[s]
        if self.rng.random() < 0.5:
            return buy_at(view, s, self.rng.uniform(0, pb))
        return sell_at(view, s, self.rng.uniform(ps, 2 * ps))


class Fundamentalist(PaperAgent):
    """Card counting (Algorithm 3), posterior over the 12 decks, separate buy and sell values."""

    name = "fundamentalist"

    def values(self, view):
        return card_values(self.counter.known(), view.hand)


class BottomFeeder(PaperAgent):
    """Values a suit at the mean of each opponent's (avg last k buy orders + avg last k sell orders) / 2."""

    name = "bottom_feeder"

    def values(self, view):
        out = {}
        for s in SUITS:
            mids = []
            for p in range(4):
                if p == view.me:
                    continue
                buys, sells = order_history(view, p, s)
                if len(buys) >= K_PREY and len(sells) >= K_PREY:
                    mids.append((sum(buys[-K_PREY:]) / K_PREY + sum(sells[-K_PREY:]) / K_PREY) / 2)
            out[s] = (sum(mids) / len(mids),) * 2 if mids else None
        return out


def order_history(view: View, player: int, suit: str) -> tuple[list[int], list[int]]:
    """Prices of the buy and sell orders `player` sent in `suit`, oldest first: quotes plus orders that traded."""
    events: list[tuple[float, str, int]] = []
    for o in view.orders:
        if o.player == player and o.suit == suit:
            events.append((o.t, o.side, o.price))
    for t in view.trades:
        if t.suit == suit and t.aggressor == player:
            events.append((t.t, "bid" if t.buyer == player else "ask", t.price))
    events.sort(key=lambda e: e[0])
    return [p for _, side, p in events if side == "bid"], [p for _, side, p in events if side == "ask"]


class Chartist(PaperAgent):
    """Chiarella, Iori & Perello (2009) as used in the paper: value = p_t * exp(mean log return * tau)."""

    name = "chartist"

    def values(self, view):
        out = {}
        for s in SUITS:
            p = trade_prices(view, s)
            # r = (1/tau) ln(p[t-1] / p[t-tau-1]); value = p[t] * exp(r * tau) = p[t] * p[t-1] / p[t-tau-1]
            out[s] = (p[-1] * p[-2] / p[-TAU - 2],) * 2 if len(p) >= TAU + 2 else None
        return out


class Noise(PaperAgent):
    """Value = highest bid * e^Z with Z ~ N(0, sigma^2), sigma = 1."""

    name = "noise"
    sigma = 1.0

    def values(self, view):
        out = {}
        for s in SUITS:
            bid = view.bids[s]
            out[s] = (bid.price * math.exp(self.rng.gauss(0, self.sigma)),) * 2 if bid is not None else None
        return out


# --- Extension --------------------------------------------------------------


class MarketMaker(Agent):
    """Quotes around a reference price (last trade, else the mid of the book), shifted against inventory; never takes."""

    name = "market_maker"

    def decide(self, view: View) -> Action:
        refs = {}
        for s in SUITS:
            prices, bid, ask = trade_prices(view, s), view.bids[s], view.asks[s]
            if prices:
                refs[s] = prices[-1]
            elif bid is not None and ask is not None:
                refs[s] = round((bid.price + ask.price) / 2)
        if not refs:
            return PASS
        s = self.rng.choice(sorted(refs))
        ref = refs[s]
        skew = view.hand[s] - self.counter.initial[s]
        bid_p, ask_p = ref - MM_HALF_SPREAD - skew, ref + MM_HALF_SPREAD - skew
        ask, bid = view.asks[s], view.bids[s]
        options = []
        if 1 <= bid_p <= view.chips and (ask is None or bid_p < ask.price):
            options.append(Action("bid", s, bid_p))
        if view.hand[s] > 0 and ask_p >= 1 and (bid is None or ask_p > bid.price):
            options.append(Action("ask", s, ask_p))
        return self.rng.choice(options) if options else PASS


__all__ = ["Fundamentalist", "BottomFeeder", "Chartist", "Noise", "MarketMaker", "EXTENSION_SOURCES", "take_edges"]
