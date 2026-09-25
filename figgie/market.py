"""Figgie's open-outcry market: one best bid and one best ask per suit.

A new bid must beat the standing bid and a new ask must undercut the standing
ask. A bid at or above the ask (or an ask at or below the bid) trades straight
away. As in the real game, every trade clears all quotes in all suits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .cards import SUITS


@dataclass(frozen=True)
class Quote:
    price: int
    player: int


@dataclass(frozen=True)
class Trade:
    t: float
    suit: str
    price: int
    buyer: int
    seller: int


@dataclass(frozen=True)
class Order:
    """A quote that was accepted onto the book (public information)."""

    t: float
    player: int
    side: str  # "bid" or "ask"
    suit: str
    price: int


@dataclass(frozen=True)
class Action:
    kind: str  # "pass", "bid", "ask", "buy", "sell"
    suit: str | None = None
    price: int | None = None

    def label(self) -> str:
        if self.kind == "pass":
            return "pass"
        if self.kind in ("buy", "sell"):
            return f"{self.kind}_{self.suit}"
        return f"{self.kind}_{self.suit}_{self.price}"


PASS = Action("pass")


@dataclass
class Market:
    hands: list[dict[str, int]]
    chips: list[int]
    bids: dict[str, Quote | None] = field(default_factory=lambda: {s: None for s in SUITS})
    asks: dict[str, Quote | None] = field(default_factory=lambda: {s: None for s in SUITS})
    trades: list[Trade] = field(default_factory=list)
    orders: list[Order] = field(default_factory=list)

    def apply(self, t: float, player: int, action: Action) -> Trade | Order | None:
        """Apply an action that has reached the exchange. Returns what happened, or None if rejected."""
        s = action.suit
        if action.kind == "pass":
            return None
        if action.kind == "buy":
            ask = self.asks[s]
            if ask is None or ask.player == player:
                return None
            return self._trade(t, s, ask.price, buyer=player, seller=ask.player)
        if action.kind == "sell":
            bid = self.bids[s]
            if bid is None or bid.player == player:
                return None
            return self._trade(t, s, bid.price, buyer=bid.player, seller=player)
        p = action.price
        if p is None or p < 1:
            return None
        if action.kind == "bid":
            if self.chips[player] < p:
                return None
            ask = self.asks[s]
            if ask is not None and p >= ask.price and ask.player != player:
                return self._trade(t, s, ask.price, buyer=player, seller=ask.player)
            if (self.bids[s] is not None and p <= self.bids[s].price) or (ask is not None and p >= ask.price):
                return None
            self.bids[s] = Quote(p, player)
        elif action.kind == "ask":
            if self.hands[player][s] < 1:
                return None
            bid = self.bids[s]
            if bid is not None and p <= bid.price and bid.player != player:
                return self._trade(t, s, bid.price, buyer=bid.player, seller=player)
            if (self.asks[s] is not None and p >= self.asks[s].price) or (bid is not None and p <= bid.price):
                return None
            self.asks[s] = Quote(p, player)
        else:
            raise ValueError(f"unknown action {action.kind}")
        order = Order(t, player, action.kind, s, p)
        self.orders.append(order)
        return order

    def _trade(self, t: float, suit: str, price: int, buyer: int, seller: int) -> Trade | None:
        if self.hands[seller][suit] < 1 or self.chips[buyer] < price:
            return None
        self.hands[seller][suit] -= 1
        self.hands[buyer][suit] += 1
        self.chips[buyer] -= price
        self.chips[seller] += price
        trade = Trade(t, suit, price, buyer, seller)
        self.trades.append(trade)
        for s in SUITS:
            self.bids[s] = None
            self.asks[s] = None
        return trade
