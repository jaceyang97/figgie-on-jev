"""The two market mechanisms.

B, `Market` (real Figgie, open outcry): one best bid and one best ask per suit.
A new bid must beat the standing bid and a new ask must undercut the standing
ask. A bid at or above the ask (or an ask at or below the bid) trades straight
away. As in the real game, every trade clears all quotes in all suits.

A, `BookMarket` (the Figgie paper's continuous limit order book, its
Algorithm 1): bids and asks at any price rest on the book until they trade or
are cancelled. An incoming order that crosses the best opposite order trades
at the resting order's price, best price first and then oldest first. When an
order comes to trade, the seller must still hold the card and the buyer must
still have the chips; a resting order that fails this check is removed. Each
player keeps at most MAX_ORDERS_PER_SIDE orders per side per suit; a new one
beyond that cancels the player's oldest. An order that would cross the
player's own resting order cancels that resting order first (self-trade
prevention). Every order is for one card.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field

from .cards import SUITS

MAX_ORDERS_PER_SIDE = 5  # rule set A; the paper gives no limit, this is our choice
MECHANISMS = ("A", "B")


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
    aggressor: int | None = None  # the player whose order took the standing quote


@dataclass(frozen=True)
class Order:
    """A quote that was accepted onto the book (public information)."""

    t: float
    player: int
    side: str  # "bid" or "ask"
    suit: str
    price: int
    oid: int = -1


@dataclass(frozen=True)
class Cancel:
    """A resting order that left the book without trading."""

    t: float
    player: int
    oid: int
    suit: str
    side: str
    price: int
    reason: str  # "cancel" (asked for), "lazy" (agent deleted stale order), "limit", "self_trade", "invalid"


@dataclass(frozen=True)
class Action:
    kind: str  # "pass", "bid", "ask", "buy", "sell", "cancel" (rule set A only)
    suit: str | None = None
    price: int | None = None

    def label(self) -> str:
        if self.kind == "pass":
            return "pass"
        if self.kind in ("buy", "sell", "cancel"):
            return f"{self.kind}_{self.suit}"
        return f"{self.kind}_{self.suit}_{self.price}"


PASS = Action("pass")


@dataclass
class Market:
    """Rule set B: real Figgie."""

    hands: list[dict[str, int]]
    chips: list[int]
    bids: dict[str, Quote | None] = field(default_factory=lambda: {s: None for s in SUITS})
    asks: dict[str, Quote | None] = field(default_factory=lambda: {s: None for s in SUITS})
    trades: list[Trade] = field(default_factory=list)
    orders: list[Order] = field(default_factory=list)
    cancels: list[Cancel] = field(default_factory=list)
    mechanism = "B"

    def apply(self, t: float, player: int, action: Action) -> Trade | Order | None:
        """Apply an action that has reached the exchange. Returns what happened, or None if rejected."""
        s = action.suit
        if action.kind in ("pass", "cancel"):
            return None
        if action.kind == "buy":
            ask = self.asks[s]
            if ask is None or ask.player == player:
                return None
            return self._trade(t, s, ask.price, buyer=player, seller=ask.player, aggressor=player)
        if action.kind == "sell":
            bid = self.bids[s]
            if bid is None or bid.player == player:
                return None
            return self._trade(t, s, bid.price, buyer=bid.player, seller=player, aggressor=player)
        p = action.price
        if p is None or p < 1:
            return None
        if action.kind == "bid":
            if self.chips[player] < p:
                return None
            ask = self.asks[s]
            if ask is not None and p >= ask.price and ask.player != player:
                return self._trade(t, s, ask.price, buyer=player, seller=ask.player, aggressor=player)
            if (self.bids[s] is not None and p <= self.bids[s].price) or (ask is not None and p >= ask.price):
                return None
            self.bids[s] = Quote(p, player)
        elif action.kind == "ask":
            if self.hands[player][s] < 1:
                return None
            bid = self.bids[s]
            if bid is not None and p <= bid.price and bid.player != player:
                return self._trade(t, s, bid.price, buyer=bid.player, seller=player, aggressor=player)
            if (self.asks[s] is not None and p >= self.asks[s].price) or (bid is not None and p <= bid.price):
                return None
            self.asks[s] = Quote(p, player)
        else:
            raise ValueError(f"unknown action {action.kind}")
        order = Order(t, player, action.kind, s, p, len(self.orders))
        self.orders.append(order)
        return order

    def _trade(self, t: float, suit: str, price: int, buyer: int, seller: int, aggressor: int | None = None) -> Trade | None:
        if self.hands[seller][suit] < 1 or self.chips[buyer] < price:
            return None
        self.hands[seller][suit] -= 1
        self.hands[buyer][suit] += 1
        self.chips[buyer] -= price
        self.chips[seller] += price
        trade = Trade(t, suit, price, buyer, seller, aggressor)
        self.trades.append(trade)
        for s in SUITS:
            self.bids[s] = None
            self.asks[s] = None
        return trade

    # The same read interface as BookMarket, so agents and the engine need not care which market runs.
    def depth(self, suit: str, n: int = 3) -> dict[str, list[Quote]]:
        bid, ask = self.bids[suit], self.asks[suit]
        return {"bids": [bid] if bid else [], "asks": [ask] if ask else []}

    def open_orders(self, player: int) -> list[Order]:
        """The player's standing quotes (B has no cancel; every trade clears them)."""
        out = []
        for s in SUITS:
            for side, q in (("bid", self.bids[s]), ("ask", self.asks[s])):
                if q is not None and q.player == player:
                    out.append(Order(-1.0, player, side, s, q.price))
        return out

    def cancel_orders(self, t: float, player: int, oids, reason: str = "lazy") -> int:
        return 0


class BookMarket:
    """Rule set A: the paper's continuous limit order book (one card per order)."""

    mechanism = "A"

    def __init__(self, hands: list[dict[str, int]], chips: list[int], max_per_side: int = MAX_ORDERS_PER_SIDE):
        self.hands = hands
        self.chips = chips
        self.max_per_side = max_per_side
        self.trades: list[Trade] = []
        self.orders: list[Order] = []  # every order accepted onto the book, in time order
        self.cancels: list[Cancel] = []
        self.live: dict[int, Order] = {}  # oid -> resting order
        self._seq = itertools.count()
        # heaps of (key, seq, oid); bids keyed by -price so the highest pops first
        self._bids: dict[str, list] = {s: [] for s in SUITS}
        self._asks: dict[str, list] = {s: [] for s in SUITS}

    # --- reading the book ---------------------------------------------------

    def _clean(self, heap: list) -> None:
        while heap and heap[0][2] not in self.live:
            heapq.heappop(heap)

    def _sorted(self, suit: str, side: str) -> list[Order]:
        heap = self._bids[suit] if side == "bid" else self._asks[suit]
        heap[:] = [e for e in heap if e[2] in self.live]  # drop removed orders, so the heap stays small
        heapq.heapify(heap)
        return [self.live[oid] for _, _, oid in sorted(heap)]

    def _best(self, suit: str, side: str) -> Order | None:
        heap = self._bids[suit] if side == "bid" else self._asks[suit]
        self._clean(heap)
        return self.live[heap[0][2]] if heap else None

    @property
    def bids(self) -> dict[str, Quote | None]:
        return {s: (None if (o := self._best(s, "bid")) is None else Quote(o.price, o.player)) for s in SUITS}

    @property
    def asks(self) -> dict[str, Quote | None]:
        return {s: (None if (o := self._best(s, "ask")) is None else Quote(o.price, o.player)) for s in SUITS}

    def depth(self, suit: str, n: int = 3) -> dict[str, list[Quote]]:
        return {side + "s": [Quote(o.price, o.player) for o in self._sorted(suit, side)[:n]] for side in ("bid", "ask")}

    def open_orders(self, player: int) -> list[Order]:
        return sorted((o for o in self.live.values() if o.player == player), key=lambda o: o.oid)

    # --- changing the book --------------------------------------------------

    def _remove(self, t: float, order: Order, reason: str) -> None:
        if self.live.pop(order.oid, None) is not None:
            self.cancels.append(Cancel(t, order.player, order.oid, order.suit, order.side, order.price, reason))

    def cancel_orders(self, t: float, player: int, oids, reason: str = "lazy") -> int:
        n = 0
        for oid in oids:
            o = self.live.get(oid)
            if o is not None and o.player == player:
                self._remove(t, o, reason)
                n += 1
        return n

    def _rest(self, t: float, player: int, side: str, suit: str, price: int) -> Order:
        mine = [o for o in self.open_orders(player) if o.suit == suit and o.side == side]
        while len(mine) >= self.max_per_side:
            self._remove(t, mine.pop(0), "limit")
        order = Order(t, player, side, suit, price, next(self._seq))
        self.live[order.oid] = order
        self.orders.append(order)
        key = -price if side == "bid" else price
        heapq.heappush(self._bids[suit] if side == "bid" else self._asks[suit], (key, order.oid, order.oid))
        return order

    def _fill(self, t: float, suit: str, price: int, buyer: int, seller: int, aggressor: int) -> Trade:
        self.hands[seller][suit] -= 1
        self.hands[buyer][suit] += 1
        self.chips[buyer] -= price
        self.chips[seller] += price
        trade = Trade(t, suit, price, buyer, seller, aggressor)
        self.trades.append(trade)
        return trade

    def _match_buy(self, t: float, player: int, suit: str, limit: float) -> Trade | None | bool:
        """Try to buy one card at or below `limit`. Trade, None (rejected), or False (nothing to cross)."""
        while True:
            best = self._best(suit, "ask")
            if best is None or best.price > limit:
                return False
            if best.player == player:
                self._remove(t, best, "self_trade")
                continue
            if self.hands[best.player][suit] < 1:
                self._remove(t, best, "invalid")
                continue
            if self.chips[player] < best.price:
                return None
            self.live.pop(best.oid)
            return self._fill(t, suit, best.price, buyer=player, seller=best.player, aggressor=player)

    def _match_sell(self, t: float, player: int, suit: str, limit: float) -> Trade | None | bool:
        while True:
            best = self._best(suit, "bid")
            if best is None or best.price < limit:
                return False
            if best.player == player:
                self._remove(t, best, "self_trade")
                continue
            if self.chips[best.player] < best.price:
                self._remove(t, best, "invalid")
                continue
            if self.hands[player][suit] < 1:
                return None
            self.live.pop(best.oid)
            return self._fill(t, suit, best.price, buyer=best.player, seller=player, aggressor=player)

    def apply(self, t: float, player: int, action: Action) -> Trade | Order | int | None:
        """Apply an action that has reached the exchange. Returns what happened, or None if rejected."""
        s, kind = action.suit, action.kind
        if kind == "pass":
            return None
        if kind == "cancel":
            n = self.cancel_orders(t, player, [o.oid for o in self.open_orders(player) if o.suit == s], "cancel")
            return n or None
        if kind == "buy":
            r = self._match_buy(t, player, s, float("inf"))
            return r or None
        if kind == "sell":
            if self.hands[player][s] < 1:
                return None
            r = self._match_sell(t, player, s, float("-inf"))
            return r or None
        p = action.price
        if p is None or p < 1:
            return None
        if kind == "bid":
            if self.chips[player] < p:
                return None
            r = self._match_buy(t, player, s, p)
            return self._rest(t, player, "bid", s, p) if r is False else r
        if kind == "ask":
            if self.hands[player][s] < 1:
                return None
            r = self._match_sell(t, player, s, p)
            return self._rest(t, player, "ask", s, p) if r is False else r
        raise ValueError(f"unknown action {kind}")


def make_market(mechanism: str, hands, chips):
    if mechanism == "A":
        return BookMarket(hands=hands, chips=chips)
    if mechanism == "B":
        return Market(hands=hands, chips=chips)
    raise ValueError(f"unknown mechanism {mechanism!r}; choose A or B")
