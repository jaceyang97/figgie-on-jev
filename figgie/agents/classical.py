"""Rule-based strategies.

The core four follow Ozerov, DiSilvio and Luo (2021), "Traders in a Strange
Land", section 2.3. They share one order rule (the paper's Algorithm 2) and
differ only in how they value a card. The two market makers are extensions,
each taken from a published model (see EXTENSION_SOURCES).

Where the paper leaves a choice open we pick: the suit to act in is drawn at
random among suits the agent can value; prices are whole chips (buy prices are
rounded down, sell prices up); the chartist horizon is TAU trades.
"""

from __future__ import annotations

import math

from ..cards import SUITS
from ..engine import View
from ..market import PASS, Action
from ..cards import GOAL_CARD_PAYOUT
from ..posterior import card_values
from .base import Agent

TAU = 3  # chartist look-back, in trades
K_PREY = 4  # bottom-feeder: orders per side it averages (the paper's k)

# Market-maker parameters. The source papers model other markets, so these values are our choices for Figgie.
GM_INFORMED = 0.3  # Glosten-Milgrom: share of traders who know the goal suit (mu)
AS_GAMMA = 0.1  # Avellaneda-Stoikov: risk aversion (gamma)
AS_K = 1.5  # Avellaneda-Stoikov: how fast fill chances fall with distance from the mid (k)
AS_SIGMA2_DEFAULT = 4.0  # Avellaneda-Stoikov: price variance (chips^2) before a suit has 3 trades

EXTENSION_SOURCES = {
    "market_maker_gm": "Glosten & Milgrom (1985), Journal of Financial Economics 14(1), 71-100",
    "market_maker_as": "Avellaneda & Stoikov (2008), Quantitative Finance 8(3), 217-224",
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
    """Card counting (Algorithm 3), probability of each of the 12 decks, separate buy and sell values."""

    name = kind = "fundamentalist"

    def values(self, view):
        return card_values(self.counter.known(), view.hand)

    def cancel_now(self, view):
        """The paper's Algorithm 4: delete own bids above the buy value and own asks below the sell value."""
        values = self.values(view)
        stale = []
        for o in view.my_orders:
            pb, ps = values[o.suit]
            if (o.side == "bid" and o.price > pb) or (o.side == "ask" and o.price < ps):
                stale.append(o.oid)
        return stale


class BottomFeeder(PaperAgent):
    """Values a suit at the mean over its prey of (avg last k buy orders + avg last k sell orders) / 2.

    The prey are the seats labelled `prey_kind` (the paper's main tests: all fundamentalists). The tested
    seat of an experiment is labelled "tested", so it is never prey and the opponents act the same whatever
    sits there.
    """

    name = kind = "bottom_feeder"

    def __init__(self, prey_kind: str = "fundamentalist", **kw):
        super().__init__(**kw)
        self.prey_kind = prey_kind

    def prey(self) -> list[int]:
        return [p for p, lab in enumerate(self.labels) if lab == self.prey_kind and p != self.me]

    def _history(self, view: View, player: int, suit: str) -> tuple[list[int], list[int]]:
        """order_history(view, player, suit), kept up to date incrementally (the same result, much faster)."""
        cache = self.__dict__.setdefault("_hist", {"n_orders": 0, "n_trades": 0, "h": {}})
        new = [(o.t, o.player, o.suit, o.side, o.price) for o in view.orders[cache["n_orders"]:]]
        new += [(t.t, t.aggressor, t.suit, "bid" if t.buyer == t.aggressor else "ask", t.price)
                for t in view.trades[cache["n_trades"]:] if t.aggressor is not None]
        new.sort(key=lambda e: e[0])
        for _, p, s, side, price in new:
            cache["h"].setdefault((p, s), ([], []))[0 if side == "bid" else 1].append(price)
        cache["n_orders"], cache["n_trades"] = len(view.orders), len(view.trades)
        return cache["h"].get((player, suit), ([], []))

    def values(self, view):
        out = {}
        prey = self.prey()
        for s in SUITS:
            mids = []
            for p in prey:
                buys, sells = self._history(view, p, s)
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

    name = kind = "chartist"

    def values(self, view):
        out = {}
        for s in SUITS:
            p = trade_prices(view, s)
            # r = (1/tau) ln(p[t-1] / p[t-tau-1]); value = p[t] * exp(r * tau) = p[t] * p[t-1] / p[t-tau-1]
            out[s] = (p[-1] * p[-2] / p[-TAU - 2],) * 2 if len(p) >= TAU + 2 else None
        return out


class Noise(PaperAgent):
    """Value = highest bid * e^Z with Z ~ N(0, sigma^2), sigma = 1."""

    name = kind = "noise"
    sigma = 1.0

    def values(self, view):
        out = {}
        for s in SUITS:
            bid = view.bids[s]
            out[s] = (bid.price * math.exp(self.rng.gauss(0, self.sigma)),) * 2 if bid is not None else None
        return out


# --- Extensions: two market makers ------------------------------------------


class QuotingMarketMaker(Agent):
    """Posts one bid and one ask per suit from `quotes`; never takes another player's order.

    In rule set B it posts, in a random suit, whichever side the market accepts (a quote must beat the
    standing one). In rule set A it keeps one bid and one ask per suit at its current quotes and deletes its
    own orders at other prices before deciding.
    """

    def quotes(self, view: View) -> dict[str, tuple[int, int]]:
        raise NotImplementedError

    def _int_quotes(self, view: View) -> dict[str, tuple[int, int]]:
        out = {}
        for s, (qb, qa) in self.quotes(view).items():
            bid, ask = math.floor(qb), max(1, math.ceil(qa))
            out[s] = (bid, max(ask, bid + 1))
        return out

    def cancel_now(self, view: View) -> list[int]:
        quotes = self._int_quotes(view)
        stale = []
        for o in view.my_orders:
            q = quotes.get(o.suit)
            if q is None or o.price != (q[0] if o.side == "bid" else q[1]):
                stale.append(o.oid)
        return stale

    def decide(self, view: View) -> Action:
        options = []
        for s, (bid_p, ask_p) in sorted(self._int_quotes(view).items()):
            bid, ask = view.bids[s], view.asks[s]
            mine = {(o.side, o.price) for o in view.my_orders if o.suit == s}
            if 1 <= bid_p <= view.chips and (ask is None or bid_p < ask.price) and ("bid", bid_p) not in mine:
                if view.mechanism == "A" or bid is None or bid_p > bid.price:
                    options.append(Action("bid", s, bid_p))
            if view.hand[s] > 0 and (bid is None or ask_p > bid.price) and ("ask", ask_p) not in mine:
                if view.mechanism == "A" or ask is None or ask_p < ask.price:
                    options.append(Action("ask", s, ask_p))
        return self.rng.choice(options) if options else PASS


class MarketMakerGM(QuotingMarketMaker):
    """Glosten & Milgrom (1985): quotes are the card's expected value given that the next trader buys or sells.

    A card of suit s pays 10 if s is the goal suit. The belief starts from card counting and is updated with
    each other player's trade direction: a share `mu` of traders know the goal suit and buy only it, the rest
    buy or sell at random. So a buy in suit s multiplies P(goal = s) by (1 + mu)/2 and every other suit by
    (1 - mu)/2, and a sell does the reverse. Ask = 10 P(goal = s | buy), bid = 10 P(goal = s | sell).
    """

    name = kind = "market_maker_gm"

    def __init__(self, mu: float = GM_INFORMED, **kw):
        super().__init__(**kw)
        self.mu = mu

    def start(self, *a, **kw):
        super().start(*a, **kw)
        self.flow = {s: 1.0 for s in SUITS}

    def on_trade(self, trade):
        super().on_trade(trade)
        if trade.aggressor is None or trade.aggressor == self.me:
            return
        up, down = (1 + self.mu) / 2, (1 - self.mu) / 2
        bought = trade.aggressor == trade.buyer
        for s in SUITS:
            self.flow[s] *= (up if bought else down) if s == trade.suit else (down if bought else up)
        z = sum(self.flow.values())
        self.flow = {s: v / z for s, v in self.flow.items()}

    def belief(self) -> dict[str, float]:
        cc = self.counter.goal_probabilities()
        w = {s: cc[s] * self.flow[s] for s in SUITS}
        z = sum(w.values())
        return {s: v / z for s, v in w.items()} if z > 0 else cc

    def quotes(self, view):
        up, down = (1 + self.mu) / 2, (1 - self.mu) / 2
        out = {}
        for s, pi in self.belief().items():
            ask = GOAL_CARD_PAYOUT * pi * up / (pi * up + (1 - pi) * down)
            bid = GOAL_CARD_PAYOUT * pi * down / (pi * down + (1 - pi) * up)
            out[s] = (bid, ask)
        return out


class MarketMakerAS(QuotingMarketMaker):
    """Avellaneda & Stoikov (2008): quotes around a reservation price that moves against inventory.

    mid = the average of the best bid and ask, else the last trade price. q = cards held minus cards dealt.
    sigma^2 = variance of the changes between consecutive trade prices in the suit. tau = fraction of the
    game left. r = mid - q gamma sigma^2 tau; half spread d = gamma sigma^2 tau / 2 + ln(1 + gamma/k) / gamma.
    """

    name = kind = "market_maker_as"

    def __init__(self, gamma: float = AS_GAMMA, k: float = AS_K, **kw):
        super().__init__(**kw)
        self.gamma, self.k = gamma, k

    def quotes(self, view):
        out = {}
        tau = max(0.0, 1.0 - view.progress)
        for s in SUITS:
            prices, bid, ask = trade_prices(view, s), view.bids[s], view.asks[s]
            if bid is not None and ask is not None:
                mid = (bid.price + ask.price) / 2
            elif prices:
                mid = prices[-1]
            else:
                continue
            diffs = [b - a for a, b in zip(prices, prices[1:])]
            if len(diffs) >= 2:
                m = sum(diffs) / len(diffs)
                sigma2 = sum((d - m) ** 2 for d in diffs) / (len(diffs) - 1)
            else:
                sigma2 = AS_SIGMA2_DEFAULT
            q = view.hand[s] - self.counter.initial[s]
            r = mid - q * self.gamma * sigma2 * tau
            d = self.gamma * sigma2 * tau / 2 + math.log(1 + self.gamma / self.k) / self.gamma
            out[s] = (r - d, r + d)
        return out


__all__ = ["Fundamentalist", "BottomFeeder", "Chartist", "Noise", "MarketMakerGM", "MarketMakerAS",
           "EXTENSION_SOURCES", "take_edges"]
