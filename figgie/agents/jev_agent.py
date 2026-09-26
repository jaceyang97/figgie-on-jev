"""A Figgie agent whose every decision is a Jev Choice query."""

from __future__ import annotations

from ..cards import SUITS
from ..engine import View
from ..market import PASS, Action
from ..personalities import ALL_PERSONAS, persona_text
from .base import Agent

RULES_B = (
    "Figgie rules. There are 4 players and a 40-card deck in four suits: spades, clubs, hearts and diamonds. "
    "Spades and clubs are black; hearts and diamonds are red. One suit has 12 cards, one suit has 8 cards and the "
    "other two suits have 10 cards each. Which suit has how many cards is secret and random. The goal suit is the "
    "other suit of the same colour as the 12-card suit. Each player starts with 350 chips, puts 50 chips into a "
    "200-chip pot, and is dealt 10 random cards. "
    "Trading lasts 240 seconds. Each suit has at most one standing bid and one standing ask. A new bid must be "
    "higher than the standing bid and a new ask must be lower than the standing ask. Buying takes the standing "
    "ask; selling takes the standing bid. You can only sell cards you hold and only bid or buy with chips you have. "
    "After any trade, all bids and asks in every suit are cancelled. Orders reach the market after a short delay, "
    "so an order can fail if the book has changed. "
    "When trading ends, the goal suit is revealed. The pot pays 10 chips for each goal-suit card a player holds, "
    "and the rest of the pot goes to the player holding the most goal-suit cards, split equally if tied. "
    "Cards of other suits pay nothing."
)

# Rule set A keeps Figgie's cards, chips and payout but uses a limit order book instead of open outcry.
RULES_A = (
    "Figgie rules. There are 4 players and a 40-card deck in four suits: spades, clubs, hearts and diamonds. "
    "Spades and clubs are black; hearts and diamonds are red. One suit has 12 cards, one suit has 8 cards and the "
    "other two suits have 10 cards each. Which suit has how many cards is secret and random. The goal suit is the "
    "other suit of the same colour as the 12-card suit. Each player starts with 350 chips, puts 50 chips into a "
    "200-chip pot, and is dealt 10 random cards. "
    "Trading lasts 240 seconds. Each suit has an order book. Bids and asks for one card at any price stay on the "
    "book until they trade or their owner cancels them. A new bid at or above the lowest ask trades at that ask's "
    "price, and a new ask at or below the highest bid trades at that bid's price; the best price trades first, and "
    "the oldest order first at the same price. Buying takes the lowest ask; selling takes the highest bid. A trade "
    "removes only the two orders that traded. Each player can have at most 5 bids and 5 asks in a suit; a sixth "
    "cancels that player's oldest one. A new order that would trade with your own order cancels your order instead. "
    "When an order comes to trade, the seller must still hold the card and the buyer must still have the chips; "
    "otherwise the resting order is removed. Cancelling removes all your orders in a suit. You can only sell cards "
    "you hold and only bid or buy with chips you have. Orders reach the market after a short delay, so an order "
    "can fail if the book has changed. "
    "When trading ends, the goal suit is revealed. The pot pays 10 chips for each goal-suit card a player holds, "
    "and the rest of the pot goes to the player holding the most goal-suit cards, split equally if tied. "
    "Cards of other suits pay nothing."
)
RULES = {"A": RULES_A, "B": RULES_B}

# Whole-chip prices Jev can quote: every price 1-20, then every second price to 40. 30 levels keep the menu
# at most 1 + 8 + 4 + 4 x 2 x 30 = 253 options, under Jev's limit of 255.
PRICE_LADDER = tuple(range(1, 21)) + tuple(range(22, 41, 2))


def describe_state(view: View, known: dict[str, int] | None = None, goal_probs: dict[str, float] | None = None,
                   n_trades: int = 12) -> dict:
    """The JSON state sent to Jev: this seat's hand and chips, the book and recent trades.

    Everything here is observed fact. `known` (cards known to exist, deduced by
    code) and `goal_probs` (the exact posterior) are derived and only added
    when given, for ablations.
    """
    book = {}
    for s in SUITS:
        if view.mechanism == "A":
            depth = view.depth[s]
            book[s] = {
                "bids_best_first": [{"price": q.price, "mine": q.player == view.me} for q in depth["bids"]],
                "asks_best_first": [{"price": q.price, "mine": q.player == view.me} for q in depth["asks"]],
            }
            continue
        bid, ask = view.bids[s], view.asks[s]
        book[s] = {
            "best_bid": None if bid is None else bid.price,
            "best_bid_is_mine": bid is not None and bid.player == view.me,
            "best_ask": None if ask is None else ask.price,
            "best_ask_is_mine": ask is not None and ask.player == view.me,
        }
    trades = [
        f"{t.suit} traded at {t.price} ("
        + ("you bought" if t.buyer == view.me else "you sold" if t.seller == view.me else f"P{t.buyer} bought from P{t.seller}")
        + ("" if t.aggressor is None else ", buyer took the ask" if t.aggressor == t.buyer else ", seller hit the bid")
        + ")"
        for t in view.trades[-n_trades:]
    ]
    state = {
        "you_are": f"P{view.me}",
        "my_hand": dict(view.hand),
        "my_chips": view.chips,
        "seconds_left": round(view.time_left),
        "order_book": book,
        "recent_trades_oldest_first": trades,
    }
    if view.mechanism == "A":
        state["my_resting_orders"] = [f"{o.side} {o.suit} {o.price}" for o in view.my_orders]
    if known is not None:
        state["cards_known_to_exist_derived"] = dict(known)
    if goal_probs is not None:
        state["goal_suit_probability_derived"] = {s: round(p, 3) for s, p in goal_probs.items()}
    return state


def describe_log(view: View) -> list[str]:
    """Every public event so far, oldest first, one line each: trades and quotes accepted onto the book."""
    who = lambda p: "you" if p == view.me else f"P{p}"  # noqa: E731
    events = [(o.t, f"t={o.t:.1f}s {who(o.player)} {o.side} {o.suit} {o.price}") for o in view.orders]
    events += [(t.t, f"t={t.t:.1f}s trade {t.suit} {t.price}: {who(t.buyer)} bought from {who(t.seller)}"
                + ("" if t.aggressor is None else f", {who(t.aggressor)} took the "
                   + ("ask" if t.aggressor == t.buyer else "bid")))
               for t in view.trades]
    return [line for _, line in sorted(events, key=lambda e: e[0])]


def _signed(counts: dict[str, int]) -> dict[str, int]:
    return {s: n for s, n in counts.items() if n}


def describe_history(view: View, n_prices: int = 6) -> dict:
    """The whole public game log so far, compressed to per-suit and per-player summaries.

    Every trade and every quote accepted onto the book is public in Figgie, so
    this adds no hidden information; it only spares Jev from reading a raw log.
    Only counts, sums and prices taken from the log: no labels or guesses.
    """
    suits = {}
    for s in SUITS:
        prices = [t.price for t in view.trades if t.suit == s]
        bids = [o.price for o in view.orders if o.suit == s and o.side == "bid"]
        asks = [o.price for o in view.orders if o.suit == s and o.side == "ask"]
        info = {"trades": len(prices), "bids_posted": len(bids), "asks_posted": len(asks)}
        if prices:
            info["avg_price"] = round(sum(prices) / len(prices), 1)
            info["last_prices_oldest_first"] = prices[-n_prices:]
        if bids:
            info["highest_bid_ever"] = max(bids)
        if asks:
            info["lowest_ask_ever"] = min(asks)
        suits[s] = info

    players = {}
    for p in range(4):
        net = {s: 0 for s in SUITS}
        cash = 0
        for t in view.trades:
            if t.buyer == p:
                net[t.suit] += 1
                cash -= t.price
            elif t.seller == p:
                net[t.suit] -= 1
                cash += t.price
        bid_counts = {s: sum(1 for o in view.orders if o.player == p and o.side == "bid" and o.suit == s) for s in SUITS}
        ask_counts = {s: sum(1 for o in view.orders if o.player == p and o.side == "ask" and o.suit == s) for s in SUITS}
        info = {
            "net_cards_bought": _signed(net),
            "chips_from_trading": cash,
            "bids_posted_by_suit": _signed(bid_counts),
            "asks_posted_by_suit": _signed(ask_counts),
        }
        if p == view.me:
            info["starting_hand"] = {s: view.hand[s] - net[s] for s in SUITS}
            players["me"] = info
        else:
            players[f"P{p}"] = info
    return {"elapsed_seconds": round(view.t), "total_trades": len(view.trades), "suits": suits, "players": players}


def action_menu(view: View) -> dict[str, tuple[Action, str]]:
    """Every legal action as label -> (action, description). Jev picks one label."""
    menu = {"pass": (PASS, "Do nothing this turn.")}
    if view.mechanism == "A":
        return _action_menu_a(view, menu)
    for s in SUITS:
        bid, ask = view.bids[s], view.asks[s]
        if ask is not None and ask.player != view.me and ask.price <= view.chips:
            a = Action("buy", s)
            menu[a.label()] = (a, f"Buy one {s} at the standing ask of {ask.price}.")
        if bid is not None and bid.player != view.me and view.hand[s] > 0:
            a = Action("sell", s)
            menu[a.label()] = (a, f"Sell one {s} at the standing bid of {bid.price}.")
        for p in PRICE_LADDER:
            if (bid is None or p > bid.price) and (ask is None or p < ask.price) and p <= view.chips:
                a = Action("bid", s, p)
                menu[a.label()] = (a, f"Bid {p} for one {s}.")
            if view.hand[s] > 0 and (ask is None or p < ask.price) and (bid is None or p > bid.price):
                a = Action("ask", s, p)
                menu[a.label()] = (a, f"Offer one {s} at {p}.")
    return menu


def _action_menu_a(view: View, menu: dict) -> dict:
    """Rule set A: take the best price, rest a bid or ask at a ladder price that does not cross, or cancel."""
    mine = {o.suit for o in view.my_orders}
    for s in SUITS:
        bid, ask = view.bids[s], view.asks[s]
        if ask is not None and ask.player != view.me and ask.price <= view.chips:
            a = Action("buy", s)
            menu[a.label()] = (a, f"Buy one {s} at the lowest ask of {ask.price}.")
        if bid is not None and bid.player != view.me and view.hand[s] > 0:
            a = Action("sell", s)
            menu[a.label()] = (a, f"Sell one {s} at the highest bid of {bid.price}.")
        if s in mine:
            a = Action("cancel", s)
            menu[a.label()] = (a, f"Cancel all my orders in {s}.")
        for p in PRICE_LADDER:
            if (ask is None or p < ask.price) and p <= view.chips:
                a = Action("bid", s, p)
                menu[a.label()] = (a, f"Bid {p} for one {s}.")
            if view.hand[s] > 0 and (bid is None or p > bid.price):
                a = Action("ask", s, p)
                menu[a.label()] = (a, f"Offer one {s} at {p}.")
    return menu


def action_question(menu: dict, personality: str, mechanism: str = "B") -> dict:
    persona = persona_text(personality)
    instructions = RULES[mechanism] + ("\n\n" + persona if persona else "") + "\n\nYou are the player in the state. Which action do you take now?"
    return {"type": "choice", "instructions": instructions, "criteria": {k: d for k, (_, d) in menu.items()}}


def goal_question(mechanism: str = "B") -> dict:
    return {
        "type": "choice",
        "instructions": RULES[mechanism] + "\n\nFrom this player's point of view, which suit is the goal suit?",
        "criteria": {s: f"The goal suit is {s}." for s in SUITS},
    }


def add_context(view: View, known=None, goal_probs=None, log: bool = False, summary: bool = False,
                my_decisions: list[str] | None = None) -> dict:
    """The state with optional context parts: the full event log, a numeric summary of it, and derived fields."""
    state = describe_state(view, known, goal_probs)
    if log:
        del state["recent_trades_oldest_first"]  # the full log already has them
        state["all_events_oldest_first"] = describe_log(view)
    if summary:
        state["game_summary"] = describe_history(view)
    if my_decisions:
        state["my_recent_decisions"] = my_decisions
    return state


def hierarchical_choice(probs: dict[str, float]) -> str:
    """Pick the action type with the most total probability, then the suit, then the single best label.

    A plain argmax over the whole menu favours 'pass': it is one option, while
    the probability of bidding is split over up to 120 (suit, price) options.
    Summing within type and then suit first compares like with like.
    """
    def best(groups: dict[str, float]) -> str:
        return max(groups, key=groups.get)

    by_kind: dict[str, float] = {}
    for label, p in probs.items():
        by_kind[label.split("_")[0]] = by_kind.get(label.split("_")[0], 0.0) + p
    kind = best(by_kind)
    labels = {k: p for k, p in probs.items() if k.split("_")[0] == kind}
    by_suit: dict[str, float] = {}
    for label, p in labels.items():
        suit = label.split("_")[1] if "_" in label else ""
        by_suit[suit] = by_suit.get(suit, 0.0) + p
    suit = best(by_suit)
    return best({k: p for k, p in labels.items() if (k.split("_")[1] if "_" in k else "") == suit})


DECODES = ("sample", "argmax", "hierarchical")


class JevAgent(Agent):
    """Asks Jev which action to take and, in the same request, which suit is the goal suit.

    The action is drawn at random with Jev's probabilities (`decode="sample"`, the experiment's setting);
    "argmax" and "hierarchical" are kept to reproduce earlier runs. Context flags add the full event log
    (`log`), a numeric summary of it (`summary`), or code-derived fields for ablations (`known`, `assist`).
    Every decision is kept in `trace` for the game record, and every request carries `meta` (run, condition,
    game, seat, time, decision number) into the client's log so calls can be joined to games.
    """

    kind = "jev"

    def __init__(self, client, personality: str = "neutral", assist: bool = False, known: bool = False,
                 log: bool = False, summary: bool = False, decode: str = "sample", meta: dict | None = None,
                 ask_goal: bool = True, **kw):
        super().__init__(**kw)
        if personality not in ALL_PERSONAS:
            raise ValueError(f"unknown personality {personality!r}; choose from {sorted(ALL_PERSONAS)}")
        if decode not in DECODES:
            raise ValueError(f"unknown decode {decode!r}; choose from {DECODES}")
        self.client = client
        self.personality = personality
        self.assist = assist
        self.known = known
        self.log = log
        self.summary = summary
        self.decode = decode
        self.meta = dict(meta or {})
        self.ask_goal = ask_goal
        flags = [f for f, on in (("log", log), ("summary", summary), ("known", known), ("assist", assist)) if on]
        self.name = f"jev:{personality}" + "".join(f"+{f}" for f in flags)
        self.last_answer = None

    def start(self, *a, **kw):
        super().start(*a, **kw)
        self.decisions: list[tuple[float, str]] = []  # (time, label) of this seat's non-pass choices
        self.trace: list[dict] = []

    def decide(self, view: View) -> Action:
        menu = action_menu(view)
        goal = self.counter.goal_probabilities() if self.assist else None
        state = add_context(view, self.counter.known() if self.known else None, goal, self.log, self.summary,
                            self.recent_decisions(view) if (self.log or self.summary) else None)
        questions = {"action": action_question(menu, self.personality, view.mechanism)}
        if self.ask_goal:
            questions["goal"] = goal_question(view.mechanism)
        n = len(self.trace)
        meta = {**self.meta, "seat": view.me, "t": round(view.t, 3), "decision": n, "mechanism": view.mechanism,
                "persona": self.personality, "decode": self.decode}
        answers = self.client.ask(state, questions, meta=meta)
        answer = answers["action"]
        self.last_answer = answer
        probs = answer.get("probabilities") or {answer["choice"]: 1.0}
        if self.decode == "argmax":
            label = answer["choice"]
        elif self.decode == "sample":
            labels = [k for k in probs if k in menu]
            label = self.rng.choices(labels, weights=[probs[k] for k in labels])[0] if labels else "pass"
        else:
            label = hierarchical_choice(probs)
        action = menu.get(label, (PASS, ""))[0]
        kinds: dict[str, float] = {}
        for k, p in probs.items():
            kinds[k.split("_")[0]] = kinds.get(k.split("_")[0], 0.0) + p
        rec = {"t": round(view.t, 3), "decision": n, "label": action.label(), "p_label": round(probs.get(label, 0.0), 5),
               "menu": len(menu), "p_kind": {k: round(v, 4) for k, v in kinds.items()}}
        if "goal" in answers:
            g = answers["goal"].get("probabilities") or {answers["goal"]["choice"]: 1.0}
            rec["goal"] = {s: round(g.get(s, 0.0), 5) for s in SUITS}
        self.trace.append(rec)
        if action != PASS:
            self.decisions.append((view.t, action.label()))
        return action

    def recent_decisions(self, view: View, n: int = 6) -> list[str]:
        """This seat's last non-pass choices and what became of them before its next one."""
        out = []
        recent = self.decisions[-n:]
        for i, (t0, label) in enumerate(recent):
            t1 = recent[i + 1][0] if i + 1 < len(recent) else view.t
            traded = any(t0 <= tr.t < t1 and view.me in (tr.buyer, tr.seller) for tr in view.trades)
            posted = any(t0 <= o.t < t1 and o.player == view.me for o in view.orders)
            if label.startswith(("buy", "sell")):
                result = "filled" if traded else "not filled"
            elif label.startswith("cancel"):
                result = "sent"
            else:
                result = "posted, then filled" if traded else "posted" if posted else "not accepted"
            out.append(f"t={round(t0)}s {label}: {result}")
        return out
