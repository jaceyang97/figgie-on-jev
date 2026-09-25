"""A Figgie agent whose every decision is a Jev Choice query."""

from __future__ import annotations

from ..cards import SUITS
from ..engine import View
from ..market import PASS, Action
from ..personalities import PERSONALITIES
from .base import Agent

RULES = (
    "Figgie: 4 players, 40 cards in 4 suits. One suit has 12 cards, one has 8 and two have 10. The goal suit is "
    "the suit of the same colour as the 12-card suit (spades/clubs are black, hearts/diamonds are red), so it has "
    "8 or 10 cards. At the end each goal-suit card pays 10 chips and whoever holds the most goal-suit cards wins "
    "the rest of the 200-chip pot (120 chips if the goal suit has 8 cards, 100 if it has 10). Cards of other suits "
    "are worth nothing. Seeing many cards of a suit means it is likely the 12-card suit, which makes the "
    "same-colour suit likely the goal suit. Every trade clears all bids and asks."
)

PRICE_LADDER = (2, 4, 6, 8, 11, 14, 18, 24)


def describe_state(view: View, known: dict[str, int], goal_probs: dict[str, float] | None = None, n_trades: int = 12) -> dict:
    """The JSON state sent to Jev. Only public information plus this seat's own hand."""
    book = {}
    for s in SUITS:
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
        "cards_known_to_exist": dict(known),
    }
    if goal_probs is not None:
        state["goal_suit_probability_from_card_counting"] = {s: round(p, 3) for s, p in goal_probs.items()}
    return state


def action_menu(view: View) -> dict[str, tuple[Action, str]]:
    """Every legal action as label -> (action, description). Jev picks one label."""
    menu = {"pass": (PASS, "Do nothing this turn.")}
    for s in SUITS:
        bid, ask = view.bids[s], view.asks[s]
        if ask is not None and ask.player != view.me and ask.price <= view.chips:
            a = Action("buy", s)
            menu[a.label()] = (a, f"Buy one {s} card now at the current ask of {ask.price} chips.")
        if bid is not None and bid.player != view.me and view.hand[s] > 0:
            a = Action("sell", s)
            menu[a.label()] = (a, f"Sell one of your {s} cards now at the current bid of {bid.price} chips.")
        for p in PRICE_LADDER:
            if (bid is None or p > bid.price) and (ask is None or p < ask.price) and p <= view.chips:
                a = Action("bid", s, p)
                menu[a.label()] = (a, f"Post a bid to buy one {s} card for {p} chips.")
            if view.hand[s] > 0 and (ask is None or p < ask.price) and (bid is None or p > bid.price):
                a = Action("ask", s, p)
                menu[a.label()] = (a, f"Post an offer to sell one of your {s} cards for {p} chips.")
    return menu


def action_question(menu: dict, personality: str) -> dict:
    persona = PERSONALITIES[personality]
    instructions = RULES + ("\n\n" + persona if persona else "") + "\n\nYou are the player in the state. Which action do you take now?"
    return {"type": "choice", "instructions": instructions, "criteria": {k: d for k, (_, d) in menu.items()}}


def goal_question() -> dict:
    return {
        "type": "choice",
        "instructions": RULES + "\n\nFrom this player's point of view, which suit is the goal suit?",
        "criteria": {s: f"The goal suit is {s}." for s in SUITS},
    }


class JevAgent(Agent):
    """Asks Jev which action to take, optionally with the classical posterior as a hint.

    `assist=True` gives Jev the card-counting goal probabilities (a hybrid:
    code does the maths, Jev decides). `latency` defaults to Jev's measured
    round-trip time when `use_measured_latency` is set.
    """

    def __init__(self, client, personality: str = "neutral", assist: bool = False,
                 greedy: bool = True, use_measured_latency: bool = False, **kw):
        kw.setdefault("wake_rate", 0.5)
        kw.setdefault("latency", 0.3)
        super().__init__(**kw)
        if personality not in PERSONALITIES:
            raise ValueError(f"unknown personality {personality!r}; choose from {sorted(PERSONALITIES)}")
        self.client = client
        self.personality = personality
        self.assist = assist
        self.greedy = greedy
        self.use_measured_latency = use_measured_latency
        self.name = f"jev:{personality}" + ("+assist" if assist else "")
        self.last_answer = None

    def latency(self) -> float:
        return self.client.last_latency if self.use_measured_latency else self._latency

    def decide(self, view: View) -> Action:
        menu = action_menu(view)
        goal = self.counter.goal_probabilities() if self.assist else None
        state = describe_state(view, self.counter.known(), goal)
        answer = self.client.ask(state, {"action": action_question(menu, self.personality)})["action"]
        self.last_answer = answer
        if self.greedy:
            label = answer["choice"]
        else:
            probs = answer["probabilities"]
            label = self.rng.choices(list(probs), weights=list(probs.values()))[0]
        return menu.get(label, (PASS, ""))[0]
