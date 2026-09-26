from __future__ import annotations

import random

from ..engine import View
from ..market import PASS, Action, Trade
from ..posterior import CardCounter

# Speed settings. EQUAL is the experiment's setting for every trader, Jev and rule-based alike, so profit
# differences come from decisions and not from speed. PAPER is our reading of the Figgie paper's default:
# it gives no consideration rate, and its figure 2 shows about 2,500 time units for a 10,000-event game of
# 4 agents, which fits a rate of about 1 per time unit; its latency is 0 except in its latency test.
EQUAL_SPEED = (0.5, 0.3)  # (wake-ups per second, seconds from decision to arrival)
PAPER_SPEED = (1.0, 0.0)
SPEEDS = {"equal": EQUAL_SPEED, "paper": PAPER_SPEED}


class Agent:
    """Base agent. Subclasses override `decide`."""

    name = "agent"
    kind = "agent"  # type label other agents may see (the bottom-feeder picks its prey by it)

    def __init__(self, wake_rate: float = EQUAL_SPEED[0], latency: float = EQUAL_SPEED[1], name: str | None = None):
        self.wake_rate = wake_rate  # expected wake-ups per second
        self._latency = latency  # seconds between deciding and the order reaching the exchange
        if name:
            self.name = name

    def start(self, me: int, hand: dict[str, int], n_players: int, rng: random.Random, labels=None) -> None:
        self.me = me
        self.rng = rng
        self.labels = list(labels) if labels is not None else [None] * n_players
        self.counter = CardCounter(me, hand, n_players)

    def on_trade(self, trade: Trade) -> None:
        self.counter.on_trade(trade.suit, trade.buyer, trade.seller)

    def latency(self) -> float:
        return self._latency

    def cancel_now(self, view: View) -> list[int]:
        """Rule set A: ids of own resting orders to delete at once before deciding (the paper's lazy deletion)."""
        return []

    def decide(self, view: View) -> Action:
        return PASS
