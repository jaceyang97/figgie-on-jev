from __future__ import annotations

import random

from ..engine import View
from ..market import PASS, Action, Trade
from ..posterior import CardCounter


class Agent:
    """Base agent. Subclasses override `decide`."""

    name = "agent"

    def __init__(self, wake_rate: float = 1.0, latency: float = 0.05, name: str | None = None):
        self.wake_rate = wake_rate  # expected wake-ups per second
        self._latency = latency  # seconds between deciding and the order reaching the exchange
        if name:
            self.name = name

    def start(self, me: int, hand: dict[str, int], n_players: int, rng: random.Random) -> None:
        self.me = me
        self.rng = rng
        self.counter = CardCounter(me, hand, n_players)

    def on_trade(self, trade: Trade) -> None:
        self.counter.on_trade(trade.suit, trade.buyer, trade.seller)

    def latency(self) -> float:
        return self._latency

    def decide(self, view: View) -> Action:
        return PASS
