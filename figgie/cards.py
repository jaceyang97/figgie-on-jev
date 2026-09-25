"""Deck, suits and dealing for Figgie.

A Figgie deck has 40 cards in four suits: one suit has 12 cards, one has 8 and
two have 10. The goal suit is the suit of the same colour as the 12-card suit,
so it always has 8 or 10 cards. There are 12 possible decks.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

SUITS = ("spades", "clubs", "hearts", "diamonds")
SAME_COLOUR = {"spades": "clubs", "clubs": "spades", "hearts": "diamonds", "diamonds": "hearts"}

N_PLAYERS = 4
HAND_SIZE = 10
ANTE = 50
STARTING_CHIPS = 350
POT = ANTE * N_PLAYERS
GOAL_CARD_PAYOUT = 10


@dataclass(frozen=True)
class Deck:
    """One of the 12 possible deck compositions."""

    common: str
    eight: str

    @property
    def goal(self) -> str:
        return SAME_COLOUR[self.common]

    def count(self, suit: str) -> int:
        if suit == self.common:
            return 12
        if suit == self.eight:
            return 8
        return 10

    @property
    def counts(self) -> dict[str, int]:
        return {s: self.count(s) for s in SUITS}

    @property
    def bonus(self) -> int:
        """Chips left in the pot for the majority holder after goal cards are paid."""
        return POT - GOAL_CARD_PAYOUT * self.count(self.goal)


ALL_DECKS = tuple(Deck(common, eight) for common in SUITS for eight in SUITS if eight != common)


def deal(rng: random.Random) -> tuple[Deck, list[dict[str, int]]]:
    """Pick a deck uniformly at random and deal 10 cards to each player."""
    deck = rng.choice(ALL_DECKS)
    cards = [s for s in SUITS for _ in range(deck.count(s))]
    rng.shuffle(cards)
    hands = []
    for p in range(N_PLAYERS):
        hand = {s: 0 for s in SUITS}
        for c in cards[p * HAND_SIZE : (p + 1) * HAND_SIZE]:
            hand[c] += 1
        hands.append(hand)
    return deck, hands
