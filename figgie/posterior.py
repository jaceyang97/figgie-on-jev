"""Exact Bayesian inference over the 12 possible decks, and card values.

This is the "classical" model the paper's fundamentalist uses: count every card
known to exist, then weight each deck by the hypergeometric likelihood of those
counts. With K known cards the likelihood of deck d is
prod_s C(n_s(d), k_s) / C(40, K); the denominator is the same for every deck.
"""

from __future__ import annotations

from math import comb

from .cards import ALL_DECKS, GOAL_CARD_PAYOUT, SUITS, Deck


def deck_posterior(known: dict[str, int]) -> dict[Deck, float]:
    weights = {}
    for deck in ALL_DECKS:
        w = 1.0
        for s in SUITS:
            w *= comb(deck.count(s), known.get(s, 0))
        weights[deck] = w
    total = sum(weights.values())
    return {d: w / total for d, w in weights.items()}


def goal_probabilities(known: dict[str, int]) -> dict[str, float]:
    post = deck_posterior(known)
    probs = {s: 0.0 for s in SUITS}
    for deck, p in post.items():
        probs[deck.goal] += p
    return probs


class CardCounter:
    """Tracks how many cards of each suit are known to exist, from one seat's view.

    Known cards are the seat's initial hand plus, for every other player, the most
    cards of a suit that player has ever been net short of (they must have held
    them to sell them). Cards the seat buys are counted through the seller.
    """

    def __init__(self, me: int, hand: dict[str, int], n_players: int):
        self.me = me
        self.initial = dict(hand)
        self.net_sold = [{s: 0 for s in SUITS} for _ in range(n_players)]
        self.max_net_sold = [{s: 0 for s in SUITS} for _ in range(n_players)]

    def on_trade(self, suit: str, buyer: int, seller: int) -> None:
        self.net_sold[seller][suit] += 1
        self.net_sold[buyer][suit] -= 1
        m = self.max_net_sold[seller]
        m[suit] = max(m[suit], self.net_sold[seller][suit])

    def known(self) -> dict[str, int]:
        known = dict(self.initial)
        for p, sold in enumerate(self.max_net_sold):
            if p != self.me:
                for s in SUITS:
                    known[s] += sold[s]
        return known

    def goal_probabilities(self) -> dict[str, float]:
        return goal_probabilities(self.known())

    def deck_posterior(self) -> dict[Deck, float]:
        return deck_posterior(self.known())


def card_values(known: dict[str, int], hand: dict[str, int]) -> dict[str, tuple[float, float]]:
    """(buy value, sell value) of one card of each suit, in chips.

    A goal card pays 10 chips. It also moves you towards the majority bonus,
    which needs n//2 + 1 cards of an n-card goal suit. The bonus is spread evenly
    over the cards needed to reach that majority, so the next card is worth
    bonus/majority while you are short of it and nothing once you have it.
    This is a simplified version of the paper's geometric weighting.
    """
    post = deck_posterior(known)
    values = {}
    for s in SUITS:
        buy = sell = 0.0
        for deck, p in post.items():
            if deck.goal != s:
                continue
            n = deck.count(s)
            majority = n // 2 + 1
            share = deck.bonus / majority
            h = hand.get(s, 0)
            buy += p * (GOAL_CARD_PAYOUT + (share if h < majority else 0.0))
            sell += p * (GOAL_CARD_PAYOUT + (share if 0 < h <= majority else 0.0))
        values[s] = (buy, sell)
    return values
