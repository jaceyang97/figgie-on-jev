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
    """Card counting from one seat's view, as in the paper's Algorithm 3.

    For each suit, L[p] is how many cards of it player p is known to hold. It
    starts as this seat's own hand. On a trade the buyer gains one; the seller
    loses one if it was known to hold one, otherwise it drops to 0 (the card was
    not known before, so a new card has been revealed). The number of distinct
    cards of a suit seen so far is the sum of L over players.
    """

    def __init__(self, me: int, hand: dict[str, int], n_players: int):
        self.me = me
        self.initial = dict(hand)
        self.held = {s: [0] * n_players for s in SUITS}
        for s in SUITS:
            self.held[s][me] = hand[s]

    def on_trade(self, suit: str, buyer: int, seller: int) -> None:
        held = self.held[suit]
        if held[seller] < 1:
            held[buyer] += 1
            held[seller] = 0
        else:
            held[buyer] += 1
            held[seller] -= 1

    def known(self) -> dict[str, int]:
        return {s: sum(self.held[s]) for s in SUITS}

    def goal_probabilities(self) -> dict[str, float]:
        return goal_probabilities(self.known())

    def deck_posterior(self) -> dict[Deck, float]:
        return deck_posterior(self.known())


R_MAJORITY = 1.2  # the paper's r > 1; it gives no value, so this is our choice


def majority_needed(deck: Deck) -> int:
    """Goal cards that guarantee the majority: 5 of 8, 6 of 10 (the paper's Table 1)."""
    return deck.count(deck.goal) // 2 + 1


def buy_value(suit: str, n: int, post: dict[Deck, float], r: float = R_MAJORITY) -> float:
    """The paper's e_b(j, n, m): expected value of buying one more card of `suit` holding n."""
    total = 0.0
    for deck, p in post.items():
        if deck.goal != suit:
            continue
        x, pay = majority_needed(deck), deck.bonus
        a = pay * (1 - r) / (1 - r ** x)
        total += p * (GOAL_CARD_PAYOUT + (a * r ** n if n < x else 0.0))
    return total


def card_values(known: dict[str, int], hand: dict[str, int], r: float = R_MAJORITY) -> dict[str, tuple[float, float]]:
    """(buy value, sell value) per suit, as the paper's fundamentalist computes them.

    Buy value e_b(n) = sum over decks of P(deck) * v, with v = 0 unless the suit is
    the goal suit, else 10 + a*r^n while n is below the majority threshold x and 0
    after (a = payout*(1-r)/(1-r^x), so the shares over n = 0..x-1 sum to the
    payout). Sell value e_s(n) = e_b(n-1).
    """
    post = deck_posterior(known)
    return {s: (buy_value(s, hand[s], post, r), buy_value(s, hand[s] - 1, post, r) if hand[s] > 0 else 0.0)
            for s in SUITS}
