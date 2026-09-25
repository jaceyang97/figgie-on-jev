"""Persona texts for Jev agents: action rules only.

The core four translate Ozerov, DiSilvio and Luo (2021), section 2.3: each
defines how to value a card and all share the paper's order rule (Algorithm
2). The extensions translate published models; their sources are in
figgie.agents.classical.EXTENSION_SOURCES. Each persona has a rule-coded twin
of the same name in figgie.agents.classical. "neutral" has no persona and is
the control.

Add your own by adding an entry; the key is what you pass on the command line
(e.g. `jev:chartist`).
"""

ORDER_RULE = (
    "Order rule: pick one suit you have a value for, then pick buy or sell with equal chance. To buy: pick a "
    "whole-chip price at random between 0 and your buy value; if the standing ask is at or below that price, buy at "
    "the ask, otherwise bid that price. To sell (only if you hold a card of the suit): pick a price at random between "
    "your sell value and twice your sell value; if the standing bid is at or above that price, sell at the bid, "
    "otherwise offer at that price. If you have no value for any suit, pass."
)

FUNDAMENTAL_VALUE = (
    "Value from card counting. For each suit, track how many cards each player is known to hold: start with your "
    "dealt hand for you and zero for everyone else. On every trade the buyer gains one card; the seller loses one if "
    "they were known to hold one, otherwise their count becomes zero. The sum of these counts is the number of "
    "distinct cards of that suit seen. From the four sums, compute the probability of each of the 12 possible decks: "
    "all decks are equally likely before any evidence, and each is weighted by the number of ways the seen cards "
    "could be drawn from it. Your buy value for a suit when you hold n cards of it is the sum, over the decks in "
    "which that suit is the goal suit, of P(deck) x (10 + a x 1.2^n) if n is below x, or P(deck) x 10 if n is at "
    "least x. Here x is 5 if the goal suit has 8 cards and 6 if it has 10, the prize P is 120 if the goal suit has 8 "
    "cards and 100 if it has 10, and a = P x (1 - 1.2) / (1 - 1.2^x). Your sell value is the buy value computed "
    "with n - 1 cards."
)

PERSONALITIES = {
    "neutral": "",
    # The Figgie paper's four strategies.
    "fundamentalist": "Strategy: fundamentalist. " + FUNDAMENTAL_VALUE + " " + ORDER_RULE,
    "bottom_feeder": (
        "Strategy: bottom-feeder. Value only from other players' orders. An order is a bid or offer a player posted, "
        "or a trade they initiated (taking an ask is a buy order at that price, hitting a bid is a sell order). For "
        "each other player with at least 4 buy orders and 4 sell orders in a suit, average the prices of their last "
        "4 buy orders and of their last 4 sell orders, and take the midpoint of the two averages. Your buy value and "
        "sell value for the suit are both the average of these midpoints. If no player qualifies, you have no value "
        "for the suit. " + ORDER_RULE
    ),
    "chartist": (
        "Strategy: chartist. Value only from the suit's trade prices. For a suit with at least 5 trades, let p0 be "
        "the last trade price, p1 the one before it, and p4 the price 4 trades before the last. Your buy value and "
        "sell value are both p0 x p1 / p4. With fewer than 5 trades you have no value for the suit. " + ORDER_RULE
    ),
    "noise": (
        "Strategy: noise trader. For a suit with a standing bid, your value is the bid price multiplied by e^Z, where "
        "Z is a fresh random draw from a standard normal distribution; buy value and sell value are both this. "
        "Without a standing bid you have no value for the suit. " + ORDER_RULE
    ),
    # Extensions from the wider literature.
    "market_maker": (
        "Strategy: market maker. Never buy at a standing ask or sell at a standing bid. Pick a suit with a reference "
        "price: its last trade price, or if it has not traded, the midpoint of its standing bid and ask. Let q be the "
        "cards of that suit you hold now minus the cards of it you were dealt. Post either a bid at reference - 2 - q "
        "or an offer at reference + 2 - q, whichever is allowed. If no suit has a reference price, pass."
    ),
    "herder": (
        "Strategy: herder. Look only at the most recent trade in the game. If a buyer initiated it by taking the ask, "
        "buy the same suit: take the standing ask if it is at or below that trade's price, otherwise bid that price. "
        "If a seller initiated it by hitting the bid, sell the same suit: take the standing bid if it is at or above "
        "that price, otherwise offer at that price. If there have been no trades, pass."
    ),
    "contrarian": (
        "Strategy: contrarian. For each suit with at least 4 trades, compare the last trade price with the price 3 "
        "trades before it. If it is higher, sell the suit: take the standing bid if it is at or above the last price, "
        "otherwise offer at the last price. If it is lower, buy: take the standing ask if it is at or below the last "
        "price, otherwise bid the last price. Skip suits where it is unchanged; if no suit qualifies, pass."
    ),
    "sniper": (
        "Strategy: sniper. " + FUNDAMENTAL_VALUE + " Never post bids or offers. If a standing ask is below your buy "
        "value or a standing bid is above your sell value, take the one with the largest difference. Otherwise pass."
    ),
    "zero_intelligence": (
        "Strategy: zero-intelligence with a budget constraint. " + FUNDAMENTAL_VALUE + " Pick a suit at random and "
        "buy or sell with equal chance. To buy: pick a random whole-chip price from 1 to your buy value; buy at the "
        "standing ask if it is at or below that price, otherwise bid it. To sell (only if you hold a card of the "
        "suit): pick a random price from your sell value to 30; sell at the standing bid if it is at or above that "
        "price, otherwise offer at it. Never buy above your buy value or sell below your sell value."
    ),
    "disposition": (
        "Strategy: disposition effect. " + FUNDAMENTAL_VALUE + " " + ORDER_RULE + " One constraint: if you have "
        "bought cards of a suit in this game, never sell that suit below the average price you paid for it; when the "
        "order rule gives a lower offer, offer at that average instead, and do not sell at a standing bid below it."
    ),
}
