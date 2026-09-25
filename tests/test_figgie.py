import random

import pytest

from figgie.agents import BottomFeeder, Chartist, Fundamentalist, JevAgent, Noise, make_agent
from figgie.agents.jev_agent import action_menu, action_question, describe_state
from figgie.cards import ALL_DECKS, POT, SUITS, deal
from figgie.engine import View, play_game, settle
from figgie.jev import MockJev
from figgie.market import Action, Market
from figgie.personalities import PERSONALITIES
from figgie.posterior import CardCounter, goal_probabilities


def test_twelve_decks_with_40_cards_and_8_or_10_card_goal():
    assert len(ALL_DECKS) == 12
    for d in ALL_DECKS:
        assert sum(d.counts.values()) == 40
        assert d.count(d.goal) in (8, 10)
        assert 10 * d.count(d.goal) + d.bonus == POT


def test_deal_gives_ten_cards_each():
    deck, hands = deal(random.Random(1))
    assert all(sum(h.values()) == 10 for h in hands)
    for s in SUITS:
        assert sum(h[s] for h in hands) == deck.count(s)


def test_posterior_uniform_with_no_information():
    probs = goal_probabilities({s: 0 for s in SUITS})
    assert all(abs(p - 0.25) < 1e-12 for p in probs.values())


def test_posterior_rules_out_impossible_decks():
    # 11 spades seen: spades must be the 12-card suit, so clubs is the goal.
    probs = goal_probabilities({"spades": 11, "clubs": 0, "hearts": 0, "diamonds": 0})
    assert probs["clubs"] == pytest.approx(1.0)


def test_counter_counts_cards_sold_by_others():
    c = CardCounter(me=0, hand={"spades": 2, "clubs": 0, "hearts": 0, "diamonds": 0}, n_players=4)
    c.on_trade("hearts", buyer=0, seller=1)
    c.on_trade("hearts", buyer=2, seller=1)
    c.on_trade("hearts", buyer=1, seller=2)  # P2 resells the card it bought: no new card
    assert c.known()["hearts"] == 2


def test_market_trade_clears_all_quotes():
    m = Market(hands=[{s: 2 for s in SUITS} for _ in range(4)], chips=[300] * 4)
    assert m.apply(0, 0, Action("bid", "spades", 5))
    assert m.apply(0, 1, Action("ask", "hearts", 9))
    assert not m.apply(0, 2, Action("bid", "spades", 4)), "bid must improve"
    trade = m.apply(1, 2, Action("sell", "spades"))
    assert trade.price == 5 and trade.buyer == 0 and trade.seller == 2
    assert all(m.bids[s] is None and m.asks[s] is None for s in SUITS)
    assert m.hands[0]["spades"] == 3 and m.chips[0] == 295


def test_crossing_bid_trades_at_ask():
    m = Market(hands=[{s: 2 for s in SUITS} for _ in range(4)], chips=[300] * 4)
    m.apply(0, 1, Action("ask", "clubs", 7))
    trade = m.apply(0, 0, Action("bid", "clubs", 10))
    assert trade.price == 7


def test_cannot_sell_what_you_do_not_hold():
    m = Market(hands=[{s: 0 for s in SUITS} for _ in range(4)], chips=[300] * 4)
    m.apply(0, 1, Action("bid", "clubs", 7))
    assert m.apply(0, 0, Action("sell", "clubs")) is None
    assert m.apply(0, 0, Action("ask", "clubs", 9)) is None


def test_settle_is_zero_sum():
    deck, hands = deal(random.Random(3))
    pnl = settle(deck, hands, [300] * 4)
    assert sum(pnl) == pytest.approx(0.0)


def test_classical_game_conserves_chips_and_cards():
    agents = [Fundamentalist(), BottomFeeder(), Chartist(), Noise()]
    res = play_game(agents, random.Random(0), duration=60)
    assert sum(res.pnl) == pytest.approx(0.0)
    for s in SUITS:
        assert sum(h[s] for h in res.final_hands) == res.deck.count(s)
    assert res.trades


def test_fundamentalist_beats_noise_on_average():
    rng = random.Random(0)
    totals = [0.0, 0.0]
    for _ in range(20):
        res = play_game([Fundamentalist(), Noise(), Noise(), Noise()], rng, duration=120)
        totals[0] += res.pnl[0]
        totals[1] += res.pnl[1]
    assert totals[0] > totals[1]


def test_jev_agent_runs_with_mock_and_every_personality():
    for p in PERSONALITIES:
        client = MockJev(seed=1)
        agents = [JevAgent(client, personality=p, wake_rate=0.2), Fundamentalist(), BottomFeeder(), Noise()]
        res = play_game(agents, random.Random(2), duration=30)
        assert sum(res.pnl) == pytest.approx(0.0)
        assert client.calls == res.decisions[0]


def test_jev_payload_shape_matches_api():
    _, hands = deal(random.Random(0))
    view = View(t=10, t_end=240, me=0, hand=hands[0], chips=300,
                bids={s: None for s in SUITS}, asks={s: None for s in SUITS}, trades=(), orders=())
    menu = action_menu(view)
    assert "pass" in menu and len(menu) <= 255
    q = action_question(menu, "hoarder")
    assert q["type"] == "choice" and set(q["criteria"]) == set(menu)
    assert PERSONALITIES["hoarder"] in q["instructions"]
    state = describe_state(view, hands[0], {s: 0.25 for s in SUITS})
    assert state["my_hand"] == hands[0]
    assert "goal_suit_probability_from_card_counting" in state


def test_make_agent_specs():
    client = MockJev()
    a = make_agent("jev:value+assist", client)
    assert a.personality == "value" and a.assist
    assert make_agent("fundamentalist").name == "fundamentalist"
    with pytest.raises(ValueError):
        make_agent("jev:nonexistent", client)


def test_jev_route_prefers_openrouter(monkeypatch, tmp_path):
    from figgie.jev import JevClient, JevError

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("JEV_MODEL", raising=False)
    missing = str(tmp_path / "none")
    with pytest.raises(JevError):
        JevClient(env_file=missing)
    monkeypatch.setenv("TYPESAFE_API_KEY", "t")
    c = JevClient(env_file=missing)
    assert c.provider == "typesafe" and c.model == "jev-latest"
    monkeypatch.setenv("OPENROUTER_API_KEY", "o")
    c = JevClient(env_file=missing)
    assert c.provider == "openrouter" and c.model == "typesafe/jev-1.13"
    assert c.url == "https://openrouter.ai/api/alpha/decisions"
    assert JevClient(provider="typesafe", env_file=missing).provider == "typesafe"


def test_env_file_does_not_override_process_env(monkeypatch, tmp_path):
    from figgie.jev import JevClient

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("TYPESAFE_API_KEY", "from-env")
    env = tmp_path / ".env.local"
    env.write_text("# keys\nOPENROUTER_API_KEY=from-file\nTYPESAFE_API_KEY=file-value\n")
    c = JevClient(env_file=str(env))
    assert c.provider == "openrouter" and c.api_key == "from-file"
    import os
    assert os.environ["TYPESAFE_API_KEY"] == "from-env"
    monkeypatch.delenv("OPENROUTER_API_KEY")
