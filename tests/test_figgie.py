import random

import pytest

from figgie.agents import BottomFeeder, Chartist, Fundamentalist, JevAgent, Noise, make_agent
from figgie.agents.jev_agent import action_menu, action_question, describe_state
from figgie.cards import ALL_DECKS, POT, SUITS, deal
from figgie.engine import View, play_game, settle
from figgie.jev import MockJev
from figgie.market import Action, Market
from figgie.personalities import ALL_PERSONAS, DESCRIPTIONS, PERSONALITIES
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
    for p in ALL_PERSONAS:
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
    q = action_question(menu, "chartist")
    assert q["type"] == "choice" and set(q["criteria"]) == set(menu)
    assert PERSONALITIES["chartist"] in q["instructions"]
    state = describe_state(view, hands[0], {s: 0.25 for s in SUITS})
    assert state["my_hand"] == hands[0]
    assert "goal_suit_probability_derived" in state and "cards_known_to_exist_derived" in state
    assert "cards_known_to_exist_derived" not in describe_state(view)


def test_history_summarises_public_log():
    from figgie.agents.jev_agent import describe_history
    from figgie.market import Order, Trade

    hand = {"spades": 3, "clubs": 2, "hearts": 4, "diamonds": 1}
    trades = (Trade(5, "spades", 7, buyer=0, seller=2), Trade(9, "spades", 9, buyer=1, seller=0),
              Trade(12, "clubs", 4, buyer=0, seller=3))
    orders = (Order(3, 2, "ask", "spades", 7), Order(8, 1, "bid", "spades", 9))
    view = View(t=20, t_end=240, me=0, hand=hand, chips=300,
                bids={s: None for s in SUITS}, asks={s: None for s in SUITS}, trades=trades, orders=orders)
    h = describe_history(view)
    assert h["suits"]["spades"]["trades"] == 2 and h["suits"]["spades"]["avg_price"] == 8.0
    assert h["suits"]["spades"]["highest_bid_ever"] == 9
    me = h["players"]["me"]
    assert me["net_cards_bought"] == {"clubs": 1}
    assert me["chips_from_trading"] == -7 + 9 - 4
    assert me["starting_hand"] == {"spades": 3, "clubs": 1, "hearts": 4, "diamonds": 1}
    assert h["players"]["P2"]["net_cards_bought"] == {"spades": -1}
    from figgie.agents.jev_agent import add_context, describe_log
    log = describe_log(view)
    assert log[0] == "t=3.0s P2 ask spades 7" and log[1] == "t=5.0s trade spades 7: you bought from P2"
    assert len(log) == 5
    st = add_context(view, log=True, summary=True, my_decisions=["t=12s buy_clubs: filled"])
    assert "recent_trades_oldest_first" not in st and st["all_events_oldest_first"] == log
    assert st["my_recent_decisions"] == ["t=12s buy_clubs: filled"] and "game_summary" in st


def test_jev_history_agent_runs():
    client = MockJev(seed=3)
    agents = [make_agent("jev:neutral+log+summary", client), Fundamentalist(), BottomFeeder(), Noise()]
    res = play_game(agents, random.Random(4), duration=40)
    assert sum(res.pnl) == pytest.approx(0.0)
    assert agents[0].name == "jev:neutral+log+summary"


def test_make_agent_specs():
    client = MockJev()
    a = make_agent("jev:bottom_feeder+assist", client)
    assert a.personality == "bottom_feeder" and a.assist and not a.log
    b = make_agent("jev:chartist+summary+known+assist", client)
    assert b.summary and b.known and b.assist and not b.log
    with pytest.raises(ValueError):
        make_agent("jev:neutral+history", client)
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


def test_openrouter_without_key_relies_on_gateway(monkeypatch, tmp_path):
    from figgie.jev import JevClient, JevError

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    missing = str(tmp_path / "none")
    with pytest.raises(JevError):
        JevClient(provider="typesafe", env_file=missing)
    c = JevClient(provider="openrouter", env_file=missing)
    assert c.provider == "openrouter" and c.api_key == ""


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


def test_every_rule_based_agent_plays_a_zero_sum_game():
    from figgie.agents import CLASSICAL
    names = list(CLASSICAL)
    for i in range(0, len(names), 3):
        lineup = (names[i:i + 3] + ["fundamentalist", "noise", "bottom_feeder"])[:4]
        agents = [make_agent(n) for n in lineup]
        res = play_game(agents, random.Random(i), duration=60)
        assert sum(res.pnl) == pytest.approx(0.0)


def test_card_counter_follows_algorithm_3():
    from figgie.posterior import CardCounter
    c = CardCounter(0, {"spades": 2, "clubs": 0, "hearts": 0, "diamonds": 0}, 4)
    c.on_trade("spades", buyer=1, seller=2)  # P2 was not known to hold one: a new card is revealed
    assert c.known()["spades"] == 3
    c.on_trade("spades", buyer=3, seller=1)  # P1 is known to hold it: nothing new
    assert c.known()["spades"] == 3
    c.on_trade("spades", buyer=1, seller=0)  # I sell one of mine: nothing new
    assert c.known()["spades"] == 3 and c.held["spades"] == [1, 1, 0, 1]


def test_paper_majority_shares_sum_to_the_prize():
    from figgie.cards import ALL_DECKS
    from figgie.posterior import R_MAJORITY, buy_value, majority_needed
    deck = ALL_DECKS[0]
    post = {d: (1.0 if d == deck else 0.0) for d in ALL_DECKS}
    x = majority_needed(deck)
    shares = sum(buy_value(deck.goal, n, post) - 10 for n in range(x))
    assert shares == pytest.approx(deck.bonus)
    assert buy_value(deck.goal, x, post) == pytest.approx(10)
    assert R_MAJORITY > 1


def test_trades_record_the_aggressor():
    from figgie.market import Market
    m = Market(hands=[{s: 2 for s in SUITS} for _ in range(4)], chips=[300] * 4)
    m.apply(1.0, 1, Action("ask", "spades", 9))
    t = m.apply(2.0, 2, Action("buy", "spades"))
    assert t.buyer == 2 and t.seller == 1 and t.aggressor == 2


def test_hierarchical_choice_sums_before_picking():
    from figgie.agents.jev_agent import hierarchical_choice
    probs = {"pass": 0.3, "bid_spades_5": 0.1, "bid_spades_6": 0.25, "bid_clubs_4": 0.2, "buy_hearts": 0.15}
    assert max(probs, key=probs.get) == "pass"
    assert hierarchical_choice(probs) == "bid_spades_6"
    assert hierarchical_choice({"pass": 0.6, "bid_spades_5": 0.4}) == "pass"


def test_duplicate_agents_keep_separate_rows():
    from types import SimpleNamespace

    from figgie.experiments.tournament import run

    args = SimpleNamespace(lineup="fundamentalist,bottom_feeder,fundamentalist,noise", games=8, duration=30.0,
                           jev_latency=None, seed=0, max_calls=None, log=None, out=None, workers=1,
                           backend="mock", provider=None)
    agents = run(args)["agents"]
    assert set(agents) == {"fundamentalist#0", "fundamentalist#2", "bottom_feeder", "noise"}
    # P&L is zero-sum, so the four per-seat means must add to zero.
    assert sum(a["mean_pnl"] for a in agents.values()) == pytest.approx(0, abs=0.05)


def test_bootstrap_resamples_whole_groups():
    from figgie.stats import bootstrap_ci

    xs = [0.0] * 5 + [1.0] * 5
    lo, hi = bootstrap_ci(xs, groups=[0] * 5 + [1] * 5)
    assert (lo, hi) == (0.0, 1.0)  # two clusters: every resample is all-0, all-1 or half and half
    lo2, hi2 = bootstrap_ci(xs)
    assert hi2 - lo2 < 1.0


def test_every_strategy_has_a_behaviour_only_description():
    assert set(DESCRIPTIONS) == set(PERSONALITIES) - {"neutral"}
    menu = {"pass": (None, "Do nothing this turn.")}
    q = action_question(menu, "fundamentalist-desc")
    assert DESCRIPTIONS["fundamentalist"] in q["instructions"]
    assert PERSONALITIES["fundamentalist"] not in q["instructions"]
    for text in DESCRIPTIONS.values():
        assert "goal suit" not in text  # describe behaviour only, never which suit wins
