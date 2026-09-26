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


# --- rule set A, stage 0-2 plumbing ---------------------------------------------------------------------------

from figgie.agents import MarketMakerAS, MarketMakerGM, parse_speed  # noqa: E402
from figgie.agents.jev_agent import PRICE_LADDER, goal_question  # noqa: E402
from figgie.experiments.stage1 import to_menu_label, view_from_json, view_to_json  # noqa: E402
from figgie.experiments.tournament import game_rng  # noqa: E402
from figgie.market import MAX_ORDERS_PER_SIDE, BookMarket, Order, Quote, make_market  # noqa: E402
from figgie.records import game_record, starting_goal_cards  # noqa: E402


def book(n=2):
    return BookMarket(hands=[{s: n for s in SUITS} for _ in range(4)], chips=[300] * 4)


def test_book_keeps_orders_and_trades_at_the_resting_price():
    m = book()
    assert isinstance(m.apply(0, 0, Action("bid", "spades", 5)), Order)
    assert isinstance(m.apply(1, 1, Action("bid", "spades", 4)), Order), "a lower bid still rests in A"
    assert [q.price for q in m.depth("spades")["bids"]] == [5, 4]
    trade = m.apply(2, 2, Action("ask", "spades", 3))
    assert trade.price == 5 and trade.buyer == 0 and trade.seller == 2 and trade.aggressor == 2
    assert m.bids["spades"] == Quote(4, 1), "the other orders stay after a trade"


def test_book_price_time_priority():
    m = book()
    m.apply(0, 0, Action("ask", "clubs", 6))
    m.apply(1, 1, Action("ask", "clubs", 6))
    assert m.apply(2, 2, Action("buy", "clubs")).seller == 0


def test_book_limit_self_trade_and_invalid_orders():
    m = book()
    for p in range(1, MAX_ORDERS_PER_SIDE + 2):
        m.apply(p, 0, Action("bid", "hearts", p))
    assert len(m.open_orders(0)) == MAX_ORDERS_PER_SIDE
    assert m.cancels[0].reason == "limit" and m.cancels[0].price == 1
    # self-trade prevention: my own resting bid is removed, not traded against
    m.apply(10, 0, Action("sell", "hearts"))
    assert all(c.player == 0 for c in m.cancels) and any(c.reason == "self_trade" for c in m.cancels)
    # a resting ask whose owner no longer holds the card is removed at match time
    m2 = book(n=1)
    m2.apply(0, 1, Action("ask", "diamonds", 3))
    m2.hands[1]["diamonds"] = 0
    assert m2.apply(1, 0, Action("buy", "diamonds")) is None
    assert m2.cancels[-1].reason == "invalid"


def test_book_cancel_removes_only_my_orders_in_that_suit():
    m = book()
    m.apply(0, 0, Action("bid", "spades", 3))
    m.apply(0, 0, Action("ask", "clubs", 9))
    m.apply(0, 1, Action("bid", "spades", 2))
    assert m.apply(1, 0, Action("cancel", "spades")) == 1
    assert [o.suit for o in m.open_orders(0)] == ["clubs"] and m.bids["spades"] == Quote(2, 1)
    assert m.apply(2, 0, Action("cancel", "hearts")) is None


def test_both_rule_sets_play_zero_sum_games():
    for mech in ("A", "B"):
        agents = [make_agent(s) for s in ("fundamentalist", "bottom_feeder", "market_maker_gm", "market_maker_as")]
        res = play_game(agents, random.Random(5), duration=120, mechanism=mech)
        assert res.mechanism == mech and sum(res.pnl) == pytest.approx(0.0)
        for s in SUITS:
            assert sum(h[s] for h in res.final_hands) == res.deck.count(s)
    assert isinstance(make_market("A", [{s: 1 for s in SUITS}] * 4, [1] * 4), BookMarket)


def test_event_mode_stops_after_max_events():
    res = play_game([make_agent("fundamentalist@1/0"), Noise(), Noise(), Noise()], random.Random(1),
                    mechanism="A", max_events=500)
    assert res.events == 500


def test_same_seed_same_deal_in_every_arm():
    a = play_game([make_agent(s) for s in ("noise", "fundamentalist", "bottom_feeder", "noise")], game_rng(0, 7),
                  duration=60, mechanism="A", tested_seat=0)
    b = play_game([make_agent(s) for s in ("market_maker_as", "fundamentalist", "bottom_feeder", "noise")],
                  game_rng(0, 7), duration=60, mechanism="A", tested_seat=0)
    assert a.deck == b.deck and a.initial_hands == b.initial_hands
    c = play_game([make_agent(s) for s in ("noise", "fundamentalist", "bottom_feeder", "noise")], game_rng(0, 7),
                  duration=60, mechanism="A", tested_seat=0)
    assert c.trades == a.trades and c.pnl == a.pnl, "a game is a function of its seed"


def test_bottom_feeder_never_preys_on_the_tested_seat():
    agents = [make_agent("fundamentalist"), make_agent("bottom_feeder"), make_agent("fundamentalist"), Noise()]
    res = play_game(agents, random.Random(3), duration=10, tested_seat=0)
    assert res.labels == ["tested", "bottom_feeder", "fundamentalist", "noise"]
    assert agents[1].prey() == [2]


def test_market_makers_quote_bid_below_ask_and_never_take():
    for cls in (MarketMakerGM, MarketMakerAS):
        agents = [cls(), Fundamentalist(), Noise(), Noise()]
        res = play_game(agents, random.Random(4), duration=120, mechanism="A")
        assert all(t.aggressor != 0 for t in res.trades), f"{cls.__name__} took an order"
        mm = agents[0]
        view = View(t=100, t_end=240, me=0, hand=res.final_hands[0], chips=300, bids={s: None for s in SUITS},
                    asks={s: None for s in SUITS}, trades=tuple(res.trades), orders=(), mechanism="A", progress=0.4)
        for bid, ask in mm._int_quotes(view).values():
            assert 0 <= bid < ask


def test_menus_fit_jev_and_use_the_ladder():
    _, hands = deal(random.Random(0))
    for mech in ("A", "B"):
        view = View(t=10, t_end=240, me=0, hand=hands[0], chips=300, bids={s: Quote(2, 1) for s in SUITS},
                    asks={s: Quote(30, 2) for s in SUITS}, trades=(), orders=(), mechanism=mech,
                    my_orders=tuple(Order(1, 0, "bid", s, 1, i) for i, s in enumerate(SUITS)))
        menu = action_menu(view)
        assert len(menu) <= 255
        prices = {a.price for a, _ in menu.values() if a.price is not None}
        assert prices <= set(PRICE_LADDER)
        q = action_question(menu, "neutral", mech)
        assert q["type"] == "choice" and set(q["criteria"]) == set(menu)
        assert set(goal_question(mech)["criteria"]) == set(SUITS)
    assert "cancel_spades" in action_menu(view.__class__(**{**view.__dict__, "mechanism": "A"}))
    assert len(PRICE_LADDER) == 30


def test_mock_log_carries_join_keys(tmp_path):
    log = tmp_path / "log.jsonl"
    client = MockJev(seed=1, log_path=str(log))
    agent = make_agent("jev:neutral+summary", client, meta={"run": "r", "condition": "A/neutral", "game": 3})
    res = play_game([agent, Fundamentalist(), BottomFeeder(), Noise()], random.Random(2), duration=20, mechanism="A")
    import json
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    assert len(lines) == client.calls == len(agent.trace) == res.decisions[0]
    m = lines[0]["meta"]
    assert (m["run"], m["condition"], m["game"], m["seat"], m["decision"], m["mechanism"]) == ("r", "A/neutral", 3, 0, 0, "A")
    assert {"goal", "card_counting", "label", "p_label"} <= set(agent.trace[0])


def test_game_record_has_what_the_figures_need():
    agents = [make_agent(s) for s in ("fundamentalist", "bottom_feeder", "noise", "noise")]
    res = play_game(agents, random.Random(9), duration=60, mechanism="A", tested_seat=0)
    rec = game_record(res, 0, 9, ["fundamentalist", "bottom_feeder", "noise", "noise"], "A/x", agents)
    for key in ("seed", "mechanism", "deck", "goal", "initial_hands", "final_hands", "pnl", "final_chips", "payout",
                "decisions", "passes", "rejected", "trades", "orders", "cancels", "labels", "speeds", "events"):
        assert key in rec
    assert starting_goal_cards(rec, 0) == res.initial_hands[0][res.deck.goal]
    assert len(rec["trades"][0]) == 6 and len(rec["orders"][0]) == 6


def test_view_json_round_trip_and_ladder_mapping():
    agents = [make_agent(s) for s in ("fundamentalist", "bottom_feeder", "noise", "noise")]
    seen = []

    class Spy(Fundamentalist):
        def decide(self, view):
            seen.append(view)
            return super().decide(view)

    agents[0] = Spy()
    play_game(agents, random.Random(1), duration=30, mechanism="A")
    v = seen[-1]
    assert view_from_json(view_to_json(v)) == v
    menu = {"pass": None, "bid_spades_22": None, "ask_spades_40": None}
    assert to_menu_label(Action("bid", "spades", 23), menu) == ("bid_spades_22", True)
    assert to_menu_label(Action("ask", "spades", 55), menu) == ("ask_spades_40", True)
    assert to_menu_label(Action("bid", "clubs", 3), menu) == ("pass", False)


def test_speed_specs():
    assert parse_speed("paper", None) == (1.0, 0.0) and parse_speed("equal", None) == (0.5, 0.3)
    a = make_agent("fundamentalist@1/200")
    assert (a.wake_rate, a.latency()) == (1.0, 200.0)
