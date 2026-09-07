"""Tests for the deterministic text parsers (conditions + targets)."""
from __future__ import annotations

import pytest

from app.agent.narration import parse_intent, _parse_condition_from_text


def test_parse_condition_buy_below():
    c = _parse_condition_from_text("if BTC drops below 70000, buy 400 USDT")
    assert c["symbol"] == "BTCUSDT"
    assert c["op"] == "BELOW"
    assert c["price"] == 70_000.0
    assert c["side"] == "BUY"
    assert c["amount_usdt"] == 400.0


def test_parse_condition_sell_above():
    c = _parse_condition_from_text("when ETH goes above 4000, sell 200 usdt")
    assert c["symbol"] == "ETHUSDT"
    assert c["op"] == "ABOVE"
    assert c["price"] == 4_000.0
    assert c["side"] == "SELL"
    assert c["amount_usdt"] == 200.0


def test_parse_condition_default_amount():
    c = _parse_condition_from_text("if SOL drops to 150 buy")
    assert c["price"] == 150.0
    assert c["amount_usdt"] == 500.0


def test_parse_intent_keywords():
    assert parse_intent("protect my positions")["intent"] == "protect"
    assert parse_intent("de-risk now")["intent"] == "protect"
    assert parse_intent("risk report")["intent"] == "risk"
    assert parse_intent("what's my liquidation risk")["intent"] == "risk"
    assert parse_intent("show status")["intent"] == "status"
    assert parse_intent("if btc drops below 60k buy 300")["intent"] == "add_condition"
    assert parse_intent("hello there")["intent"] == "chat"


# ---------------------------------------------------------------------------
# Console command parsing (plain-language control of the app)
# ---------------------------------------------------------------------------


def test_parse_open_commands():
    r = parse_intent("open long BTC 500 at 10x")
    assert r["intent"] == "open"
    assert r == {"intent": "open", "symbol": "BTCUSDT", "side": "LONG",
                 "margin_usdt": 500.0, "leverage": 10.0}
    r2 = parse_intent("long btc 500 20x")
    assert r2["side"] == "LONG" and r2["margin_usdt"] == 500.0 and r2["leverage"] == 20.0
    r3 = parse_intent("open short eth 300")
    assert r3["side"] == "SHORT" and r3["symbol"] == "ETHUSDT" and r3["leverage"] == 10.0


def test_parse_close_commands():
    r = parse_intent("close BTC")
    assert r["intent"] == "close" and r["symbol"] == "BTCUSDT"
    r2 = parse_intent("close all my positions")
    assert r2["intent"] == "close" and r2["close_all"] is True and not r2["symbol"]


def test_parse_add_margin_commands():
    for s in ["add 300 margin to BTC", "add margin 250 to btc", "top up eth 100",
              "add 200 more margin on SOL"]:
        r = parse_intent(s)
        assert r["intent"] == "add_margin", s
        assert r["amount_usdt"], s
        assert r["symbol"].endswith("USDT")


def test_parse_release_margin_commands():
    for s in ["release margin on btc", "pull margin from sol", "return margin on eth"]:
        r = parse_intent(s)
        assert r["intent"] == "release", s
        assert r["symbol"].endswith("USDT")


def test_parse_funds_move_commands():
    r = parse_intent("move 500 to futures")
    assert r == {"intent": "funds", "direction": "to_futures", "amount_usdt": 500.0}
    r2 = parse_intent("transfer 250 to spot")
    assert r2["direction"] == "to_spot" and r2["amount_usdt"] == 250.0
    r3 = parse_intent("move to futures")
    assert r3["direction"] == "to_futures" and r3["amount_usdt"] is None


def test_parse_agent_mode_commands():
    assert parse_intent("auto mode")["intent"] == "agent_auto"
    assert parse_intent("enable automatic agent")["intent"] == "agent_auto"
    assert parse_intent("manual mode")["intent"] == "agent_manual"
    assert parse_intent("back to manual")["intent"] == "agent_manual"


def test_parse_rail_edits():
    r = parse_intent("set danger zone to 5%")
    assert r["intent"] == "rail_edit" and r["rail_key"] == "liq_danger_pct"
    assert r["rail_value"] == pytest.approx(0.05)
    r2 = parse_intent("set de-risk target to 12%")
    assert r2["rail_key"] == "liq_target_dist_pct" and r2["rail_value"] == pytest.approx(0.12)
    # retired knobs are not editable through the console any more
    assert parse_intent("raise leverage cap to 30x")["intent"] == "chat"
    assert parse_intent("set min trade value to 20")["intent"] == "chat"
    # fractions work without a % sign too
    r4 = parse_intent("set watch zone to 0.08")
    assert r4["rail_key"] == "liq_warn_pct" and r4["rail_value"] == pytest.approx(0.08)
