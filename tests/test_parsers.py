"""Tests for the deterministic text parsers (conditions + targets)."""
from __future__ import annotations

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
