"""Unit tests for the guardrail engine. Pure logic, no network."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.agent.guardrails import (
    check_cash_sufficient,
    check_cooldown,
    check_daily_trade_count,
    check_drawdown,
    check_max_trade_size,
    check_min_trade_value,
    check_symbol_allowed,
    evaluate_proposal,
)
from app.config import GuardrailConfig
from app.models import AccountState, Position, Side, TradeProposal


def cfg(**kw) -> GuardrailConfig:
    base = GuardrailConfig()
    for k, v in kw.items():
        setattr(base, k, v)
    return base


def state(total=10_000.0, cash=1_000.0, peak=None, trades=0, last=None) -> AccountState:
    return AccountState(
        total_value_usdt=total,
        cash_usdt=cash,
        positions=[
            Position(symbol="BTCUSDT", base="BTC", quote="USDT",
                     quantity=0.12, price=60_000.0, value_usdt=7_200.0, weight=0.72),
            Position(symbol="ETHUSDT", base="ETH", quote="USDT",
                     quantity=10.0, price=1_800.0, value_usdt=18_000.0, weight=1.8),
        ],
        peak_value_usdt=peak or total,
        realized_pnl_usdt=0.0,
        trade_count_today=trades,
        last_trade_at=last,
        updated_at=datetime.now(timezone.utc),
    )


def prop(symbol="BTCUSDT", side=Side.BUY, value=500.0) -> TradeProposal:
    return TradeProposal(
        id="t1", symbol=symbol, side=side,
        est_value_usdt=value, est_quantity=value / 60_000.0,
        est_price=60_000.0, reason="test",
    )


# ---------------------------------------------------------------------------


def test_symbol_allowlist_blocks_and_allows():
    g = cfg(symbol_allowlist=["BTCUSDT"])
    assert check_symbol_allowed("BTCUSDT", g).ok
    assert not check_symbol_allowed("DOGEUSDT", g).ok
    g2 = cfg(symbol_allowlist=[])
    assert check_symbol_allowed("DOGEUSDT", g2).ok


def test_min_trade_value():
    g = cfg(min_trade_value_usdt=10.0)
    assert check_min_trade_value(50.0, g).ok
    assert not check_min_trade_value(5.0, g).ok


def test_max_trade_size():
    g = cfg(max_trade_pct=0.10)
    assert check_max_trade_size(900.0, 10_000.0, g).ok
    assert not check_max_trade_size(1_100.0, 10_000.0, g).ok


def test_drawdown_halt():
    g = cfg(max_drawdown_pct=0.15)
    ok_state = state(total=9_000.0, peak=10_000.0)
    assert check_drawdown(ok_state, g).ok
    deep_state = state(total=8_000.0, peak=10_000.0)  # -20%
    assert not check_drawdown(deep_state, g).ok


def test_cooldown():
    g = cfg(cooldown_seconds=3600)
    now = datetime.now(timezone.utc)
    assert check_cooldown(None, now, g).ok
    assert check_cooldown(now - timedelta(seconds=7200), now, g).ok
    assert not check_cooldown(now - timedelta(seconds=60), now, g).ok


def test_daily_trade_cap():
    g = cfg(max_daily_trades=10)
    assert check_daily_trade_count(9, g).ok
    assert not check_daily_trade_count(10, g).ok


def test_cash_sufficient():
    g = cfg(fee_rate=0.001)
    assert check_cash_sufficient(1_000.0, 500.0, g).ok
    assert not check_cash_sufficient(100.0, 500.0, g).ok


def test_evaluate_proposal_buy_all_checks():
    g = cfg(symbol_allowlist=["BTCUSDT"], max_trade_pct=0.10,
            max_drawdown_pct=0.15, cooldown_seconds=3600, max_daily_trades=10,
            fee_rate=0.001, min_trade_value_usdt=10.0)
    now = datetime.now(timezone.utc)
    st = state(total=10_000.0, cash=9_000.0, trades=0, last=now - timedelta(hours=2))
    allowed, results = evaluate_proposal(prop("BTCUSDT", Side.BUY, 500.0), st, g, now=now)
    assert allowed
    assert all(r.ok for r in results)

    # drawdown halts buys
    st2 = state(total=8_000.0, cash=9_000.0, peak=10_000.0, last=now - timedelta(hours=2))
    allowed2, results2 = evaluate_proposal(prop("BTCUSDT", Side.BUY, 500.0), st2, g, now=now)
    assert not allowed2
    assert any("drawdown" in r.rule for r in results2 if not r.ok)


def test_evaluate_proposal_sell_allowed_in_drawdown():
    g = cfg(symbol_allowlist=["BTCUSDT"], max_trade_pct=0.10,
            max_drawdown_pct=0.15, cooldown_seconds=3600, max_daily_trades=10,
            fee_rate=0.001, min_trade_value_usdt=10.0)
    now = datetime.now(timezone.utc)
    st = state(total=8_000.0, cash=1_000.0, peak=10_000.0, trades=9, last=now - timedelta(seconds=5))
    allowed, _ = evaluate_proposal(prop("BTCUSDT", Side.SELL, 500.0), st, g, now=now)
    assert allowed  # de-risking is always allowed


def test_max_trade_size_epsilon_no_false_block():
    """A leg sized exactly at the cap (float noise ±1e-12) must not be blocked."""
    g = cfg(max_trade_pct=0.10)
    total = 9988.10
    cap = total * g.max_trade_pct  # 998.8099999999999...
    value = round(cap, 2)          # 998.81 — slightly ABOVE the raw float
    assert check_max_trade_size(value, total, g).ok


def test_evaluate_proposal_cooldown_waivable():
    g = cfg(symbol_allowlist=["BTCUSDT"], max_trade_pct=0.10,
            max_drawdown_pct=0.15, cooldown_seconds=3600, max_daily_trades=10,
            fee_rate=0.001, min_trade_value_usdt=10.0)
    now = datetime.now(timezone.utc)
    st = state(total=10_000.0, cash=9_000.0, last=now - timedelta(seconds=5))
    allowed, _ = evaluate_proposal(prop("BTCUSDT", Side.BUY, 500.0), st, g, now=now)
    assert not allowed  # cooldown blocks by default

    allowed2, results2 = evaluate_proposal(
        prop("BTCUSDT", Side.BUY, 500.0), st, g, now=now, respect_cooldown=False)
    assert allowed2
    assert any("cooldown" in r.rule and r.ok for r in results2)
