"""Unit tests for the guardrail engine. Pure logic, no network.

The engine keeps exactly the checks the app's flows actually use. Removed on
purpose (they could not work with a protection agent): per-day trade budget,
max-order-size, cooldown, drawdown-halt — none of which may ever freeze a
de-risk. The remaining knobs are min-trade-value and the leverage cap, both
gated elsewhere in the orchestrator; here we test the pure checks.
"""
from __future__ import annotations

from app.agent.guardrails import (
    check_futures_funds_sufficient,
    check_futures_position_exists,
    check_min_trade_value,
    check_spot_funds_sufficient,
    check_symbol_allowed,
)
from app.config import GuardrailConfig


def cfg(**kw) -> GuardrailConfig:
    base = GuardrailConfig()
    for k, v in kw.items():
        setattr(base, k, v)
    return base


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


def test_spot_funds_sufficient():
    assert check_spot_funds_sufficient(1_000.0, 500.0).ok
    assert not check_spot_funds_sufficient(100.0, 500.0).ok


def test_futures_funds_sufficient():
    assert check_futures_funds_sufficient(800.0, 500.0).ok
    assert not check_futures_funds_sufficient(100.0, 500.0).ok


def test_futures_position_exists():
    assert check_futures_position_exists(True, "BTCUSDT").ok
    assert not check_futures_position_exists(False, "BTCUSDT").ok


def test_removed_rails_do_not_exist():
    """Max order size, cooldown, drawdown halt and the per-day budget were
    removed: a calendar/portfolio cap must never freeze protective action."""
    g = GuardrailConfig()
    for gone in ("max_trade_pct", "cooldown_seconds", "max_drawdown_pct",
                 "max_daily_trades", "daily_trades_locked", "daily_lock_at",
                 "drift_threshold_pct"):
        assert not hasattr(g, gone), gone


def test_config_surfaces_only_live_rails():
    d = GuardrailConfig().to_dict()
    for kept in ("liq_warn_pct", "liq_danger_pct", "liq_target_dist_pct",
                 "min_trade_value_usdt", "max_leverage"):
        assert kept in d
    for gone in ("max_trade_pct", "cooldown_seconds", "max_daily_trades"):
        assert gone not in d
