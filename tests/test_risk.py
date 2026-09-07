"""Unit tests for the liquidation risk engine. Pure math, no network."""

from __future__ import annotations

import pytest

from app.agent.risk import (
    distance_pct,
    evaluate_position,
    liq_price,
    margin_for_target,
    portfolio_risk_summary,
    position_risk_context,
    releasable_margin,
    required_cut,
    required_margin,
    risk_level,
)
from app.models import FuturesPosition, PositionSide, RiskLevel


def make_pos(side=PositionSide.LONG, entry=65_000.0, margin=1_000.0, leverage=20.0):
    qty = (margin * leverage) / entry
    return FuturesPosition(
        symbol="BTCUSDT",
        side=side,
        entry_price=entry,
        quantity=qty,
        leverage=leverage,
        margin_usdt=margin,
        mmr=0.005,
        mark_price=entry,
    )


def test_liq_price_20x_long_about_5_percent_below():
    pos = make_pos()
    liq = liq_price(pos.side, pos.entry_price, pos.quantity, pos.margin_usdt, pos.mmr)
    dist = (pos.entry_price - liq) / pos.entry_price
    assert 0.04 < dist < 0.05  # ~1/leverage minus MMR buffer


def test_liq_price_short_above_entry():
    pos = make_pos(side=PositionSide.SHORT)
    liq = liq_price(pos.side, pos.entry_price, pos.quantity, pos.margin_usdt, pos.mmr)
    assert liq > pos.entry_price


def test_distance_and_levels():
    pos = make_pos()
    d = distance_pct(PositionSide.LONG, 68_000.0, 62_000.0)
    assert abs(d - (68_000 - 62_000) / 68_000) < 1e-9
    assert risk_level(0.05, 0.10, 0.06) == RiskLevel.DANGER
    assert risk_level(0.08, 0.10, 0.06) == RiskLevel.WATCH
    assert risk_level(0.20, 0.10, 0.06) == RiskLevel.OK


def test_evaluate_position_sets_fields():
    pos = make_pos()
    out = evaluate_position(pos, 66_000.0, 0.10, 0.06)
    assert out.risk == RiskLevel.DANGER  # 20x position sits inside the danger zone
    assert out.unrealized_pnl_usdt > 0  # mark above entry on a long
    assert out.distance_pct > 0


def test_required_cut_moves_liq_to_target():
    """Deleverage: closing the returned qty (and keeping margin) must put the
    position at ~target headroom."""
    pos = make_pos()  # 20x, liq ~4.5% below entry
    mark = pos.entry_price
    target = 0.15
    cut = required_cut(pos, mark, target)
    assert 0 < cut < pos.quantity  # partial, not the whole position

    # apply the deleverage manually
    remaining = pos.quantity - cut
    new_liq = liq_price(pos.side, pos.entry_price, remaining, pos.margin_usdt, pos.mmr)
    dist = distance_pct(pos.side, mark, new_liq)
    assert dist >= target - 0.005  # within half a percent of the target


def test_required_cut_zero_when_already_safe():
    pos = make_pos(leverage=3.0)  # liq far away
    assert required_cut(pos, pos.entry_price, 0.15) == 0.0


def test_required_cut_smaller_for_lower_leverage():
    pos20 = make_pos(leverage=20.0)
    pos10 = make_pos(leverage=10.0)
    cut20 = required_cut(pos20, pos20.entry_price, 0.15)
    cut10 = required_cut(pos10, pos10.entry_price, 0.15)
    assert cut20 > cut10 > 0  # more levered = bigger cut needed


def test_required_margin_moves_liq():
    pos = make_pos()
    need = required_margin(pos, pos.entry_price, 0.15)
    assert need > 0
    new_liq = liq_price(
        pos.side, pos.entry_price, pos.quantity, pos.margin_usdt + need, pos.mmr
    )
    dist = distance_pct(pos.side, pos.entry_price, new_liq)
    assert dist >= 0.15 - 0.005


def test_required_margin_zero_when_safe():
    pos = make_pos(leverage=3.0)
    assert required_margin(pos, pos.entry_price, 0.15) == 0.0


def test_position_risk_context_returns_actionable_advice():
    pos = make_pos(leverage=20.0)
    pos.mark_price = pos.entry_price * 0.98
    evaluate_position(pos, pos.mark_price, 0.10, 0.06)
    ctx = position_risk_context(
        pos, portfolio_total=10_000.0, warn_pct=0.10, danger_pct=0.06
    )
    assert ctx["score"] >= 60
    assert ctx["advice"]
    advice = ctx["advice"].lower()
    assert "reduce" in advice or "add margin" in advice
    assert "risk" in ctx["reason"].lower() or "leverage" in ctx["reason"].lower()


def test_portfolio_risk_summary_ranks_the_most_dangerous_trade_first():
    safe = make_pos(leverage=5.0)
    safe.symbol = "ETHUSDT"
    safe.mark_price = safe.entry_price * 1.02
    evaluate_position(safe, safe.mark_price, 0.10, 0.06)

    danger = make_pos(leverage=25.0)
    danger.symbol = "BTCUSDT"
    danger.mark_price = danger.entry_price * 0.97
    evaluate_position(danger, danger.mark_price, 0.10, 0.06)

    summary = portfolio_risk_summary(
        [safe, danger], total_value_usdt=20_000.0, warn_pct=0.10, danger_pct=0.06
    )
    assert summary["top_risk"]["symbol"] == "BTCUSDT"
    assert (
        "reduce" in summary["top_risk"]["advice"].lower()
        or "add margin" in summary["top_risk"]["advice"].lower()
    )


# ---------------------------------------------------------------------------
# Two-way money movement helpers (margin_for_target / releasable_margin)
# ---------------------------------------------------------------------------


def test_margin_for_target_is_positive_when_position_is_under_margined():
    pos = make_pos(entry=65_000.0, margin=1_000.0, leverage=20.0)  # qty = 0.3077...
    need = margin_for_target(pos, 65_000.0, 0.15)
    # at 20x the position needs ~15.5x margin to reach 15% headroom, so the
    # target margin must exceed the current 1,000
    assert need > pos.margin_usdt
    # consistency: required_margin is exactly the shortfall
    assert required_margin(pos, 65_000.0, 0.15) == pytest.approx(need - pos.margin_usdt)


def test_releasable_margin_zero_when_still_under_margined():
    pos = make_pos(entry=65_000.0, margin=1_000.0, leverage=20.0)
    assert releasable_margin(pos, 65_000.0, 0.15, 50.0) == 0.0


def test_releasable_margin_pulls_only_excess_down_to_target_and_leverage():
    # over-margined position (5x effective) — safe to release some margin
    pos = make_pos(entry=65_000.0, margin=10_000.0, leverage=5.0)  # notional 50k
    mark = 65_000.0
    rel = releasable_margin(pos, mark, 0.15, 50.0)
    assert rel > 0.0
    keep = pos.margin_usdt - rel
    # after releasing, the remaining margin keeps at least 15% headroom
    assert keep >= margin_for_target(pos, mark, 0.15) - 0.01
    # and effective leverage stays far inside the generous 50x cap
    assert pos.entry_price * pos.quantity / keep <= 50.0 + 1e-6


def test_releasable_margin_never_breaks_leverage_cap():
    # a strict leverage cap must win over the headroom floor: releasing down to
    # the de-risk target would still leave effective leverage above the cap,
    # so the leverage bound keeps more margin on the position
    pos = make_pos(entry=65_000.0, margin=20_000.0, leverage=5.0)  # notional 100k
    mark = 65_000.0
    rel = releasable_margin(pos, mark, 0.15, 6.0)   # strict cap of 6x
    assert rel > 0.0
    keep = pos.margin_usdt - rel
    assert pos.entry_price * pos.quantity / keep <= 6.0 + 1e-6
    assert keep >= margin_for_target(pos, mark, 0.15) - 0.01
    # with a looser cap the same position could return more margin
    assert releasable_margin(pos, mark, 0.15, 50.0) > rel


def test_releasable_margin_zero_for_degenerate_inputs():
    empty = FuturesPosition(
        symbol="BTCUSDT", side=PositionSide.LONG, entry_price=0.0,
        quantity=0.0, leverage=10.0, margin_usdt=0.0, mmr=0.005, mark_price=0.0,
    )
    assert releasable_margin(empty, 0.0, 0.15, 50.0) == 0.0
    assert margin_for_target(empty, 0.0, 0.15) == 0.0
