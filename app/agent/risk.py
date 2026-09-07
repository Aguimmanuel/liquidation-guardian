"""Liquidation risk engine.

Pure math for USDⓈ-M futures liquidation protection:

- ``liq_price``          — isolated-margin liquidation price for long/short
- ``distance_pct``       — how far the asset price is from liquidation
- ``risk_level``         — OK / WATCH / DANGER zones
- ``required_cut``       — base qty to close so the remaining position sits at
                           a target distance from liquidation
- ``required_margin``    — USDT to add so the position sits at a target
                           distance from liquidation

Isolated-margin model (no fees): liquidation occurs when
    margin + unrealized PnL = maintenance margin
    P_liq · qty · MMR = M + (P_liq − entry) · qty          (long)
    P_liq · qty · MMR = M + (entry − P_liq) · qty          (short)

which solves to:

    long:   P_liq = (entry · qty − M) / (qty · (1 − MMR))
    short:  P_liq = (entry · qty + M) / (qty · (1 + MMR))

With M = entry·qty/leverage this reduces to the familiar
    P_liq ≈ entry · (1 ∓ 1/leverage).

Fees, funding and cross-margin positions are not modelled; the live adapter
reads real positions from the Binance MCP account scope. Everything here is
deterministic and unit-tested.
"""

from __future__ import annotations

from app.models import FuturesPosition, PositionSide, RiskLevel


def liq_price(
    side: PositionSide,
    entry_price: float,
    quantity: float,
    margin_usdt: float,
    mmr: float,
) -> float:
    if quantity <= 0 or mmr >= 1.0:
        return 0.0
    notional = entry_price * quantity
    if side == PositionSide.LONG:
        denom = quantity * (1.0 - mmr)
        return (notional - margin_usdt) / denom if denom > 0 else 0.0
    denom = quantity * (1.0 + mmr)
    return (notional + margin_usdt) / denom if denom > 0 else 0.0


def distance_pct(side: PositionSide, mark_price: float, liq: float) -> float:
    """Fractional move of the asset price that triggers liquidation.
    Positive = safe (price must move *against* you by this much)."""
    if mark_price <= 0 or liq <= 0:
        return 0.0
    if side == PositionSide.LONG:
        return (mark_price - liq) / mark_price
    return (liq - mark_price) / mark_price


def risk_level(dist: float, warn_pct: float, danger_pct: float) -> RiskLevel:
    if dist <= danger_pct:
        return RiskLevel.DANGER
    if dist <= warn_pct:
        return RiskLevel.WATCH
    return RiskLevel.OK


def evaluate_position(
    pos: FuturesPosition, mark_price: float, warn_pct: float, danger_pct: float
) -> FuturesPosition:
    """Refresh a position's derived risk fields from a live mark price."""
    pos.mark_price = mark_price
    pos.liq_price = liq_price(
        pos.side, pos.entry_price, pos.quantity, pos.margin_usdt, pos.mmr
    )
    pos.distance_pct = distance_pct(pos.side, mark_price, pos.liq_price)
    pos.risk = risk_level(pos.distance_pct, warn_pct, danger_pct)
    if pos.side == PositionSide.LONG:
        pos.unrealized_pnl_usdt = (mark_price - pos.entry_price) * pos.quantity
    else:
        pos.unrealized_pnl_usdt = (pos.entry_price - mark_price) * pos.quantity
    return pos


def _liq_at_distance(
    pos: FuturesPosition, mark_price: float, target_dist: float
) -> float:
    """The liquidation price that yields exactly ``target_dist`` of headroom."""
    if pos.side == PositionSide.LONG:
        return mark_price * (1.0 - target_dist)
    return mark_price * (1.0 + target_dist)


def required_cut(
    pos: FuturesPosition,
    mark_price: float,
    target_dist: float,
) -> float:
    """Base quantity to close so the remaining position is ``target_dist``
    away from liquidation — **deleveraging**: the proceeds from the closed
    part are kept as margin on the remainder, so the liq price actually moves.

    Closing a position proportionally (releasing margin proportionally) leaves
    the liq price unchanged — that's why a naive "cut" doesn't help. The
    rational de-risk is: close `cut` of the position AND keep the margin, which
    lowers the effective leverage.

    Closed-form (isolated margin, long):
        Q' = M / (entry - target_liq·(1 - MMR))
        cut = Q - Q'

    Returns 0.0 if the position is already safe enough.
    """
    if pos.quantity <= 0 or pos.margin_usdt <= 0:
        return 0.0
    if distance_pct(pos.side, mark_price, pos.liq_price) >= target_dist:
        return 0.0
    target_liq = _liq_at_distance(pos, mark_price, target_dist)
    if pos.side == PositionSide.LONG:
        denom = pos.entry_price - target_liq * (1.0 - pos.mmr)
    else:
        denom = target_liq * (1.0 + pos.mmr) - pos.entry_price
    if denom <= 0:
        return round(pos.quantity, 8)  # even closing everything can't reach it
    q_remain = pos.margin_usdt / denom
    cut = pos.quantity - q_remain
    if cut <= 0:
        return 0.0
    # cap at 99% — a "reduce" leaves a sliver; full exit is a separate choice
    cut = min(cut, pos.quantity * 0.99)
    return round(cut, 8)


def required_margin(
    pos: FuturesPosition,
    mark_price: float,
    target_dist: float,
) -> float:
    """USDT of additional margin so the position is ``target_dist`` away from
    liquidation (position size unchanged)."""
    target_liq = _liq_at_distance(pos, mark_price, target_dist)
    notional = pos.entry_price * pos.quantity
    if pos.side == PositionSide.LONG:
        needed = notional - target_liq * pos.quantity * (1.0 - pos.mmr)
    else:
        needed = target_liq * pos.quantity * (1.0 + pos.mmr) - notional
    return round(max(0.0, needed - pos.margin_usdt), 2)


def position_summary(pos: FuturesPosition) -> dict:
    """One-line plain-English summary used in narration and the UI."""
    return {
        "symbol": pos.symbol,
        "side": pos.side.value,
        "entry": pos.entry_price,
        "mark": pos.mark_price,
        "leverage": pos.leverage,
        "margin": pos.margin_usdt,
        "liq": pos.liq_price,
        "distance": pos.distance_pct,
        "risk": pos.risk.value,
    }


def position_risk_context(
    pos: FuturesPosition,
    portfolio_total: float,
    warn_pct: float,
    danger_pct: float,
) -> dict:
    """Turn a single position into a simple, actionable risk recommendation.

    The goal is not to replace the engine; it is to improve the user's decision
    quality by attaching a concrete reason and next action.
    """
    dist = float(pos.distance_pct)
    score = 10
    if pos.risk == RiskLevel.DANGER:
        score = max(65, int(85 + (danger_pct - dist) * 250))
    elif pos.risk == RiskLevel.WATCH:
        score = max(35, int(55 + (warn_pct - dist) * 200))
    else:
        score = max(10, int(20 + (0.15 - dist) * 120))
    score = max(0, min(100, score))

    share = (pos.notional_usdt / portfolio_total) if portfolio_total > 0 else 0.0
    if pos.risk == RiskLevel.DANGER:
        target = max(warn_pct + 0.05, 0.15)
        cut = required_cut(pos, pos.mark_price, target)
        add = required_margin(pos, pos.mark_price, target)
        if cut > 0:
            advice = (
                f"Reduce {pos.symbol} now by about {cut:.4f} units or add margin of "
                f"${add:,.2f} to restore roughly {target * 100:.0f}% headroom."
            )
        else:
            advice = (
                f"Add margin of ${add:,.2f} to {pos.symbol} to restore safer risk."
                if add > 0
                else f"Reduce {pos.symbol} immediately — it is too close to liquidation."
            )
        reason = (
            f"{pos.symbol} is {dist * 100:.1f}% from liquidation, effective leverage is "
            f"{pos.effective_leverage:.1f}x, and it represents {share * 100:.1f}% of portfolio value."
        )
    elif pos.risk == RiskLevel.WATCH:
        add = required_margin(pos, pos.mark_price, warn_pct)
        advice = (
            f"Add margin of ${add:,.2f} or trim {pos.symbol} before it enters the danger zone."
            if add > 0
            else f"Monitor {pos.symbol} closely; keep leverage in check until risk improves."
        )
        reason = (
            f"{pos.symbol} is nearing liquidation risk at {dist * 100:.1f}% away, with "
            f"{pos.effective_leverage:.1f}x effective leverage and {share * 100:.1f}% portfolio weight."
        )
    else:
        advice = (
            f"Keep monitoring {pos.symbol}; no immediate de-risk action is required."
        )
        reason = (
            f"{pos.symbol} remains relatively stable at {dist * 100:.1f}% from liquidation, "
            f"with {pos.effective_leverage:.1f}x effective leverage."
        )

    return {
        "symbol": pos.symbol,
        "side": pos.side.value,
        "risk": pos.risk.value,
        "distance_pct": round(dist * 100.0, 2),
        "score": score,
        "portfolio_weight_pct": round(share * 100.0, 2),
        "reason": reason,
        "advice": advice,
    }


def portfolio_risk_summary(
    positions: list[FuturesPosition],
    total_value_usdt: float,
    warn_pct: float,
    danger_pct: float,
) -> dict:
    """Rank the current open positions by risk and return the top recommendation."""
    analyzed = [
        position_risk_context(pos, total_value_usdt, warn_pct, danger_pct)
        for pos in positions
    ]
    if not analyzed:
        return {
            "total_positions": 0,
            "danger_count": 0,
            "watch_count": 0,
            "top_risk": None,
            "positions": [],
        }
    ranked = sorted(analyzed, key=lambda x: x["score"], reverse=True)
    top = ranked[0]
    return {
        "total_positions": len(analyzed),
        "danger_count": sum(1 for p in ranked if p["risk"] == RiskLevel.DANGER.value),
        "watch_count": sum(1 for p in ranked if p["risk"] == RiskLevel.WATCH.value),
        "top_risk": top,
        "positions": ranked,
    }
