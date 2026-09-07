"""Guardrail engine.

Deterministic, unit-testable risk checks. The agent may *propose* anything it
computes, but every proposal is filtered through this engine, mirroring how
Binance Agent OS keeps agents inside a dedicated sub-account with user-set
permissions: no withdrawals, hard caps, and confirm-before-execute.

Each check returns a `CheckResult` — (ok, reason). A failing check NEVER throws;
it produces a transparent, human-readable block reason that is surfaced in the
UI and the audit trail. Transparency is a feature, not an afterthought.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from app.config import GuardrailConfig
from app.models import AccountState, Side, TradeProposal


@dataclass
class CheckResult:
    ok: bool
    rule: str
    reason: str = ""

    def to_dict(self) -> dict:
        return {"ok": self.ok, "rule": self.rule, "reason": self.reason}


class GuardrailError(Exception):
    """Raised only for truly unrecoverable config errors (never for blocks)."""


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_symbol_allowed(symbol: str, cfg: GuardrailConfig) -> CheckResult:
    if not cfg.symbol_allowlist:
        return CheckResult(True, "symbol_allowlist", "no allowlist configured")
    ok = symbol in cfg.symbol_allowlist
    return CheckResult(
        ok,
        "symbol_allowlist",
        "" if ok else f"{symbol} is not on the approved symbol list",
    )


def check_min_trade_value(value_usdt: float, cfg: GuardrailConfig) -> CheckResult:
    ok = value_usdt >= cfg.min_trade_value_usdt - 1e-6
    return CheckResult(
        ok,
        "min_trade_value",
        "" if ok else f"order value ${value_usdt:,.2f} is below the ${cfg.min_trade_value_usdt:,.2f} minimum",
    )


def check_max_trade_size(
    value_usdt: float,
    total_value_usdt: float,
    cfg: GuardrailConfig,
) -> CheckResult:
    cap = round(total_value_usdt * cfg.max_trade_pct, 2)
    ok = value_usdt <= cap + 1e-6
    return CheckResult(
        ok,
        "max_trade_size",
        "" if ok else f"order value ${value_usdt:,.2f} exceeds {cfg.max_trade_pct:.0%} of portfolio (${cap:,.2f})",
    )


def check_drawdown(state: AccountState, cfg: GuardrailConfig) -> CheckResult:
    """If drawdown exceeds the cap, block new BUYS. Selling to de-risk is always
    allowed — that is the safe direction."""
    if state.peak_value_usdt <= 0:
        return CheckResult(True, "drawdown", "")
    drawdown = state.total_value_usdt / state.peak_value_usdt - 1.0
    ok = drawdown >= -cfg.max_drawdown_pct
    return CheckResult(
        ok,
        "drawdown",
        "" if ok else f"drawdown {drawdown:.1%} exceeds the {cfg.max_drawdown_pct:.0%} cap — buys are paused",
    )


def check_cooldown(
    last_trade_at: datetime | None,
    now: datetime,
    cfg: GuardrailConfig,
) -> CheckResult:
    if last_trade_at is None:
        return CheckResult(True, "cooldown", "")
    elapsed = (now - last_trade_at).total_seconds()
    ok = elapsed >= cfg.cooldown_seconds
    return CheckResult(
        ok,
        "cooldown",
        "" if ok else f"cooldown active — last trade {int(elapsed // 60)}m ago (need {cfg.cooldown_seconds // 60}m)",
    )


def check_cash_sufficient(
    cash_usdt: float,
    buy_value_usdt: float,
    cfg: GuardrailConfig,
) -> CheckResult:
    ok = cash_usdt >= buy_value_usdt * (1.0 + cfg.fee_rate)
    return CheckResult(
        ok,
        "cash_sufficient",
        "" if ok else f"insufficient cash: need ${buy_value_usdt * (1 + cfg.fee_rate):,.2f}, have ${cash_usdt:,.2f}",
    )


def check_transfer_sufficient(
    cash_usdt: float,
    amount_usdt: float,
    cfg: GuardrailConfig,
) -> CheckResult:
    """A margin transfer must be fully covered by available spot cash."""
    ok = cash_usdt >= amount_usdt
    return CheckResult(
        ok,
        "transfer_sufficient",
        "" if ok else f"not enough cash to add ${amount_usdt:,.2f} margin (have ${cash_usdt:,.2f})",
    )


def check_futures_position_exists(
    has_position: bool,
    symbol: str,
) -> CheckResult:
    return CheckResult(
        has_position,
        "futures_position",
        "" if has_position else f"no open {symbol} futures position to protect",
    )


# ---------------------------------------------------------------------------
# Aggregate gate
# ---------------------------------------------------------------------------


def evaluate_proposal(
    proposal: TradeProposal,
    state: AccountState,
    cfg: GuardrailConfig,
    *,
    now: datetime | None = None,
    respect_cooldown: bool = True,
) -> tuple[bool, list[CheckResult]]:
    """Run every applicable guardrail against a proposal.

    Returns (allowed, results). For SELL proposals, the drawdown check is
    relaxed (selling to de-risk is always permitted).

    ``respect_cooldown``: the cooldown gate applies to *autonomous* actions
    (conditions firing on their own). Explicit user-commanded actions pass
    ``respect_cooldown=False`` — the human is already in the loop approving
    every execution, so the 1h clock would otherwise freeze completing a plan
    right after the sells that funded it.
    """
    now = now or datetime.now(timezone.utc)
    results: list[CheckResult] = []

    results.append(check_symbol_allowed(proposal.symbol, cfg))
    results.append(check_min_trade_value(proposal.est_value_usdt, cfg))
    results.append(check_max_trade_size(proposal.est_value_usdt, state.total_value_usdt, cfg))

    if proposal.side == Side.BUY:
        results.append(check_drawdown(state, cfg))
        if respect_cooldown:
            results.append(check_cooldown(state.last_trade_at, now, cfg))
        else:
            results.append(CheckResult(True, "cooldown", "user-commanded action — cooldown waived"))
        results.append(check_cash_sufficient(state.cash_usdt, proposal.est_value_usdt, cfg))
    else:
        results.append(CheckResult(True, "drawdown", "SELL allowed in drawdown (de-risking direction)"))

    blocked = [r for r in results if not r.ok]
    allowed = not blocked
    return allowed, results
