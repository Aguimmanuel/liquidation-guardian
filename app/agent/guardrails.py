"""Guardrail engine.

Deterministic, unit-testable risk checks. Every proposal the agent queues is
filtered through this engine, mirroring how Binance Agent OS keeps agents
inside a dedicated sub-account with user-set permissions: no withdrawals, hard
caps, and confirm-before-execute.

The guardian only *protects* — it never opens positions for profit and never
caps protective action by calendar or portfolio-size rules. (A legacy per-day
trade budget, and a max-order-size/cooldown pair that could freeze a de-risk
mid-crash, were removed.) What remains gates the flows that actually run:
allowed symbols, dust-sized actions, cash/transfer sufficiency, and the
existence of the position being protected.

Each check returns a `CheckResult` — (ok, reason). A failing check NEVER throws;
it produces a transparent, human-readable block reason that is surfaced in the
UI and the audit trail. Transparency is a feature, not an afterthought.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.config import GuardrailConfig


@dataclass
class CheckResult:
    ok: bool
    rule: str
    reason: str = ""

    def to_dict(self) -> dict:
        return {"ok": self.ok, "rule": self.rule, "reason": self.reason}


class GuardrailError(Exception):
    """Raised only for truly unrecoverable config errors (never for blocks)."""


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


def check_spot_funds_sufficient(
    cash_usdt: float,
    amount_usdt: float,
) -> CheckResult:
    """A spot->futures move must be fully covered by available spot cash."""
    ok = cash_usdt >= amount_usdt
    return CheckResult(
        ok,
        "spot_cash",
        "" if ok else f"not enough spot cash: have ${cash_usdt:,.2f}, want ${amount_usdt:,.2f}",
    )


def check_futures_funds_sufficient(
    wallet_usdt: float,
    amount_usdt: float,
) -> CheckResult:
    """A futures->spot move must be covered by the free futures wallet."""
    ok = wallet_usdt >= amount_usdt
    return CheckResult(
        ok,
        "futures_balance",
        "" if ok else f"free futures balance is ${wallet_usdt:,.2f} — cannot move ${amount_usdt:,.2f}",
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
