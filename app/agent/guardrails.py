"""Guardrail engine.

Deterministic, unit-testable risk checks. Every proposal the agent queues is
filtered through this engine, mirroring how Binance Agent OS keeps agents
inside a dedicated sub-account with user-set permissions: no withdrawals, hard
caps, and confirm-before-execute.

The guardian only *protects*, so nothing here may ever freeze a protective
action. Retired on purpose: a per-day trade budget, max-order-size, cooldown,
a dust minimum, and a leverage cap — each proved decorative for a protection
agent (they could block or gate nothing that actually protects you), so they
are gone. What remains are the checks the flows truly run: allowed symbols,
cash/transfer sufficiency, and the existence of the position being protected.

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
