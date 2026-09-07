"""Liquidation Guardian configuration.

All settings can be overridden through environment variables so the same code
runs in simulation (default), with a live Binance Agentic sub-account, or in
tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Runtime-editable guardrail profile
# ---------------------------------------------------------------------------

# The numeric limits a user can edit from the UI/API and that are persisted
# with the profile (survive restarts).
EDITABLE_RAIL_KEYS: tuple[str, ...] = (
    "liq_warn_pct",
    "liq_danger_pct",
    "liq_target_dist_pct",
    "max_trade_pct",
    "min_trade_value_usdt",
    "cooldown_seconds",
    "max_daily_trades",
    "max_leverage",
)

# One-way ratchet: for each editable limit, which direction of change is a
# *loosening* (blocked while the profile is locked):
#   +1  -> increasing the value loosens the limit
#   -1  -> decreasing the value loosens the limit
#    0  -> not a risk ceiling (either direction allowed while locked)
LOOSEN_DIRECTION: dict[str, int] = {
    "max_daily_trades": +1,     # more trades allowed = looser
    "max_leverage": +1,         # more leverage = looser
    "max_trade_pct": +1,        # bigger orders = looser
    "min_trade_value_usdt": 0,  # dust filter — not a risk ceiling
    "cooldown_seconds": -1,     # shorter gap between auto actions = looser
    "liq_warn_pct": -1,         # later warning = looser
    "liq_danger_pct": -1,       # later intervention = looser
    "liq_target_dist_pct": -1,  # less post-de-risk headroom = looser
}

# Numeric sanity bounds applied on every guardrail edit (kept in one place so
# a locked profile can still never be driven into a nonsense value).
RAIL_BOUNDS: dict[str, tuple[float, float]] = {
    "liq_warn_pct": (0.0, 1.0),
    "liq_danger_pct": (0.0, 1.0),
    "liq_target_dist_pct": (0.0, 1.0),
    "max_trade_pct": (0.0, 1.0),
    "min_trade_value_usdt": (0.0, float("inf")),
    "cooldown_seconds": (0.0, float("inf")),
    "max_daily_trades": (1.0, float("inf")),
    "max_leverage": (1.0, 125.0),  # Binance USDⓈ-M hard max
}


# ---------------------------------------------------------------------------
# Guardrails: hard limits the agent can never cross, mirroring the philosophy
# of Binance Agent OS (sub-account isolation, no withdrawals, confirm-first).
# ---------------------------------------------------------------------------


@dataclass
class GuardrailConfig:
    max_trade_pct: float = 0.10  # one order may not exceed 10% of portfolio value
    max_drawdown_pct: float = 0.15  # portfolio drawdown from peak halts new buys
    min_trade_value_usdt: float = 10.0  # ignore dust-sized orders
    max_daily_trades: int = 10  # hard cap on executions per UTC day
    cooldown_seconds: int = 3600  # minimum gap between autonomous actions
    drift_threshold_pct: float = 0.05  # unused by the guardian (kept for config parity)
    target_cash_pct: float = 0.10  # default cash reserve target
    # -- liquidation protection (futures) --
    liq_warn_pct: float = 0.10  # distance to liq below this = WATCH
    liq_danger_pct: float = (
        0.06  # distance to liq below this = DANGER -> propose de-risk
    )
    liq_target_dist_pct: float = 0.15  # de-risk until 15% headroom to liquidation
    futures_fee_rate: float = 0.0004  # simulated taker fee (0.04%, USDⓈ-M)
    symbol_allowlist: list[str] = field(
        default_factory=lambda: [
            "BTCUSDT",
            "ETHUSDT",
            "BNBUSDT",
            "SOLUSDT",
        ]
    )  # empty list = allow any symbol from market feed
    fee_rate: float = 0.001  # simulated taker fee (0.1%), Binance spot
    # -- runtime-editable guardrail state (set from the UI) --
    max_leverage: float = 50.0  # cap when opening a position (UI enforces <= this)
    # The commitment lock. When engaged it freezes the entire editable risk
    # profile (not just the daily cap) for one 24-hour window: loosening any
    # limit is refused until the window ends, tightening is always allowed,
    # and there is no early unlock. Persisted (values + lock) in
    # guardrail_state.json.
    daily_trades_locked: bool = False
    daily_lock_at: Optional[datetime] = None
    mcp_url: str = "https://agent.binance.com/mcp/agentic"
    mcp_access_token: str = ""  # optional pre-issued Bearer token

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_trade_pct": self.max_trade_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "min_trade_value_usdt": self.min_trade_value_usdt,
            "max_daily_trades": self.max_daily_trades,
            "cooldown_seconds": self.cooldown_seconds,
            "drift_threshold_pct": self.drift_threshold_pct,
            "target_cash_pct": self.target_cash_pct,
            "symbol_allowlist": self.symbol_allowlist,
            "fee_rate": self.fee_rate,
            "liq_warn_pct": self.liq_warn_pct,
            "liq_danger_pct": self.liq_danger_pct,
            "liq_target_dist_pct": self.liq_target_dist_pct,
            "futures_fee_rate": self.futures_fee_rate,
            "max_leverage": self.max_leverage,
            "daily_trades_locked": self.daily_trades_locked,
            "daily_lock_at": self.daily_lock_at.isoformat()
            if self.daily_lock_at
            else None,
        }


# ---------------------------------------------------------------------------
# Application config
# ---------------------------------------------------------------------------


@dataclass
class AppConfig:
    mode: str = "sim"  # "sim" | "live"
    market_base_url: str = "https://data-api.binance.vision"
    market_ttl_seconds: int = 30
    ticker_ttl_seconds: int = 1  # live ticker strip refresh interval (s)
    state_file: str = "data/sim_state.json"
    guardrail_state_file: str = "data/guardrail_state.json"
    default_initial_cash: float = 10_000.0
    default_targets: dict[str, float] = field(
        default_factory=lambda: {
            "BTCUSDT": 0.40,
            "ETHUSDT": 0.30,
            "BNBUSDT": 0.20,
            "USDT": 0.10,
        }
    )
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    narration_llm: bool = False  # True = LLM narration, False = template
    host: str = "0.0.0.0"
    port: int = 8000

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "market_base_url": self.market_base_url,
            "ticker_ttl_seconds": self.ticker_ttl_seconds,
            "narration_llm": self.narration_llm,
        }


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if not raw:
        return default
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def _env_targets() -> dict[str, float]:
    raw = os.environ.get("RS_TARGETS")
    if not raw:
        return {"BTCUSDT": 0.40, "ETHUSDT": 0.30, "BNBUSDT": 0.20, "USDT": 0.10}
    out: dict[str, float] = {}
    for part in raw.split(","):
        if "=" not in part:
            continue
        sym, weight = part.split("=")
        out[sym.strip().upper()] = float(weight)
    return out


def load_config() -> AppConfig:
    cfg = AppConfig(
        mode=os.environ.get("RS_MODE", "sim").lower(),
        market_base_url=os.environ.get(
            "RS_MARKET_BASE_URL", "https://data-api.binance.vision"
        ),
        market_ttl_seconds=_env_int("RS_MARKET_TTL", 30),
        ticker_ttl_seconds=_env_int("RS_TICKER_TTL", 1),
        state_file=os.environ.get("RS_STATE_FILE", "data/sim_state.json"),
        guardrail_state_file=os.environ.get(
            "RS_GUARDRAIL_STATE_FILE", "data/guardrail_state.json"
        ),
        default_initial_cash=_env_float("RS_INITIAL_CASH", 10_000.0),
        default_targets=_env_targets(),
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        openai_base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        openai_model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        narration_llm=_env_bool("RS_NARRATION_LLM", False),
    )
    return cfg


def load_guardrails() -> GuardrailConfig:
    """Build the guardrail config from environment (independent of app config)."""
    return GuardrailConfig(
        max_trade_pct=_env_float("RS_MAX_TRADE_PCT", 0.10),
        max_drawdown_pct=_env_float("RS_MAX_DRAWDOWN_PCT", 0.15),
        min_trade_value_usdt=_env_float("RS_MIN_TRADE_VALUE_USDT", 10.0),
        max_daily_trades=_env_int("RS_MAX_DAILY_TRADES", 10),
        cooldown_seconds=_env_int("RS_COOLDOWN_SECONDS", 3600),
        drift_threshold_pct=_env_float("RS_DRIFT_THRESHOLD_PCT", 0.05),
        target_cash_pct=_env_float("RS_TARGET_CASH_PCT", 0.10),
        symbol_allowlist=_env_list(
            "RS_SYMBOL_ALLOWLIST", ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
        ),
        fee_rate=_env_float("RS_FEE_RATE", 0.001),
        liq_warn_pct=_env_float("RS_LIQ_WARN_PCT", 0.10),
        liq_danger_pct=_env_float("RS_LIQ_DANGER_PCT", 0.06),
        liq_target_dist_pct=_env_float("RS_LIQ_TARGET_DIST_PCT", 0.15),
        futures_fee_rate=_env_float("RS_FUTURES_FEE_RATE", 0.0004),
        max_leverage=_env_float("RS_MAX_LEVERAGE", 50.0),
        mcp_url=os.environ.get("RS_MCP_URL", "https://agent.binance.com/mcp/agentic"),
        mcp_access_token=os.environ.get("RS_MCP_ACCESS_TOKEN", ""),
    )
