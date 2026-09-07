"""Domain models for the Liquidation Guardian."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime | None) -> str:
    return dt.isoformat() if dt else ""


# ---------------------------------------------------------------------------
# Market
# ---------------------------------------------------------------------------


@dataclass
class Ticker:
    symbol: str
    price: float
    change_24h_pct: Optional[float] = None
    ts: datetime = field(default_factory=utcnow)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "price": self.price,
            "change_24h_pct": self.change_24h_pct,
            "ts": iso(self.ts),
        }


# ---------------------------------------------------------------------------
# Futures positions
# ---------------------------------------------------------------------------


class PositionSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class RiskLevel(str, Enum):
    OK = "OK"
    WATCH = "WATCH"
    DANGER = "DANGER"


@dataclass
class FuturesPosition:
    """A USDⓈ-M futures position (isolated margin model)."""

    symbol: str
    side: PositionSide
    entry_price: float
    quantity: float            # base-asset quantity
    leverage: float
    margin_usdt: float         # margin allocated to this position
    mmr: float = 0.005         # maintenance margin rate (0.5% default)
    mark_price: float = 0.0
    # user-armed exits (0 = not armed)
    take_profit_price: float = 0.0
    stop_loss_price: float = 0.0
    tp_hit: bool = False
    sl_hit: bool = False
    # derived (computed by the risk engine on refresh)
    liq_price: float = 0.0
    distance_pct: float = 0.0  # % move of the asset price to liquidation
    risk: RiskLevel = RiskLevel.OK
    unrealized_pnl_usdt: float = 0.0

    @property
    def notional_usdt(self) -> float:
        return self.quantity * (self.mark_price or self.entry_price)

    @property
    def effective_leverage(self) -> float:
        """Actual leverage right now: notional / margin. Drops when the
        guardian deleverages (margin stays, quantity shrinks)."""
        if self.margin_usdt <= 0:
            return 0.0
        return self.notional_usdt / self.margin_usdt

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "entry_price": self.entry_price,
            "mark_price": self.mark_price,
            "quantity": self.quantity,
            "notional_usdt": round(self.notional_usdt, 2),
            "leverage": self.leverage,
            "effective_leverage": round(self.effective_leverage, 1),
            "margin_usdt": round(self.margin_usdt, 2),
            "take_profit_price": self.take_profit_price,
            "stop_loss_price": self.stop_loss_price,
            "tp_hit": self.tp_hit,
            "sl_hit": self.sl_hit,
            "liq_price": self.liq_price,
            "distance_pct": round(self.distance_pct * 100, 2),
            "risk": self.risk.value,
            "unrealized_pnl_usdt": round(self.unrealized_pnl_usdt, 2),
        }


# ---------------------------------------------------------------------------
# Portfolio / account
# ---------------------------------------------------------------------------


@dataclass
class Position:
    symbol: str                # e.g. BTCUSDT (spot)
    base: str
    quote: str
    quantity: float
    price: float
    value_usdt: float
    weight: float

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "base": self.base,
            "quote": self.quote,
            "quantity": self.quantity,
            "price": self.price,
            "value_usdt": round(self.value_usdt, 2),
            "weight": round(self.weight, 4),
        }


@dataclass
class AccountState:
    total_value_usdt: float          # spot value + futures equity
    cash_usdt: float                 # spot wallet cash
    positions: list[Position]        # spot positions
    peak_value_usdt: float
    spot_value_usdt: float = 0.0
    futures_wallet_usdt: float = 0.0  # free USDT in the futures wallet
    futures_positions: list[FuturesPosition] = field(default_factory=list)
    realized_pnl_usdt: float = 0.0
    trade_count_today: int = 0
    daily_window_start: Optional[datetime] = None
    last_trade_at: Optional[datetime] = None
    updated_at: datetime = field(default_factory=utcnow)

    @property
    def futures_equity_usdt(self) -> float:
        margin = sum(p.margin_usdt for p in self.futures_positions)
        unreal = sum(p.unrealized_pnl_usdt for p in self.futures_positions)
        return self.futures_wallet_usdt + margin + unreal

    def to_dict(self) -> dict:
        drawdown = (self.total_value_usdt / self.peak_value_usdt - 1.0) if self.peak_value_usdt else 0.0
        return {
            "total_value_usdt": round(self.total_value_usdt, 2),
            "cash_usdt": round(self.cash_usdt, 2),
            "spot_value_usdt": round(self.spot_value_usdt, 2),
            "futures_wallet_usdt": round(self.futures_wallet_usdt, 2),
            "futures_equity_usdt": round(self.futures_equity_usdt, 2),
            "positions": [p.to_dict() for p in sorted(self.positions, key=lambda p: -p.value_usdt)],
            "futures_positions": [p.to_dict() for p in self.futures_positions],
            "peak_value_usdt": round(self.peak_value_usdt, 2),
            "realized_pnl_usdt": round(self.realized_pnl_usdt, 2),
            "drawdown_pct": round(drawdown * 100, 2),
            "trade_count_today": self.trade_count_today,
            "daily_window_start": iso(self.daily_window_start),
            "last_trade_at": iso(self.last_trade_at),
            "updated_at": iso(self.updated_at),
        }


# ---------------------------------------------------------------------------
# Proposals
# ---------------------------------------------------------------------------


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class ProposalKind(str, Enum):
    TRADE = "TRADE"            # buy/sell (spot or futures)
    TRANSFER = "TRANSFER"      # move USDT between wallets (e.g. add margin)
    OPEN = "OPEN"              # open a new leveraged futures position
    CLOSE = "CLOSE"            # close a futures position (all or part)


class ProposalStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXECUTED = "EXECUTED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass
class TradeProposal:
    id: str
    symbol: str
    side: Side
    est_value_usdt: float      # notional value in quote currency
    est_quantity: float        # base-asset quantity (est.)
    est_price: float
    reason: str                # human-readable "why"
    kind: ProposalKind = ProposalKind.TRADE
    # used when kind == OPEN: margin & leverage for the new position
    open_margin_usdt: float = 0.0
    open_leverage: float = 0.0
    # used when kind == CLOSE: reduce the full position (True) or a slice
    close_all: bool = False
    keep_margin: bool = True   # CLOSE slice: deleverage semantics by default
    current_weight: float = 0.0
    target_weight: float = 0.0
    status: ProposalStatus = ProposalStatus.PENDING
    executed_value_usdt: Optional[float] = None
    executed_price: Optional[float] = None
    created_at: datetime = field(default_factory=utcnow)
    decided_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.value,
            "kind": self.kind.value,
            "open_margin_usdt": round(self.open_margin_usdt, 2) if self.open_margin_usdt else None,
            "open_leverage": self.open_leverage or None,
            "close_all": self.close_all,
            "est_value_usdt": round(self.est_value_usdt, 2),
            "est_quantity": round(self.est_quantity, 6),
            "est_price": round(self.est_price, 2),
            "reason": self.reason,
            "current_weight": round(self.current_weight, 4),
            "target_weight": round(self.target_weight, 4),
            "status": self.status.value,
            "executed_value_usdt": round(self.executed_value_usdt, 2) if self.executed_value_usdt else None,
            "executed_price": round(self.executed_price, 2) if self.executed_price else None,
            "created_at": iso(self.created_at),
            "decided_at": iso(self.decided_at),
        }


# ---------------------------------------------------------------------------
# Conditions (user-set triggers the agent watches)
# ---------------------------------------------------------------------------


class ConditionOp(str, Enum):
    ABOVE = "ABOVE"
    BELOW = "BELOW"


@dataclass
class Condition:
    id: str
    symbol: str
    op: ConditionOp
    price: float               # trigger price
    side: Side                 # what the agent proposes when triggered
    amount_usdt: float         # notional of the proposal
    note: str = ""
    active: bool = True
    position_id: Optional[str] = None   # when tied to an open position (TP/SL)
    close_all: bool = False
    created_at: datetime = field(default_factory=utcnow)
    last_fired_at: Optional[datetime] = None
    fires: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "op": self.op.value,
            "price": self.price,
            "side": self.side.value,
            "amount_usdt": self.amount_usdt,
            "note": self.note,
            "active": self.active,
            "position_id": self.position_id,
            "close_all": self.close_all,
            "created_at": iso(self.created_at),
            "last_fired_at": iso(self.last_fired_at),
            "fires": self.fires,
        }


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------


class AuditLevel(str, Enum):
    INFO = "INFO"
    WARN = "WARN"
    ACTION = "ACTION"
    GUARDRAIL = "GUARDRAIL"
    ERROR = "ERROR"


@dataclass
class AuditEntry:
    ts: datetime = field(default_factory=utcnow)
    level: AuditLevel = AuditLevel.INFO
    event: str = ""
    detail: str = ""
    proposal_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "ts": iso(self.ts),
            "level": self.level.value,
            "event": self.event,
            "detail": self.detail,
            "proposal_id": self.proposal_id,
        }
