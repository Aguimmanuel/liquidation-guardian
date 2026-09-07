"""Execution adapter protocol.

Liquidation Guardian talks to Binance through a thin adapter so the exact same
agent code runs against a paper portfolio (``SimAdapter``, the default demo
path) or a real Binance Agentic sub-account (``MCPLiveAdapter``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.models import AccountState, Side, TradeProposal


@dataclass
class OrderResult:
    ok: bool
    proposal_id: str
    symbol: str
    side: Side
    executed_price: float
    executed_value_usdt: float
    message: str = ""
    fee_usdt: float = 0.0


class ExecutionAdapter(Protocol):
    mode: str

    def get_account(self) -> AccountState:
        """Current account state (positions, cash, stats)."""
        ...

    def execute(self, proposal: TradeProposal) -> OrderResult:
        """Execute an APPROVED proposal. Must be idempotent-safe by proposal id."""
        ...

    def close(self) -> None:
        ...
