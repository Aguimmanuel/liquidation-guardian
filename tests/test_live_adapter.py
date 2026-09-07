"""Live (Binance MCP) adapter guards.

B3: requirements pin ``mcp>=2.2,<3`` — the high-level ``mcp.client.Client``
used by the adapter only exists on the 2.x SDK (and the ``[client]`` extra
does not exist at all). The import smoke below locks that surface.
B2: the adapter must dispatch on proposal kind and refuse anything that is
not a real order instead of mangling it into a zero-quantity market order.
"""

import uuid

import pytest

from app.adapters.mcp_live import MCPLiveAdapter
from app.agent.guardrails import GuardrailConfig
from app.config import AppConfig
from app.models import (
    ProposalKind,
    Side,
    TradeProposal,
    TransferKind,
)


def _transfer_proposal(**kw) -> TradeProposal:
    fields = dict(
        id=uuid.uuid4().hex[:10],
        symbol="BTCUSDT",
        side=Side.BUY,
        est_value_usdt=200.0,
        est_quantity=0.0,
        est_price=0.0,
        reason="add margin",
        kind=ProposalKind.TRANSFER,
        transfer_kind=TransferKind.ADD_MARGIN,
    )
    fields.update(kw)
    return TradeProposal(**fields)


def _adapter() -> MCPLiveAdapter:
    """Adapter with a live session object but NO network: execute() must never
    reach the session for the paths these tests exercise."""
    adapter = MCPLiveAdapter(
        config=AppConfig(),
        guardrails=GuardrailConfig(),
        market=None,  # type: ignore[arg-type]  # not touched by these paths
    )
    # prove the test would catch a network attempt
    def _explode(*_a, **_k):
        raise AssertionError("execute() must not touch the network here")

    adapter.session.connect = _explode  # type: ignore[method-assign]
    adapter.session.find_tool = _explode  # type: ignore[method-assign]
    return adapter


def test_live_module_and_mcp_sdk_import():
    """B3: the SDK surface the adapter imports exists on the pinned version."""
    from mcp.client import Client  # noqa: F401
    from mcp.client.streamable_http import streamable_http_client  # noqa: F401

    import app.adapters.mcp_live as mod  # noqa: F401

    assert Client is not None


def test_live_transfer_refused_without_touching_network():
    """B2: free-balance transfers are not orders — refuse them loudly instead
    of sending a zero-quantity market order."""
    adapter = _adapter()
    result = adapter.execute(_transfer_proposal())
    assert not result.ok
    assert "margin" in result.message.lower() or "fund" in result.message.lower()
    assert "cannot" in result.message.lower()


def test_live_open_requires_quantity():
    """B2: an OPEN with no computable quantity is refused before the adapter
    ever connects — never sent as a zero-quantity order."""
    adapter = _adapter()
    prop = _transfer_proposal(
        kind=ProposalKind.OPEN,
        transfer_kind=None,
        est_value_usdt=5000.0,
        est_quantity=0.0,
        est_price=1000.0,
        open_margin_usdt=500.0,
        open_leverage=10.0,
    )
    result = adapter.execute(prop)
    assert not result.ok
    assert "no quantity" in result.message.lower()
