"""Live adapter: executes against a real Binance Agentic sub-account via the
Binance MCP Server (https://agent.binance.com/mcp/agentic).

Security model (enforced by Binance, not by us):
- the agent runs inside a dedicated *Agentic sub-account* you fund manually;
- there is NO withdrawal scope — the agent can never move funds to an external
  address;
- every non-read action goes through Binance's confirm-before-execute flow.

This adapter uses the official MCP Python SDK's high-level Client. OAuth: when
``RS_MCP_ACCESS_TOKEN`` is set, it is sent as a Bearer token; otherwise the
client attempts OAuth resource-server discovery, which typically opens a
Binance consent page in your browser on first use.

To use it: fund an Agentic sub-account, complete the OAuth consent for your
account, then run with ``RS_MODE=live``. ``SimAdapter`` runs the identical
agent loop against a paper portfolio, so behaviour matches.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from app.adapters.base import OrderResult
from app.agent.guardrails import GuardrailConfig
from app.config import AppConfig
from app.market.client import MarketClient
from app.models import (
    AccountState,
    FuturesPosition,
    Position,
    PositionSide,
    Side,
    TradeProposal,
)

logger = logging.getLogger("liquidation_guardian.live")

# ---------------------------------------------------------------------------
# Tool-name mapping. The Binance MCP server exposes tools for market data,
# account, trade and transfer scopes; names follow the documented surface.
# We discover tools at runtime and match on these prefixes/aliases so small
# naming differences don't break execution.
# ---------------------------------------------------------------------------

TOOL_HINTS = {
    "get_tickers": ["ticker", "price", "market"],
    "get_account": ["account", "balance", "portfolio"],
    "place_order": ["order", "trade", "place", "new_order"],
}


class MCPSession:
    """Thin wrapper around the MCP SDK high-level client with lazy connect."""

    def __init__(self, url: str, access_token: str = ""):
        self.url = url
        self.access_token = access_token
        self._client: Any = None
        self._tools: dict[str, Any] = {}
        self._tools_loaded = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()

    def run_sync(self, coro):  # noqa: ANN001
        """Run MCP coroutines on one persistent loop, including from FastAPI."""
        if self._loop is None or self._loop.is_closed():
            self._ready.clear()

            def runner() -> None:
                loop = asyncio.new_event_loop()
                self._loop = loop
                asyncio.set_event_loop(loop)
                self._ready.set()
                loop.run_forever()
                loop.close()

            self._thread = threading.Thread(target=runner, name="mcp-loop", daemon=True)
            self._thread.start()
            self._ready.wait(timeout=5)
        if self._loop is None:
            raise RuntimeError("could not start the MCP event loop")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=30)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise TimeoutError("Binance MCP request timed out") from exc

    async def connect(self) -> None:
        if self._client is not None:
            return
        from mcp.client import Client
        from mcp.client.streamable_http import streamable_http_client

        if self.access_token:
            # Inject the Bearer token via a custom transport-level client.
            import httpx

            http = httpx.AsyncClient(
                timeout=15.0,
                headers={"Authorization": f"Bearer {self.access_token}"},
            )
            transport = streamable_http_client(self.url, http_client=http)
            self._client = Client(transport)
        else:
            # OAuth discovery + consent flow (browser) where supported.
            self._client = Client(self.url)
        await self._client.__aenter__()
        await self._load_tools()

    async def _load_tools(self) -> None:
        assert self._client is not None
        tools = await self._client.list_tools()
        self._tools = {t.name: t for t in tools}
        self._tools_loaded = True
        logger.info("Binance MCP tools available: %s", sorted(self._tools))

    def find_tool(self, hint_key: str) -> Optional[str]:
        hints = TOOL_HINTS[hint_key]
        for name in self._tools:
            low = name.lower()
            if any(h in low for h in hints):
                return name
        return None

    async def call(self, tool_name: str, args: dict[str, Any]) -> Any:
        assert self._client is not None
        result = await self._client.call_tool(tool_name, args)
        # result.content: list of text/image blocks
        text = ""
        for block in result.content:
            if getattr(block, "type", "") == "text":
                text += block.text or ""
        return {
            "text": text,
            "structured": getattr(result, "structuredContent", None),
            "error": bool(
                getattr(result, "isError", False) or getattr(result, "is_error", False)
            ),
        }

    async def close(self) -> None:
        if self._client is not None:
            await self._client.__aexit__(None, None, None)
            self._client = None
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if (
            self._thread is not None
            and self._thread.is_alive()
            and threading.current_thread() is not self._thread
        ):
            self._thread.join(timeout=5)
        self._loop = None
        self._thread = None


class MCPLiveAdapter:
    """Execution adapter backed by the Binance MCP server.

    ``get_account``/``execute`` are synchronous in the adapter protocol, so we
    bridge the async MCP calls with a tiny event-loop shim.
    """

    mode = "live"

    def __init__(
        self,
        config: AppConfig,
        guardrails: GuardrailConfig,
        market: MarketClient,
    ):
        self.config = config
        self.guardrails = guardrails
        self.market = market
        self.session = MCPSession(
            url=guardrails.mcp_url,
            access_token=guardrails.mcp_access_token,
        )

    # -- async plumbing ------------------------------------------------------
    def _run(self, coro):  # noqa: ANN001
        return self.session.run_sync(coro)

    async def connect_async(self) -> None:
        await asyncio.to_thread(self.session.run_sync, self.session.connect())

    # -- adapter protocol ----------------------------------------------------
    def get_account(self) -> AccountState:
        self._run(self.session.connect())
        tool = self.session.find_tool("get_account")
        if tool is None:
            raise RuntimeError(
                "no account/balance tool found on Binance MCP server — "
                "grant the 'Account' scope and reconnect"
            )
        res = self._run(self.session.call(tool, {}))
        return self._parse_account(res)

    def execute(self, proposal: TradeProposal) -> OrderResult:
        self._run(self.session.connect())
        tool = self.session.find_tool("place_order")
        if tool is None:
            return OrderResult(
                ok=False,
                proposal_id=proposal.id,
                symbol=proposal.symbol,
                side=proposal.side,
                executed_price=0.0,
                executed_value_usdt=0.0,
                message="no order tool found — grant the 'Trade' scope and reconnect",
            )
        symbol, qty = proposal.symbol, proposal.est_quantity
        res = self._run(
            self.session.call(
                tool,
                {
                    "symbol": symbol,
                    "side": proposal.side.value.lower(),
                    "type": "MARKET",
                    "quantity": str(qty),
                },
            )
        )
        structured = res.get("structured") or {}
        if res.get("error"):
            return OrderResult(
                ok=False,
                proposal_id=proposal.id,
                symbol=symbol,
                side=proposal.side,
                executed_price=0.0,
                executed_value_usdt=0.0,
                message=res["text"][:500] or "Binance MCP rejected the order",
            )
        executed_price = self._number(structured, "executedPrice", "avgPrice", "price")
        return OrderResult(
            ok=True,
            proposal_id=proposal.id,
            symbol=symbol,
            side=proposal.side,
            executed_price=executed_price or proposal.est_price,
            executed_value_usdt=proposal.est_value_usdt,
            message=res["text"][:500],
        )

    def close(self) -> None:
        try:
            self._run(self.session.close())
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("error closing MCP session: %s", exc)

    # -- helpers ---------------------------------------------------------------
    def _parse_account(self, res: dict[str, Any]) -> AccountState:
        payload = res.get("structured") or self._decode_text(res.get("text", ""))
        if not isinstance(payload, dict):
            raise RuntimeError("Binance MCP account response was not structured JSON")

        account = (
            payload.get("account")
            if isinstance(payload.get("account"), dict)
            else payload
        )
        raw_positions = (
            account.get("positions") or account.get("futuresPositions") or []
        )
        if isinstance(raw_positions, dict):
            raw_positions = (
                raw_positions.get("items") or raw_positions.get("data") or []
            )

        futures: list[FuturesPosition] = []
        for raw in raw_positions:
            if not isinstance(raw, dict):
                continue
            symbol = str(raw.get("symbol") or raw.get("asset", "")).upper()
            qty = self._number(raw, "positionAmt", "quantity", "qty", "amount")
            if not symbol or not qty:
                continue
            side_raw = str(raw.get("positionSide") or raw.get("side", "")).upper()
            side = (
                PositionSide.SHORT
                if side_raw == "SHORT" or qty < 0
                else PositionSide.LONG
            )
            qty = abs(qty)
            mark = self._number(raw, "markPrice", "mark", "price")
            entry = self._number(raw, "entryPrice", "avgEntryPrice", "averagePrice")
            margin = self._number(raw, "isolatedMargin", "initialMargin", "margin")
            leverage = self._number(raw, "leverage") or 1.0
            futures.append(
                FuturesPosition(
                    symbol=symbol,
                    side=side,
                    entry_price=entry,
                    quantity=qty,
                    leverage=leverage,
                    margin_usdt=margin,
                    mark_price=mark or entry,
                    liq_price=self._number(raw, "liquidationPrice", "liqPrice"),
                    unrealized_pnl_usdt=self._number(
                        raw, "unRealizedProfit", "unrealizedProfit", "unrealizedPnl"
                    ),
                )
            )

        usdt = self._find_asset(account, "USDT")
        cash = self._number(usdt, "free", "available", "availableBalance", "balance")
        wallet = self._number(
            account, "futuresWalletBalance", "walletBalance", "availableBalance"
        )
        unreal = sum(p.unrealized_pnl_usdt for p in futures)
        margin = sum(p.margin_usdt for p in futures)
        total = (
            self._number(
                account, "totalWalletBalance", "totalEquity", "equity", "total"
            )
            or cash + wallet + margin + unreal
        )
        now = datetime.now(timezone.utc)
        return AccountState(
            total_value_usdt=total,
            cash_usdt=cash,
            positions=[],
            spot_value_usdt=cash,
            futures_wallet_usdt=wallet,
            futures_positions=futures,
            peak_value_usdt=total,
            daily_window_start=now,
            updated_at=now,
        )

    @staticmethod
    def _decode_text(text: str) -> Any:
        try:
            return json.loads(text)
        except (TypeError, json.JSONDecodeError):
            return None

    @staticmethod
    def _number(data: Any, *keys: str) -> float:
        if not isinstance(data, dict):
            return 0.0
        for key in keys:
            value = data.get(key)
            if value not in (None, ""):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return 0.0

    @staticmethod
    def _find_asset(account: dict[str, Any], symbol: str) -> dict[str, Any]:
        assets = account.get("assets") or account.get("balances") or []
        if isinstance(assets, dict):
            assets = assets.get("items") or assets.get("data") or []
        for asset in assets:
            if (
                isinstance(asset, dict)
                and str(asset.get("asset") or asset.get("symbol", "")).upper() == symbol
            ):
                return asset
        return {}
