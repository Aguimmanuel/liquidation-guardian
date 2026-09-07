"""FastAPI routes for the Liquidation Guardian."""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.agent.orchestrator import Orchestrator


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class DecideIn(BaseModel):
    approve: bool


class ConditionIn(BaseModel):
    symbol: str
    op: str = "BELOW"           # ABOVE | BELOW
    price: float
    side: str = "BUY"           # BUY | SELL
    amount_usdt: float = 500.0
    note: str = ""


class OpenIn(BaseModel):
    symbol: str
    side: str = "LONG"          # LONG | SHORT
    margin_usdt: float = Field(gt=0)
    leverage: float = Field(default=10.0, gt=0)


class CloseIn(BaseModel):
    symbol: str
    close_all: bool = True
    notional_usdt: float = 0.0


class TpSlIn(BaseModel):
    symbol: str
    take_profit: float = 0.0    # 0 = not armed / cleared
    stop_loss: float = 0.0      # 0 = not armed / cleared


class AgentModeIn(BaseModel):
    mode: str = "manual"        # "manual" | "auto"
    consent: bool = False       # required to enable automatic mode


class MarketAnalyzeIn(BaseModel):
    symbol: Optional[str] = None


class GuardrailsIn(BaseModel):
    values: dict[str, Any] = Field(default_factory=dict)


class ReturnFundsIn(BaseModel):
    scope: str = "wallet"       # "wallet" = free futures balance -> spot
                                # "position" = excess margin on an open position
    symbol: Optional[str] = None


def build_router(orch: Orchestrator) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/state")
    async def state() -> dict[str, Any]:
        return orch.snapshot()

    @router.post("/chat")
    async def chat(body: ChatIn) -> dict[str, Any]:
        resp = await orch.handle_message(body.message)
        return resp.to_dict()

    # ------------------------------------------------------------- TP / SL
    @router.post("/tpsl")
    async def set_tp_sl(body: TpSlIn) -> dict[str, Any]:
        resp = orch.set_tp_sl(body.symbol, body.take_profit, body.stop_loss)
        return resp.to_dict()

    @router.get("/tpsl/scan")
    async def scan_tp_sl() -> dict[str, Any]:
        """Manually sweep armed TP/SL levels (the server also does this on a
        timer while it runs)."""
        responses = await orch.scan_tp_sl()
        return {"fired": [r.to_dict() for r in responses], "count": len(responses)}

    # -------------------------------------------------------- agent mode
    @router.post("/agent/mode")
    async def set_agent_mode(body: AgentModeIn) -> dict[str, Any]:
        """Switch manual/automatic. Enabling automatic requires consent=True
        (the client shows the risk notice first)."""
        resp = orch.set_agent_mode(body.mode, body.consent)
        return resp.to_dict()

    # ------------------------------------------------------------- trading
    @router.post("/trade/open")
    async def open_position(body: OpenIn) -> dict[str, Any]:
        resp = await orch.queue_open(body.symbol, body.side, body.margin_usdt, body.leverage)
        return resp.to_dict()

    @router.post("/trade/close")
    async def close_position(body: CloseIn) -> dict[str, Any]:
        resp = await orch.queue_close(body.symbol, close_all=body.close_all, notional_usdt=body.notional_usdt)
        return resp.to_dict()

    # ---------------------------------------------------------- funds return
    @router.post("/return")
    async def return_funds(body: ReturnFundsIn) -> dict[str, Any]:
        """Move money out of the futures side back to spot: either the free
        futures-wallet balance, or the excess margin on one open position."""
        if body.scope == "position":
            if not body.symbol:
                raise HTTPException(status_code=400, detail="symbol is required for scope='position'")
            resp = await orch.release_margin_to_spot(body.symbol)
        else:
            resp = await orch.return_futures_wallet_to_spot()
        return resp.to_dict()

    # -------------------------------------------------------------- market
    @router.get("/market/watch")
    async def watch() -> dict[str, Any]:
        tickers = await orch.watch_tickers()
        return {"tickers": tickers, "symbols": orch.watch_symbols()}

    @router.post("/market/analyze")
    async def analyze(body: MarketAnalyzeIn) -> dict[str, Any]:
        resp = await orch.market_analysis(body.symbol)
        return resp.to_dict()

    # ------------------------------------------------------------- proposals
    @router.post("/proposals/{proposal_id}/decide")
    async def decide(proposal_id: str, body: DecideIn) -> dict[str, Any]:
        resp = orch.decide(proposal_id, body.approve)
        return resp.to_dict()

    @router.post("/proposals/approve-all")
    async def approve_all() -> dict[str, Any]:
        responses = orch.approve_all()
        return {"approved": len(responses), "responses": [r.to_dict() for r in responses]}

    # ------------------------------------------------------------ conditions
    @router.post("/conditions")
    async def add_condition(body: ConditionIn) -> dict[str, Any]:
        cond = {
            "symbol": body.symbol,
            "op": body.op,
            "price": body.price,
            "side": body.side,
            "amount_usdt": body.amount_usdt,
            "note": body.note,
        }
        resp = await orch._do_add_condition(cond)
        return resp.to_dict()

    @router.api_route("/conditions/check", methods=["GET", "POST"])
    async def check_conditions() -> dict[str, Any]:
        """Evaluate armed conditions (also swept automatically server-side)."""
        responses = await orch.check_conditions()
        return {"fired": [r.to_dict() for r in responses], "count": len(responses)}

    @router.delete("/conditions/{condition_id}")
    async def delete_condition(condition_id: str) -> dict[str, Any]:
        orch.conditions.pop(condition_id, None)
        return {"ok": True, "state": orch.snapshot()}

    # ------------------------------------------------------------ guardrails
    @router.get("/guardrails")
    async def guardrails() -> dict[str, Any]:
        return orch.guardrail_status()

    @router.post("/guardrails")
    async def update_guardrails(body: GuardrailsIn) -> dict[str, Any]:
        resp = orch.update_guardrails(body.values)
        return resp.to_dict()

    # ---------------------------------------------------------------- admin
    @router.post("/reset")
    async def reset() -> dict[str, Any]:
        resp = orch.reset()
        return resp.to_dict()

    return router


def serve_index() -> FileResponse:
    import os

    path = os.path.join(os.path.dirname(__file__), "..", "static", "index.html")
    return FileResponse(os.path.abspath(path))
