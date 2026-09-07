"""Liquidation Guardian — app entry point.

Run:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
or:
    python -m app.main
"""
from __future__ import annotations

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Make the repo root importable when run as `python app/main.py`
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.chdir(ROOT)  # so relative state_file paths resolve from the repo root

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.adapters.mcp_live import MCPLiveAdapter
from app.adapters.sim import SimAdapter
from app.agent.orchestrator import Orchestrator
from app.api.routes import build_router, serve_index
from app.config import load_config, load_guardrails
from app.market.client import MarketClient


def create_app() -> FastAPI:
    cfg = load_config()
    guard = load_guardrails()
    market = MarketClient(cfg)

    if cfg.mode == "live":
        adapter = MCPLiveAdapter(cfg, guard, market)
    else:
        adapter = SimAdapter(cfg, guard, market, state_file=str(ROOT / cfg.state_file))

    orch = Orchestrator(cfg, guard, adapter, market)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        async def autopilot() -> None:
            """Background guard sweep: fires armed TP/SL levels (auto-close, the
            user pre-authorized the exit when they armed it) and evaluates armed
            price conditions (which queue proposals for approval)."""
            while True:
                try:
                    await asyncio.sleep(2.0)
                    if orch.conditions:
                        await orch.check_conditions()
                    await orch.scan_tp_sl()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    pass  # a single bad sweep must never kill the guardian

        task = asyncio.create_task(autopilot())
        try:
            await orch.refresh()
        except Exception:
            pass  # never crash the app on a market-data hiccup
        yield
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        try:
            await market.aclose()
        except Exception:
            pass

    app = FastAPI(title="Liquidation Guardian", version="0.3.0", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_request: Request, exc: RequestValidationError):
        """422s become readable messages, not opaque generic errors."""
        bits = []
        for err in exc.errors():
            where = ".".join(str(x) for x in err.get("loc", []) if x != "body")
            bits.append(f"{where}: {err.get('msg', 'invalid value')}")
        return JSONResponse(
            status_code=422,
            content={"error": "validation", "message": "Invalid request — " + "; ".join(bits)},
        )

    @app.exception_handler(Exception)
    async def unhandled_handler(_request: Request, exc: Exception):
        """500s name the real problem instead of a bare 'Internal Server Error'."""
        return JSONResponse(
            status_code=500,
            content={"error": "server", "message": f"Server error: {exc}"},
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", include_in_schema=False)
    async def index():
        return serve_index()

    app.include_router(build_router(orch))
    app.state.orchestrator = orch  # expose for tests / scripts
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    cfg = load_config()
    uvicorn.run("app.main:app", host=cfg.host, port=cfg.port, reload=False)
