"""Liquidation Guardian — app entry point.

Run:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
or:
    python -m app.main
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

logger = logging.getLogger("liquidation_guardian.api")

# Make the repo root importable when run as `python app/main.py`
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def _auth_enabled() -> str:
    """Bearer token that gates state-changing /api calls, or "" for none.

    Opt-in via ``RS_API_TOKEN``. When unset the app stays open (the default
    demo posture); when set, every /api request except harmless GETs must
    carry it as ``Authorization: Bearer <token>`` or ``X-API-Token``.
    """
    return (os.environ.get("RS_API_TOKEN") or "").strip()


def _cors_origins() -> list[str]:
    """Comma-separated ``RS_CORS_ORIGINS`` override; default = same-origin only."""
    raw = os.environ.get("RS_CORS_ORIGINS", "").strip()
    if not raw:
        return []
    return [o.strip() for o in raw.split(",") if o.strip()]


def create_app() -> FastAPI:
    cfg = load_config()
    guard = load_guardrails()
    market = MarketClient(cfg)

    # B15: chdir lives here (idempotent) so importing app.main as a library
    # never changes the caller's working directory; relative state_file paths
    # resolve against the repo root for the CLI/server entry points.
    os.chdir(ROOT)  # noqa: PTH109 - deliberate repo-root baseline

    if cfg.mode == "live":
        adapter = MCPLiveAdapter(cfg, guard, market)
    else:
        adapter = SimAdapter(cfg, guard, market, state_file=str(ROOT / cfg.state_file))

    orch = Orchestrator(cfg, guard, adapter, market)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        async def autopilot() -> None:
            """Background guardian loop: sweeps TP/SL levels, evaluates armed
            price conditions, and monitors every open position around the clock
            — logging zone changes and, on DANGER, escalating (queues a plan in
            manual mode, executes it in automatic mode). All autonomous actions
            respect the guardrails; a single bad sweep never kills the loop."""
            while True:
                try:
                    await asyncio.sleep(2.0)
                    # Protective exits first, then the guardian monitor, then
                    # price conditions last — so a condition can see any plan
                    # the monitor just queued for the same symbol and defer
                    # (#1 mutual exclusion) instead of stacking a second action.
                    await orch.scan_tp_sl()
                    await orch.monitor_positions()
                    if orch.conditions:
                        await orch.check_conditions()
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
    async def unhandled_handler(request: Request, exc: Exception):
        """500s reply generically to the client; the real exception goes to the
        server log where it belongs (never echoed raw to the browser)."""
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"error": "server", "message": "Internal server error"},
        )

    # B4: CORS posture — same-origin only by default; open it up only when the
    # operator explicitly lists allowed origins via RS_CORS_ORIGINS.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    api_token = _auth_enabled()

    @app.middleware("http")
    async def api_token_gate(request: Request, call_next):
        """Optional token gate (B4): when RS_API_TOKEN is set, state-changing
        /api calls must present it (Bearer or X-API-Token). Read-only GETs and
        the static UI are exempt so the browser app and simple probes keep
        working; the gate is the tripwire for programmatic/live deployments.
        """
        if api_token and request.url.path.startswith("/api") and request.method != "OPTIONS":
            if request.method not in ("GET",):
                supplied = request.headers.get("authorization", "")
                if supplied.lower().startswith("bearer "):
                    supplied = supplied[7:].strip()
                else:
                    supplied = request.headers.get("x-api-token", "")
                if supplied != api_token:
                    return JSONResponse(
                        status_code=401,
                        content={
                            "error": "unauthorized",
                            "message": "Missing or invalid API token "
                            "(set RS_API_TOKEN on the server and send "
                            "Authorization: Bearer <token>).",
                        },
                    )
        return await call_next(request)

    @app.middleware("http")
    async def no_store(request: Request, call_next):
        """Never let a browser serve a stale page/state: the UI is rebuilt and
        the sim state changes constantly, so everything ships uncached."""
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    @app.get("/", include_in_schema=False)
    async def index():
        return serve_index()

    if cfg.mode == "live" and not api_token:
        logger.warning(
            "running LIVE without RS_API_TOKEN — the API is open to anyone who "
            "can reach it; set the token for any non-local deployment"
        )

    app.include_router(build_router(orch))
    app.state.orchestrator = orch  # expose for tests / scripts
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    cfg = load_config()
    uvicorn.run("app.main:app", host=cfg.host, port=cfg.port, reload=False)
