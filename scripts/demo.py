#!/usr/bin/env python3
"""Scripted end-to-end demo of the Liquidation Guardian.

Runs the agent loop directly (no HTTP) and prints a clean walkthrough.
Mirrors exactly what the web UI does: guardrailed open with dynamic
leverage, TP/SL arming, the de-risk flow, and two-way fund movement
(futures -> spot — nothing is ever trapped).

Usage:
    python scripts/demo.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.adapters.sim import SimAdapter  # noqa: E402
from app.agent.orchestrator import Orchestrator  # noqa: E402
from app.config import load_config, load_guardrails  # noqa: E402
from app.market.client import MarketClient  # noqa: E402
from app.models import Ticker  # noqa: E402

BANNER = "─" * 64


class DemoMarket:
    """Deterministic fallback so the scripted walkthrough works offline."""

    prices = {
        "BTCUSDT": 80_000.0,
        "ETHUSDT": 2_500.0,
        "BNBUSDT": 770.0,
        "SOLUSDT": 105.0,
    }

    def _ticker(self, symbol: str) -> Ticker:
        return Ticker(symbol=symbol, price=self.prices[symbol], change_24h_pct=1.0)

    async def get_watch_tickers(self, symbols: list[str]) -> dict[str, Ticker]:
        return {symbol: self._ticker(symbol) for symbol in symbols}

    async def get_tickers(self, symbols: list[str]) -> dict[str, Ticker]:
        return await self.get_watch_tickers(symbols)

    async def get_ticker(self, symbol: str) -> Ticker:
        return self._ticker(symbol)

    async def get_price(self, symbol: str) -> float:
        return self.prices[symbol]

    async def analyze_symbol(self, symbol: str, bars: int = 168) -> dict:
        price = self.prices[symbol]
        return {
            "symbol": symbol,
            "base": symbol.replace("USDT", ""),
            "price": price,
            "change_24h_pct": 1.0,
            "change_7d_pct": 2.0,
            "high_24h": price * 1.02,
            "low_24h": price * 0.98,
            "high_7d": price * 1.05,
            "low_7d": price * 0.95,
            "rsi14": 55.0,
            "ma7": price * 1.01,
            "ma25": price * 0.99,
            "ma7_slope_pct": 0.1,
            "atr14_pct": 0.8,
            "support": price * 0.95,
            "support_scope": "7d",
            "resistance": price * 1.05,
            "resistance_scope": "7d",
            "regime": "neutral",
            "signal": "BULLISH",
        }

    async def aclose(self) -> None:
        return None


def step(title: str) -> None:
    print(f"\n{BANNER}\n▶ {title}\n{BANNER}")


async def main() -> None:
    cfg = load_config()
    guard = load_guardrails()
    market = MarketClient(cfg)
    sim = SimAdapter(cfg, guard, market, state_file="data/demo_state.json")
    sim.reset(10_000.0)
    orch = Orchestrator(cfg, guard, sim, market)

    print("LIQUIDATION GUARDIAN — scripted demo (SIM mode, live Binance prices)\n")

    step("1. Live watch + deep per-coin market analysis")
    tickers = await orch.watch_tickers()
    if not tickers:
        await market.aclose()
        market = DemoMarket()
        sim.market = market
        orch.market = market
        print("  (Binance public feed unavailable; using deterministic demo prices.)")
        tickers = await orch.watch_tickers()
    for sym, t in tickers.items():
        print(f"   {sym:<9} ${t['price']:,.2f}  ({t['change_24h_pct']:+.2f}% 24h)")
    r = await orch.market_analysis("BTCUSDT")
    print("\n" + r.message)

    step("2. Queue a guardrailed LONG — BTC at 20x with $600 margin (approve first)")
    r = await orch.queue_open("BTCUSDT", "LONG", 600.0, 20.0)
    print("  " + r.message)
    pid = r.proposals[0]["id"]
    print("  ✓ user approves…")
    r = orch.decide(pid, True)
    print("  " + r.message)

    step("3. Over-leverage is stopped by the guardrail (max 50x here)")
    r = await orch.queue_open("ETHUSDT", "LONG", 200.0, 99.0)
    print("  " + r.message)

    step("3b. Agent modes — manual queues, automatic executes after consent")
    denied = orch.set_agent_mode("auto", consent=False)
    print("  (enable auto without consent?) " + denied.message.splitlines()[0])
    granted = orch.set_agent_mode("auto", consent=True)
    print("  (with consent) " + granted.message.splitlines()[-1][:120])
    orch.set_agent_mode("manual", consent=False)
    print("  (back to manual) every action waits for approval again.")

    step("4. Arm a take-profit and stop-loss on the BTC position")
    pos = next(p for p in orch.account().futures_positions if p.symbol == "BTCUSDT")
    mark = pos.mark_price
    r = orch.set_tp_sl("BTCUSDT", round(mark * 1.05, 2), round(mark * 0.92, 2))
    print("  " + r.message)

    step("5. Risk report — the guardian tracks distance to liquidation")
    r = await orch.handle_message("risk report")
    print(r.message)

    step("6. “Protect my positions” — guardian proposes guardrailed de-risks")
    r = await orch.handle_message("protect my positions")
    print(r.message)
    for p in r.proposals:
        print(f"   [{p['kind']}] ≈ ${p['est_value_usdt']:,.2f} — {p['reason'][:90]}…")

    step("7. Approve the de-risk and check the fresh risk report")
    for p in r.proposals:
        if p["kind"] == "TRADE":
            resp = orch.decide(p["id"], True)
            print("  " + resp.message.split("\n")[0])
            break
    r = await orch.handle_message("risk report")
    print(r.message)

    step("8. Money flows back to spot — nothing is trapped")
    r = await orch.return_futures_wallet_to_spot()
    print("  " + r.message)
    for p in r.proposals:
        resp = orch.decide(p["id"], True)
        print("  " + resp.message.split("\\n")[0])
    for pos in orch.account().futures_positions:
        r2 = await orch.release_margin_to_spot(pos.symbol)
        print("  " + r2.message)

    step("9. Audit trail (transparency)")
    for a in orch.snapshot()["audit"][-10:]:
        print(
            f"  [{a['ts'][11:19]}] {a['level']:<9} {a['event']:<20} {a['detail'][:64]}"
        )

    print(
        f"\n{BANNER}\nDemo complete. Run the web UI for the interactive version:\n"
        "  uvicorn app.main:app --port 8000\n"
    )
    await market.aclose()


if __name__ == "__main__":
    asyncio.run(main())
