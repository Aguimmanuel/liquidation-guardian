#!/usr/bin/env python3
"""Backtest the Liquidation Guardian on real historical Binance data.

Picks the most volatile window in the requested period, simulates a leveraged
long opened at the window start, and runs the guardian hourly: whenever the
distance to liquidation drops below the danger threshold, it proposes a cut
back to the target headroom (executed at that hour's close with futures fees).

Compares three outcomes: no guardian (hold to liquidation or end), guardian
active, and an "add margin" variant.

Usage:
    python scripts/backtest.py [--days 90] [--symbol BTCUSDT] [--leverage 20]
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent.risk import liq_price, required_cut, required_margin  # noqa: E402
from app.config import load_config, load_guardrails  # noqa: E402
from app.market.client import MarketClient  # noqa: E402
from app.models import FuturesPosition, PositionSide  # noqa: E402

MMR = 0.005          # maintenance margin rate
FEE = 0.0004         # USDⓈ-M taker fee
DANGER = 0.06        # distance to liq that triggers the guardian
TARGET = 0.15        # headroom the guardian restores


class PositionSim:
    """Minimal isolated-margin long simulator (no funding, no slippage).
    Tracks a wallet so capital that exits the position (cuts) or enters it
    (margin injections) is accounted for fairly."""

    def __init__(self, entry_price: float, margin: float, leverage: float, symbol: str):
        self.symbol = symbol
        self.entry = entry_price
        self.margin = margin
        self.leverage = leverage
        self.quantity = (margin * leverage) / entry_price
        self.wallet = 0.0            # free USDT in the futures wallet
        self.injected = 0.0          # total margin injected over the run
        self.closes = 0
        self.liquidated = False
        self.side = PositionSide.LONG
        self.mmr = MMR
        self.mark_price = entry_price

    def as_pos(self) -> FuturesPosition:
        return FuturesPosition(
            symbol=self.symbol, side=self.side, entry_price=self.entry,
            quantity=self.quantity, leverage=self.leverage, margin_usdt=self.margin,
            mmr=self.mmr, mark_price=self.mark_price,
        )

    @property
    def liq(self) -> float:
        return liq_price(PositionSide.LONG, self.entry, self.quantity, self.margin, MMR)

    def reduce(self, price: float, qty: float) -> None:
        """Deleverage: close qty, keep the margin on the remainder (only the
        realized PnL goes to the wallet), so the liq price actually moves."""
        if qty <= 0 or qty > self.quantity:
            return
        qty_before = self.quantity
        pnl = (price - self.entry) * qty
        self.wallet += pnl * (1 - FEE)
        self.quantity = qty_before - qty
        self.margin = self.margin  # margin stays (released margin re-added)
        self.closes += 1

    def add_margin(self, amount: float) -> None:
        self.margin += amount
        self.injected += amount

    def total(self, price: float) -> float:
        """Wallet + margin + unrealized PnL (0 if liquidated)."""
        if self.liquidated:
            return self.wallet
        return self.wallet + self.margin + (price - self.entry) * self.quantity


def run_window(prices: list[float], lows: list[float], margin: float, leverage: float,
               symbol: str, guardian: bool, add_margin: bool = False) -> dict:
    entry = prices[0]
    sim = PositionSim(entry, margin, leverage, symbol)
    # act on the previous close's distance, then let the current candle test us
    for i in range(1, len(prices)):
        if sim.liquidated:
            break
        prev_close = prices[i - 1]
        dist = (prev_close - sim.liq) / prev_close if prev_close > 0 else 0.0
        if guardian and dist < DANGER:
            if add_margin:
                need = required_margin(sim.as_pos(), prev_close, TARGET)
                if need > 0:
                    sim.add_margin(need)
            else:
                cut = required_cut(sim.as_pos(), prev_close, TARGET)
                if cut > 0:
                    sim.reduce(prev_close, cut)
        # liquidation triggers intraday if this candle's low crosses the liq price
        if lows[i] <= sim.liq:
            sim.liquidated = True
            break
    end_price = prices[-1]
    return {
        "liquidated": sim.liquidated,
        "total": sim.total(end_price),
        "injected": sim.injected,
        "closes": sim.closes,
        "liq_after": sim.liq,
    }


async def run(days: int, symbol: str, leverage: float, margin: float = 1000.0) -> None:
    cfg = load_config()
    guard = load_guardrails()
    DANGER = guard.liq_danger_pct
    TARGET = guard.liq_target_dist_pct
    market = MarketClient(cfg)
    print(f"Fetching {days} days of hourly {symbol} klines …")
    klines = await market.get_klines(symbol, interval="1h", limit=days * 24)
    await market.aclose()
    if not klines:
        print("No data — check network access to the market feed.")
        return

    closes = [k["close"] for k in klines]
    lows = [k["low"] for k in klines]

    # find the worst *forward-looking* crash: the start price vs the lowest
    # price that follows it within the window. This models "you opened a long
    # here, then the market fell" — the scenario the guardian exists for.
    window = 5 * 24
    best_start, best_dd = 0, 0.0
    for start in range(0, len(closes) - window, 24):
        seg = closes[start : start + window]
        trough = min(seg)
        dd = (trough - closes[start]) / closes[start]
        if dd < best_dd:
            best_dd, best_start = dd, start

    seg_closes = closes[best_start : best_start + window]
    seg_lows = lows[best_start : best_start + window]
    from datetime import datetime, timezone

    start_dt = datetime.fromtimestamp(klines[best_start]["ts"] / 1000, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(klines[best_start + window - 1]["ts"] / 1000, tz=timezone.utc)

    no_guard = run_window(seg_closes, seg_lows, margin, leverage, symbol, guardian=False)
    guard_cut = run_window(seg_closes, seg_lows, margin, leverage, symbol, guardian=True)
    guard_margin = run_window(seg_closes, seg_lows, margin, leverage, symbol, guardian=True, add_margin=True)

    invested = margin
    print("\n" + "=" * 66)
    print(f"Liquidation Guardian backtest · {symbol} · {leverage:.0f}x · ${margin:,.0f} margin")
    print(f"Window: {start_dt:%Y-%m-%d %H:%M} → {end_dt:%Y-%m-%d %H:%M} UTC "
          f"({len(seg_closes)} hours, {best_dd:.1%} peak-to-trough)")
    print("=" * 66)
    rows = (
        ("No guardian (hold)     ", no_guard),
        ("Guardian: cut positions", guard_cut),
        ("Guardian: add margin   ", guard_margin),
    )
    for label, r in rows:
        status = "💀 LIQUIDATED" if r["liquidated"] else "survived"
        net = r["total"] - invested - r["injected"]
        detail = ""
        if r["closes"]:
            detail += f"  · {r['closes']} cut(s)"
        if r["injected"] > 0:
            detail += f"  · injected ${r['injected']:,.0f}"
        print(f"  {label}  ${r['total']:>9,.2f}  (net {net:+,.0f})  {status}{detail}")
    print("=" * 66)
    print(f"\nSim: isolated long, 0.04% taker fee, hourly scans at close prices, "
          f"trigger at <{DANGER:.0%} to liq, de-risk back to {TARGET:.0%} headroom.")
    print("Illustrative scenario, not investment advice.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--leverage", type=float, default=20)
    parser.add_argument("--margin", type=float, default=1000)
    args = parser.parse_args()
    asyncio.run(run(args.days, args.symbol.upper(), args.leverage, args.margin))


if __name__ == "__main__":
    main()
