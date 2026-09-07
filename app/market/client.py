"""Binance public market data client.

Uses the official public market-data endpoint (``data-api.binance.vision``),
which needs no API key and is not geo-restricted — the same feed Binance's own
docs point developers to. A tiny TTL cache keeps us polite and fast.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Optional

import httpx

from app.config import AppConfig
from app.models import Ticker


class MarketDataError(Exception):
    pass


def _change_pct_over(closes: list[float], hours: int = 24) -> float:
    """Percent change over the trailing ``hours`` of 1h closes. The reference
    is the close from the same wall-clock hour ``hours`` ago (``closes[-h-1]``)
    — the standard 24h-change definition on hourly bars, used consistently by
    every analysis path (B9)."""
    if len(closes) <= hours:
        return 0.0
    ref = closes[-(hours + 1)]
    return (closes[-1] / ref - 1.0) * 100.0 if ref else 0.0


def _wilder_rsi(closes: list[float], period: int = 14) -> float:
    """Wilder's RSI(``period``) on a close series.

    First average = simple mean of the first ``period`` gains/losses, then
    Wilder smoothing: ``avg = (prev * (period - 1) + cur) / period``. Returns
    100.0 on ``period``+ straight up-bars and 0.0 on straight down-bars —
    never a ZeroDivisionError (B7/B8).
    """
    if len(closes) <= period:
        return 50.0
    gains = losses = 0.0
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        if d >= 0:
            gains += d
        else:
            losses -= d
    avg_g, avg_l = gains / period, losses / period
    for i in range(period + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        up, dn = (d, 0.0) if d >= 0 else (0.0, -d)
        avg_g = (avg_g * (period - 1) + up) / period
        avg_l = (avg_l * (period - 1) + dn) / period
    if avg_g == 0.0 and avg_l == 0.0:
        return 50.0
    if avg_l == 0.0:
        return 100.0
    if avg_g == 0.0:
        return 0.0
    rs = avg_g / avg_l
    return 100.0 - 100.0 / (1.0 + rs)


class MarketClient:
    def __init__(self, config: AppConfig, http_client: Optional[httpx.AsyncClient] = None):
        self.base = config.market_base_url.rstrip("/")
        self.ttl = config.market_ttl_seconds
        self.ticker_ttl = config.ticker_ttl_seconds
        self._http = http_client or httpx.AsyncClient(timeout=10.0)
        self._cache: dict[str, tuple[float, Ticker]] = {}
        self._watch_cache: dict[str, tuple[float, Ticker]] = {}

    async def get_watch_tickers(self, symbols: list[str]) -> dict[str, Ticker]:
        """Tickers refreshed at the fast watch interval (for the live strip)."""
        now = time.monotonic()
        missing = [
            s for s in symbols
            if self._watch_cache.get(s, (0, None))[0] + self.ticker_ttl < now
        ]
        if missing:
            try:
                payload = await self._get_json(
                    "/api/v3/ticker/24hr", params={"symbols": json.dumps(missing)}
                )
                for item in payload:
                    if isinstance(item, dict) and "symbol" in item:
                        t = Ticker(
                            symbol=item["symbol"],
                            price=float(item["lastPrice"]),
                            change_24h_pct=float(item["priceChangePercent"]),
                        )
                        self._watch_cache[item["symbol"]] = (time.monotonic(), t)
            except MarketDataError:
                # on a hiccup, fall back to whatever we already have / single fetch
                for sym in missing:
                    try:
                        t = await self.get_ticker(sym)
                        self._watch_cache[sym] = (time.monotonic(), t)
                    except MarketDataError:
                        continue
        out = {}
        for s in symbols:
            hit = self._watch_cache.get(s)
            if hit:
                out[s] = hit[1]
        return out

    # -- lifecycle ---------------------------------------------------------
    async def aclose(self) -> None:
        await self._http.aclose()

    # -- tickers ------------------------------------------------------------
    async def get_ticker(self, symbol: str) -> Ticker:
        cached = self._cache.get(symbol)
        if cached and time.monotonic() - cached[0] < self.ttl:
            return cached[1]
        data = await self._get_json("/api/v3/ticker/24hr", params={"symbol": symbol})
        ticker = Ticker(
            symbol=data["symbol"],
            price=float(data["lastPrice"]),
            change_24h_pct=float(data["priceChangePercent"]),
        )
        self._cache[symbol] = (time.monotonic(), ticker)
        return ticker

    async def get_tickers(self, symbols: list[str]) -> dict[str, Ticker]:
        """Batch fetch tickers for a list of symbols (24h window), cached."""
        missing = [s for s in symbols if self._cache.get(s, (0, None))[0] + self.ttl < time.monotonic()]
        if missing:
            try:
                payload = await self._get_json(
                    "/api/v3/ticker/24hr", params={"symbols": json.dumps(missing)}
                )
                for item in payload:
                    if isinstance(item, dict) and "symbol" in item:
                        t = Ticker(
                            symbol=item["symbol"],
                            price=float(item["lastPrice"]),
                            change_24h_pct=float(item["priceChangePercent"]),
                        )
                        self._cache[item["symbol"]] = (time.monotonic(), t)
            except MarketDataError:
                # fall back to per-symbol fetches for the missing ones
                for sym in missing:
                    try:
                        t = await self.get_ticker(sym)
                        self._cache[sym] = (time.monotonic(), t)
                    except MarketDataError:
                        continue
        out = {}
        for s in symbols:
            hit = self._cache.get(s)
            if hit:
                out[s] = hit[1]
        return out

    async def get_price(self, symbol: str) -> float:
        return (await self.get_ticker(symbol)).price

    # -- klines (for backtests) ---------------------------------------------
    async def get_klines(
        self, symbol: str, interval: str = "1h", limit: int = 500
    ) -> list[dict]:
        """Return list of {ts, open, high, low, close} for a symbol."""
        data = await self._get_json(
            "/api/v3/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
        )
        out = []
        for k in data:
            out.append(
                {
                    "ts": int(k[0]),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                }
            )
        return out

    # -- analysis helpers -----------------------------------------------------
    async def trend_summary(self, symbol: str, bars: int = 96) -> dict[str, Any]:
        """Lightweight trend read from 1h klines (no TA lib dependency):
        returns price, 24h change %, short momentum, RSI(14) and a signal."""
        klines = await self.get_klines(symbol, interval="1h", limit=max(bars, 96))
        closes = [k["close"] for k in klines]
        if len(closes) < 24:
            return {"symbol": symbol, "signal": "NEUTRAL", "note": "not enough data"}
        price = closes[-1]
        change_24h = _change_pct_over(closes, 24)          # same def as analyze_symbol
        rsi = _wilder_rsi(closes)                           # true Wilder RSI(14)
        # short MAs
        ma7 = sum(closes[-7:]) / 7
        ma25 = sum(closes[-25:]) / 25 if len(closes) >= 25 else ma7
        note = ""
        if rsi >= 75 or rsi <= 25:
            signal = "NEUTRAL"  # extreme momentum is a warning, not a trend
            note = "overbought" if rsi >= 75 else "oversold"
        elif ma7 > ma25:
            signal = "BULLISH"
        elif ma7 < ma25:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"
        return {
            "symbol": symbol,
            "price": price,
            "change_24h_pct": round(change_24h, 2),
            "rsi14": round(rsi, 1),
            "ma7": round(ma7, 2),
            "ma25": round(ma25, 2),
            "signal": signal,
            "note": note,
        }

    async def analyze_symbol(self, symbol: str, bars: int = 168) -> dict[str, Any]:
        """Deep single-coin analysis used by the Market analysis flow.

        Pulls hourly klines (default 7 days) and returns trend, momentum,
        volatility and level reads — no TA library, all deterministic math:
          price / 24h+7d change / 24h+7d ranges / RSI(14) / MA7 / MA25 /
          MA7 slope / ATR(14h)% / nearest support & resistance / bias signal.
        """
        base = symbol.replace("USDT", "")
        klines = await self.get_klines(symbol, interval="1h", limit=max(bars, 168))
        if len(klines) < 49:
            return {"symbol": symbol, "note": "not enough market data"}
        closes = [k["close"] for k in klines]
        highs = [k["high"] for k in klines]
        lows = [k["low"] for k in klines]
        price = closes[-1]

        def pct(a: float, b: float) -> float:
            return (a / b - 1.0) * 100.0 if b else 0.0

        change_24h = _change_pct_over(closes, 24)
        change_7d = pct(price, closes[0]) if len(closes) >= 168 else None

        rsi = _wilder_rsi(closes)                           # true Wilder RSI(14)

        ma7 = sum(closes[-7:]) / 7
        ma25 = sum(closes[-25:]) / 25 if len(closes) >= 25 else ma7
        ma7_prev = sum(closes[-13:-6]) / 7 if len(closes) >= 13 else ma7
        ma7_slope = pct(ma7, ma7_prev)

        # ATR(14h) as % of price
        trs = []
        for i in range(1, len(closes)):
            hi, lo, pc = highs[i], lows[i], closes[i - 1]
            trs.append(max(hi - lo, abs(hi - pc), abs(lo - pc)))
        atr = sum(trs[-14:]) / 14 if len(trs) >= 14 else (sum(trs) / len(trs) if trs else 0.0)
        atr14_pct = (atr / price * 100.0) if price else 0.0

        high_24h = max(highs[-24:])
        low_24h = min(lows[-24:])
        high_7d = max(highs[-168:]) if len(highs) >= 168 else max(highs)
        low_7d = min(lows[-168:]) if len(lows) >= 168 else min(lows)

        # nearest support below / resistance above from recent swing ranges
        support, sup_scope = low_24h, "24h"
        if low_7d < low_24h and low_7d < price:
            support, sup_scope = low_7d, "7d"
        resistance, res_scope = high_24h, "24h"
        if high_7d > high_24h and high_7d > price:
            resistance, res_scope = high_7d, "7d"

        regime = "neutral"
        if rsi >= 75:
            regime = "overbought"
        elif rsi <= 25:
            regime = "oversold"
        ma_bull = price > ma7 > ma25
        ma_bear = price < ma7 < ma25
        if regime != "neutral":
            signal = "NEUTRAL"          # extremes warn against chasing
        elif ma_bull and rsi > 50:
            signal = "BULLISH"
        elif ma_bear and rsi < 50:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"

        return {
            "symbol": symbol,
            "base": base,
            "price": price,
            "change_24h_pct": round(change_24h, 2),
            "change_7d_pct": round(change_7d, 2) if change_7d is not None else None,
            "high_24h": high_24h,
            "low_24h": low_24h,
            "high_7d": high_7d,
            "low_7d": low_7d,
            "rsi14": round(rsi, 1),
            "ma7": ma7,
            "ma25": ma25,
            "ma7_slope_pct": round(ma7_slope, 2),
            "atr14_pct": round(atr14_pct, 3),
            "support": support,
            "support_scope": sup_scope,
            "resistance": resistance,
            "resistance_scope": res_scope,
            "regime": regime,
            "signal": signal,
        }

    # -- internals -----------------------------------------------------------
    async def _get_json(self, path: str, params: Optional[dict] = None) -> Any:
        url = f"{self.base}{path}"
        try:
            resp = await self._http.get(url, params=params)
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPError, ValueError) as exc:  # network / non-JSON
            raise MarketDataError(f"market data request failed for {url}: {exc}") from exc


async def ping_market(base_url: str) -> bool:
    """Quick connectivity probe used by the UI's status badge."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{base_url.rstrip('/')}/api/v3/ping")
            return resp.status_code == 200 and resp.json() == {}
    except Exception:
        return False
