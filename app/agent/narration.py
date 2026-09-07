"""Narration: turning engine output into plain-English updates.

Two modes:
- template (default): deterministic, zero-dependency prose builders.
- llm: optional OpenAI-compatible chat call (works with OpenAI, Groq,
  OpenRouter, etc.) that rephrases the engine's structured plan.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import httpx

from app.agent.prompts import SYSTEM_NARRATOR
from app.config import AppConfig
from app.models import AccountState

MONEY = "${:,.2f}"


# ---------------------------------------------------------------------------
# Template narration
# ---------------------------------------------------------------------------


def narrate_risk_report(state: AccountState, guardrails: Any) -> str:
    from app.agent.risk import portfolio_risk_summary

    lines = [
        f"🛰️ Risk report — total ${state.total_value_usdt:,.2f} "
        f"(spot ${state.spot_value_usdt:,.2f}, futures equity ${state.futures_equity_usdt:,.2f})."
    ]
    if not state.futures_positions:
        lines.append("No open futures positions — nothing to liquidate. ✅")
        return "\n".join(lines)

    summary = portfolio_risk_summary(
        state.futures_positions,
        state.total_value_usdt,
        guardrails.liq_warn_pct,
        guardrails.liq_danger_pct,
    )
    ranked = summary["positions"]
    for item in ranked:
        icon = {"OK": "✅", "WATCH": "👀", "DANGER": "🚨"}[item["risk"]]
        lines.append(
            f"{icon} {item['symbol']} {item['risk']} — {item['distance_pct']:.1f}% from liq, "
            f"{item['portfolio_weight_pct']:.1f}% of portfolio, {item['score']}/100 risk score."
        )

    top = summary["top_risk"]
    if top:
        lines.append(f"Top action: {top['advice']}")
        lines.append(f"Why: {top['reason']}")

    danger = [p for p in state.futures_positions if p.risk.value == "DANGER"]
    if danger:
        lines.append(
            "Say “protect my positions” and I’ll turn this into a concrete de-risk plan."
        )
    else:
        lines.append("No position is in the danger zone right now.")
    return "\n".join(lines)


def narrate_de_risk(danger_positions: list[Any], pending: list[Any]) -> str:
    if not pending:
        return (
            "🚨 Positions in danger, but every de-risk option was blocked by a guardrail. "
            "See the audit trail for reasons (likely daily trade cap or insufficient cash)."
        )
    lines = ["🛡️ De-risk plan — review and approve:"]
    for p in pending:
        if p.kind.value == "TRANSFER":
            lines.append(
                f"  • ADD MARGIN ${p.est_value_usdt:,.2f} to {p.symbol}. {p.reason}"
            )
        else:
            lines.append(
                f"  • REDUCE {p.symbol.replace('USDT', '')}: sell {p.est_quantity:,.6f} ≈ ${p.est_value_usdt:,.2f}. {p.reason}"
            )
    lines.append(
        "Each action is size-capped and reversible. Approve what you want; reject what you don't."
    )
    return "\n".join(lines)


def narrate_execution(result: Any, summary: Optional[str] = None) -> str:
    if result.ok:
        if summary:
            return f"✅ {summary} {result.message}"
        return (
            f"✅ Executed {result.side.value} {result.symbol} for {MONEY.format(result.executed_value_usdt)}. "
            f"{result.message}"
        )
    return f"❌ Execution failed for {result.symbol}: {result.message}"


def narrate_condition_added(cond: Any) -> str:
    return (
        f"🎯 Condition armed: if {cond.symbol} goes {cond.op.value.lower()} "
        f"{MONEY.format(cond.price)}, I'll propose a {cond.side.value} of "
        f"{MONEY.format(cond.amount_usdt)} — pending your approval."
    )


def narrate_condition_fired(cond: Any, price: float) -> str:
    return (
        f"🚨 Condition fired: {cond.symbol} hit {MONEY.format(price)} "
        f"({cond.op.value} {MONEY.format(cond.price)}). Proposing {cond.side.value} "
        f"{MONEY.format(cond.amount_usdt)} — approve to execute."
    )


# ---------------------------------------------------------------------------
# LLM narration (optional; graceful fallback)
# ---------------------------------------------------------------------------


async def llm_narrate(
    config: AppConfig,
    plan: dict[str, Any],
    fallback: str,
) -> str:
    if not config.narration_llm or not config.openai_api_key:
        return fallback
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                f"{config.openai_base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {config.openai_api_key}"},
                json={
                    "model": config.openai_model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_NARRATOR},
                        {"role": "user", "content": json.dumps(plan)[:4000]},
                    ],
                    "temperature": 0.4,
                    "max_tokens": 300,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return fallback


# ---------------------------------------------------------------------------
# Intent parsing (optional LLM; deterministic keyword fallback)
# ---------------------------------------------------------------------------


_COIN_TOKENS = {
    "btcusdt": "BTCUSDT",
    "btc": "BTCUSDT",
    "ethusdt": "ETHUSDT",
    "eth": "ETHUSDT",
    "bnbusdt": "BNBUSDT",
    "bnb": "BNBUSDT",
    "solusdt": "SOLUSDT",
    "sol": "SOLUSDT",
}


def _market_symbol(m: str) -> str:
    """Pull a coin token out of a market-analysis request, if any."""
    import re

    for tok in re.findall(r"\b(btcusdt|ethusdt|bnbusdt|solusdt|btc|eth|bnb|sol)\b", m):
        return _COIN_TOKENS[tok]
    return ""


def parse_intent(message: str) -> dict[str, Any]:
    """Rule-based intent parse — fast, deterministic, always available."""
    m = message.lower().strip()
    if any(
        k in m
        for k in [
            "protect",
            "de-risk",
            "de risk",
            "save my position",
            "defend",
            "deleverage now",
            "reduce exposure",
        ]
    ):
        return {"intent": "protect"}
    if any(
        k in m
        for k in [
            "risk report",
            "risk check",
            "liquidation risk",
            "how are we doing",
            "check portfolio",
            "check risk",
            "what's my risk",
            "liq price",
            "danger zone",
        ]
    ):
        return {"intent": "risk"}
    if any(
        k in m
        for k in [
            "status",
            "summary",
            "show me",
            "portfolio status",
            "what's up",
            "balance",
        ]
    ):
        return {"intent": "status"}
    if any(k in m for k in ["if ", "when ", "condition"]):
        return {"intent": "add_condition", "condition": _parse_condition_from_text(m)}
    if any(
        k in m
        for k in [
            "market analysis",
            "analyze",
            "analysis",
            "market trend",
            "read the tape",
            "what's the market",
            "analyse",
            "tape",
        ]
    ):
        sym = _market_symbol(m)
        out = {"intent": "market"}
        if sym:
            out["symbol"] = sym
        return out
    if any(k in m for k in ["open long ", "open short ", "go long ", "go short "]):
        return {"intent": "market"}
    return {"intent": "chat"}


async def parse_intent_llm(
    config: AppConfig, message: str, targets: dict
) -> dict[str, Any]:
    """Optional LLM intent parse; falls back to the rule-based parser."""
    if not config.openai_api_key:
        return parse_intent(message)
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{config.openai_base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {config.openai_api_key}"},
                json={
                    "model": config.openai_model,
                    "messages": [
                        {"role": "system", "content": INTENT_CHECK_SYSTEM},
                        {"role": "user", "content": json.dumps({"message": message})},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 300,
                },
            )
            resp.raise_for_status()
            parsed = json.loads(resp.json()["choices"][0]["message"]["content"])
            if isinstance(parsed, dict) and "intent" in parsed:
                return parsed
    except Exception:
        pass
    return parse_intent(message)


INTENT_CHECK_SYSTEM = """You are a strict intent parser for a futures risk agent.
From the user message, output ONLY JSON:
{"intent": "protect"|"risk"|"status"|"add_condition"|"market"|"chat",
 "condition": {symbol, op: "ABOVE"|"BELOW", price, side: "BUY"|"SELL", amount_usdt, note},
 "symbol": "BTCUSDT"}
- "protect my positions" / "de-risk now" -> protect
- "risk report" / "what's my liquidation risk" -> risk
- "status" / "summary" -> status
- "market analysis" / "analyze the market" -> market (no symbol)
- "analyze BTC" / "market analysis ethusdt" -> market + symbol field
- "if BTC drops below 60000 buy 400" -> add_condition
- else chat. Never invent numbers or symbols."""


# ---------------------------------------------------------------------------
# Text parsers (deterministic)
# ---------------------------------------------------------------------------


def _parse_condition_from_text(message: str) -> dict[str, Any]:
    import re

    m = re.search(r"(BTC|ETH|BNB|SOL)", message, re.I)
    symbol = (m.group(1).upper() + "USDT") if m else "BTCUSDT"
    op = (
        "BELOW"
        if re.search(
            r"(below|drops? (to|below)|under|falls? (to|below))", message, re.I
        )
        else "ABOVE"
    )
    side = (
        "BUY" if re.search(r"\b(buy|purchase|invest|add)\b", message, re.I) else "SELL"
    )

    numbers = [float(x.replace(",", "")) for x in re.findall(r"\d[\d,.]*", message)]
    if not numbers:
        return {
            "symbol": symbol,
            "op": op,
            "price": 0.0,
            "side": side,
            "amount_usdt": 500.0,
            "note": message.strip(),
        }

    price = None
    for kw in ("above", "below", "at", "to"):
        km = re.search(kw + r"\s*(\d[\d,.]*)", message, re.I)
        if km:
            price = float(km.group(1).replace(",", ""))
            break
    if price is None:
        price = numbers[0]

    amount = None
    for kw in ("buy", "sell", "invest", "spend", "purchase", "add"):
        km = re.search(r"\b" + kw + r"[^\d]*(\d[\d,.]*)", message, re.I)
        if km:
            cand = float(km.group(1).replace(",", ""))
            if abs(cand - price) > 1e-6:
                amount = cand
                break
    if amount is None:
        amount = next((n for n in numbers if abs(n - price) > 1e-6), 500.0)

    return {
        "symbol": symbol,
        "op": op,
        "price": price,
        "side": side,
        "amount_usdt": amount,
        "note": message.strip(),
    }
