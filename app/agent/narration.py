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
            "See the audit trail for reasons (e.g. the coin is off the approved list or cash is insufficient)."
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
        "Each action is guardrailed and reversible. Approve what you want; reject what you don't."
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
    """Rule-based intent parse — fast, deterministic, always available.

    Understands not just read-only queries (risk / status / analysis) but the
    full command surface: open/close positions, add or release margin, move
    free balance between spot and the futures wallet, switch agent mode, and
    edit guardrail limits — all from plain language in the console.
    """
    m = message.lower().strip()

    # --- guardrail edits (checked first: "de-risk target" shares wording with
    # the protect intent, so an edit like "raise de-risk target to 15%" must
    # not be swallowed by the "de-risk" protect trigger) -------------------
    rail = _parse_rail_edit(m)
    if rail:
        return rail

    # --- read-only queries ------------------------------------------------
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
    if any(k in m for k in ["if ", "when ", "condition", "once btc", "once eth"]):
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

    # --- agent mode ---------------------------------------------------------
    if any(k in m for k in ["manual mode", "back to manual", "switch to manual",
                            "disable auto", "turn off auto", "off auto"]):
        return {"intent": "agent_manual"}
    if any(k in m for k in ["auto mode", "automatic mode", "automatic agent",
                            "enable auto", "go auto", "auto agent"]):
        return {"intent": "agent_auto"}

    # --- position margin moves ----------------------------------------------
    if any(k in m for k in ["release margin", "pull margin", "remove margin",
                            "return margin", "take margin off"]):
        sym = _market_symbol(m)
        if sym:
            return {"intent": "release", "symbol": sym}
    add_markers = ("add ", "more ", "increase ", "top up", "put ", "boost ")
    if any(k in m for k in ["top up", "put more margin", "boost margin"]) or (
        "margin" in m and any(k in m for k in add_markers)
    ):
        sym = _market_symbol(m) or "BTCUSDT"
        return {"intent": "add_margin", "symbol": sym,
                "amount_usdt": _first_number(m)}

    # --- free-balance moves spot <-> futures wallet --------------------------
    if _is_funds_move(m):
        return _parse_funds_move(m)

    # --- close ---------------------------------------------------------------
    if any(k in m for k in ["close", "exit", "flat", "close out", "square"]) or (
        m.startswith("close")
    ):
        sym = _market_symbol(m)
        close_all = any(k in m for k in ["all", "everything", "all positions"])
        return {"intent": "close", "symbol": sym or "", "close_all": close_all}

    # --- open -----------------------------------------------------------------
    open_match = None
    for pat in ("open long", "open short", "go long", "go short", "buy long",
                "sell short", "take a long", "take a short"):
        if pat in m:
            open_match = pat
            break
    if m.startswith(("long ", "short ")):
        open_match = m.split()[0]
    if open_match:
        side = "LONG" if "long" in open_match else "SHORT"
        sym = _market_symbol(m) or "BTCUSDT"
        numbers = _all_numbers(m)
        margin = numbers[0] if numbers else None
        lev = _number_before_x(m) or 10.0
        return {
            "intent": "open",
            "symbol": sym,
            "side": side,
            "margin_usdt": margin,
            "leverage": lev,
        }
    return {"intent": "chat"}


def _has_symbol(m: str) -> bool:
    return bool(_market_symbol(m))


def _first_number(m: str):
    import re

    nums = re.findall(r"\d+(?:\.\d+)?", m)
    try:
        return float(nums[0]) if nums else None
    except (TypeError, ValueError):
        return None


def _all_numbers(m: str):
    import re

    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", m)]


def _number_before_x(m: str):
    import re

    hit = re.search(r"(\d+(?:\.\d+)?)\s*x", m)
    return float(hit.group(1)) if hit else None


def _is_funds_move(m: str) -> bool:
    verb = any(
        k in m
        for k in ["move ", "transfer ", "send ", "deposit ", "withdraw ",
                  "shift ", "put ", "take out", "top up wallet", "pull out"]
    )
    dest = any(
        k in m
        for k in ["to futures", "into futures", "futures wallet", "futures balance",
                  "to spot", "into spot", "spot wallet", "spot balance", "back to spot"]
    )
    return verb and dest


def _parse_funds_move(m: str) -> dict[str, Any]:
    import re

    to_futures = any(
        k in m
        for k in ["to futures", "into futures", "futures wallet", "futures balance",
                  "deposit", "futures side"]
    )
    to_spot = any(
        k in m
        for k in ["to spot", "into spot", "spot wallet", "spot balance", "back to spot",
                  "withdraw", "return to spot"]
    )
    direction = "to_futures" if to_futures and not to_spot else "to_spot"
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", m)]
    return {
        "intent": "funds",
        "direction": direction,
        "amount_usdt": nums[0] if nums else None,
    }


RAIL_ALIASES: dict[str, str] = {
    "danger": "liq_danger_pct",
    "danger zone": "liq_danger_pct",
    "danger threshold": "liq_danger_pct",
    "watch": "liq_warn_pct",
    "watch zone": "liq_warn_pct",
    "warning zone": "liq_warn_pct",
    "de-risk target": "liq_target_dist_pct",
    "de risk target": "liq_target_dist_pct",
    "headroom": "liq_target_dist_pct",
    "target distance": "liq_target_dist_pct",
    "liquidation distance": "liq_target_dist_pct",
}


def _parse_rail_edit(m: str) -> Optional[dict[str, Any]]:
    import re

    verb = any(
        k in m
        for k in ["set ", "raise ", "lower ", "change ", "make ", "edit ", "put "]
    )
    if not verb:
        return None
    for phrase, key in RAIL_ALIASES.items():
        if phrase not in m:
            continue
        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", m)]
        if not nums:
            break
        raw = nums[-1]
        has_pct = "%" in m
        if key.endswith("_pct"):
            value = raw / 100.0 if has_pct or raw > 1 else raw
        elif "leverage" in key:
            value = raw
        else:
            value = raw
        return {"intent": "rail_edit", "rail_key": key, "rail_value": value}
    return None


async def parse_intent_llm(
    config: AppConfig, message: str, targets: dict
) -> dict[str, Any]:
    """Optional LLM intent parse; falls back to the rule-based parser.

    ``targets`` doubles as portfolio context: when provided it is sent to the
    model alongside the message so conversational answers ("what is my biggest
    risk right now?") can be grounded in the real account state. When no key is
    configured the deterministic parser handles the full command surface.
    """
    if not config.openai_api_key:
        return parse_intent(message)
    user = {"message": message}
    if targets:
        user["context"] = targets
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{config.openai_base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {config.openai_api_key}"},
                json={
                    "model": config.openai_model,
                    "messages": [
                        {"role": "system", "content": INTENT_CHECK_SYSTEM},
                        {"role": "user", "content": json.dumps(user)[:6000]},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 400,
                },
            )
            resp.raise_for_status()
            parsed = json.loads(resp.json()["choices"][0]["message"]["content"])
            if isinstance(parsed, dict) and parsed.get("intent"):
                return parsed
            if isinstance(parsed, dict) and parsed.get("reply"):
                parsed["intent"] = "chat"
                return parsed
    except Exception:
        pass
    return parse_intent(message)


INTENT_CHECK_SYSTEM = """You are the Liquidation Guardian's front brain — a futures
risk-protection agent for Binance. The user just typed something into the console.
Portfolio context is attached when available. Decide what they want and reply with
ONE JSON object, nothing else.

If the message is a COMMAND (open/close a position, add/release margin, move funds
between spot and futures, arm a price condition, ask for risk/status/protect/analyze,
switch agent mode, edit a danger/watch/de-risk threshold), output the matching command
object with keys chosen from {"intent","symbol","side","margin_usdt","leverage",
"amount_usdt","direction","close_all","condition","rail_key","rail_value"}:
- "protect" for "protect my positions" / "de-risk now"
- "risk" for "risk report" / "what's my liquidation risk"
- "status" for "status" / "summary"
- "market" for "analyze BTC" (symbol optional)
- "open" for "open long BTC 500 at 10x" (side, symbol, margin_usdt, leverage default 10)
- "close" for "close BTC" or "close all my positions" (close_all true when all)
- "add_margin" for "add 300 margin to BTC" (symbol, amount_usdt)
- "release" for "release margin on BTC" (symbol)
- "funds" for "move 500 to futures" / "move 500 to spot" (direction, amount_usdt)
- "add_condition" for "if BTC drops below 60000 add 400 margin" (fill condition object)
- "agent_auto" for "auto mode", "agent_manual" for "manual mode"
- "rail_edit" for "set danger zone to 5%" / "set watch zone to 8%" (rail_key,
  rail_value) — only danger/watch/de-risk target are editable
Never invent numbers or symbols not in the message; symbols are always "XXXUSDT".

Otherwise (a question, greeting, or general chat), output {"intent":"chat",
"reply":"<short, helpful answer>"}. Answer concisely using the attached context;
you may explain concepts (liquidation price, margin, leverage). If the user seems
to ask you to do something that is not a supported command, say what you can do and
invite them. Never claim you executed an action — actions always need a command
object and the operator's approval. Reply ONLY with the JSON object."""



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
