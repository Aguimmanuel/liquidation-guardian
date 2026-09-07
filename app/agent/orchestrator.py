"""Agent orchestrator.

The agent loop — the "brain" of the Liquidation Guardian:

    observe (live prices + positions) -> analyze (liq distance, risk zones)
    -> decide (guardrailed de-risk proposals) -> narrate -> [human approval]
    -> execute -> audit

Every decision path is deterministic and guardrailed; the LLM (optional) only
improves narration and free-text intent parsing. Nothing executes without an
explicit human approve, mirroring Binance Agent OS's confirm-before-execute.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from app.adapters.base import ExecutionAdapter
from app.agent.guardrails import (
    GuardrailConfig,
    check_futures_funds_sufficient,
    check_futures_position_exists,
    check_spot_funds_sufficient,
    check_symbol_allowed,
)
from app.agent.narration import (
    narrate_condition_added,
    narrate_condition_fired,
    narrate_de_risk,
    narrate_execution,
    narrate_risk_report,
    parse_intent,
    parse_intent_llm,
)
from app.agent.risk import (
    portfolio_risk_summary,
    releasable_margin,
    required_cut,
    required_margin,
)
from app.config import (
    EDITABLE_RAIL_KEYS,
    RAIL_BOUNDS,
    AppConfig,
)
from app.market.client import MarketClient
from app.models import (
    AccountState,
    AuditEntry,
    AuditLevel,
    Condition,
    ConditionOp,
    FuturesPosition,
    PositionSide,
    ProposalKind,
    ProposalStatus,
    Side,
    TradeProposal,
    TransferKind,
    utcnow,
)


@dataclass
class AgentResponse:
    message: str
    intent: str
    state: dict[str, Any]
    proposals: list[dict[str, Any]] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    options: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "intent": self.intent,
            "state": self.state,
            "proposals": self.proposals,
            "events": self.events,
            "options": self.options,
        }


def _as_float(v: Any) -> Optional[float]:
    """Best-effort number extraction from parser output (may already be a
    float, a numeric string, or an unparsable token)."""
    if v is None:
        return None
    try:
        f = float(v)
        return f
    except (TypeError, ValueError):
        return None


AUTO_MODE_NOTICE = (
    "⚠️ Automatic agent — read before enabling.\n\n"
    "When automatic mode is on, the agent acts on your behalf: it can reduce risk, "
    "add margin, close positions, and execute the conditions you arm, without asking "
    "you to approve each step. It also watches your open positions around the clock "
    "and will de-risk them on its own before they reach liquidation — you do not have "
    "to ask it to. Guardrails still apply, but guardrails cannot guarantee "
    "profit or prevent losses.\n\n"
    "Note that when using the automatic agent, agents can make mistakes."
)


class Orchestrator:
    def __init__(
        self,
        config: AppConfig,
        guardrails: GuardrailConfig,
        adapter: ExecutionAdapter,
        market: MarketClient,
    ):
        self.config = config
        self.guardrails = guardrails
        self.adapter = adapter
        self.market = market
        self.conditions: dict[str, Condition] = {}
        self.proposals: dict[str, TradeProposal] = {}
        self.audit: list[AuditEntry] = []
        # manual = every action waits for explicit approval (default);
        # auto = the agent may act on its own once the user consents.
        self.agent_mode: str = "manual"
        self.auto_consent: bool = False
        # per-symbol throttle for the danger sweep (see monitor_positions /
        # auto_protect_scan) so a blocked retry never spams the audit trail.
        self._auto_risk_attempt: dict[str, datetime] = {}
        self._auto_risk_retry_s: float = 120.0
        # per-symbol last-known risk zone (OK / WATCH / DANGER) so the always-on
        # monitor can log every zone change instead of only reacting to DANGER.
        self._monitor_zone: dict[str, str] = {}
        self._load_guardrail_profile()

    # ------------------------------------------- guardrail profile persistence
    def _profile_path(self):
        from pathlib import Path

        base = getattr(self.config, "guardrail_state_file", "data/guardrail_state.json")
        path = Path(base)
        if not path.is_absolute():
            path = Path.cwd() / base
        return path

    def _save_guardrail_profile(self) -> None:
        """Persist the editable risk profile.

        An edited profile is a choice the user made, so it must survive a
        restart instead of silently resetting to defaults. Best-effort:
        persistence must never break the agent.
        """
        g = self.guardrails
        try:
            payload = {
                "values": {k: getattr(g, k) for k in EDITABLE_RAIL_KEYS},
                "symbol_allowlist": list(g.symbol_allowlist),
            }
            path = self._profile_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload))
        except Exception:
            pass  # persistence is best-effort; never break the agent for it

    def _load_guardrail_profile(self) -> None:
        """Apply a persisted profile (values) at startup, overriding env
        defaults. Older files that also carried the retired 24h lock fields
        are read back-compatibly — the lock keys are simply ignored.
        """
        g = self.guardrails
        try:
            path = self._profile_path()
            if not path.exists():
                return
            payload = json.loads(path.read_text())
            if isinstance(payload.get("values"), dict):
                for k in EDITABLE_RAIL_KEYS:
                    if k not in payload["values"]:
                        continue
                    v = payload["values"][k]
                    lo, hi = RAIL_BOUNDS.get(k, (float("-inf"), float("inf")))
                    try:
                        vf = float(v)
                        if math.isfinite(vf) and lo <= vf <= hi:
                            setattr(g, k, vf)
                    except (TypeError, ValueError):
                        continue
            allow = payload.get("symbol_allowlist")
            if isinstance(allow, list) and allow:
                g.symbol_allowlist = [
                    str(s).strip().upper() for s in allow if str(s).strip()
                ]
        except Exception:
            pass

    # ------------------------------------------------------------------ audit
    def log(
        self,
        level: AuditLevel,
        event: str,
        detail: str = "",
        proposal_id: Optional[str] = None,
    ) -> None:
        self.audit.append(
            AuditEntry(level=level, event=event, detail=detail, proposal_id=proposal_id)
        )
        if len(self.audit) > 500:
            self.audit = self.audit[-500:]

    # ------------------------------------------------------------------ state
    async def refresh(self) -> None:
        refresh = getattr(self.adapter, "refresh_prices", None)
        if refresh is not None:
            await refresh()

    def account(self) -> AccountState:
        return self.adapter.get_account()

    def snapshot(self) -> dict[str, Any]:
        state = self.account()
        danger = [p for p in state.futures_positions if p.risk.value == "DANGER"]
        guardrail_status = self.guardrail_status(state)
        pending = [
            p.to_dict()
            for p in self.proposals.values()
            if p.status == ProposalStatus.PENDING
        ]
        risk_summary = portfolio_risk_summary(
            state.futures_positions,
            state.total_value_usdt,
            self.guardrails.liq_warn_pct,
            self.guardrails.liq_danger_pct,
        )
        # enrich each open position with how much of its margin could be safely
        # returned to spot right now (None = nothing releaseable / not allowed)
        rel_by_symbol = {
            p.symbol: self._releasable_margin(p) for p in state.futures_positions
        }
        account = state.to_dict()
        for pd in account["futures_positions"]:
            pd["releasable_margin_usdt"] = rel_by_symbol.get(pd["symbol"])
        return {
            "mode": self.config.mode,
            "config": self.config.to_dict(),
            "guardrails": self.guardrails.to_dict(),
            "account": account,
            "proposals": pending,
            "conditions": [c.to_dict() for c in self.conditions.values()],
            "watch_symbols": self.watch_symbols(),
            "agent": {"mode": self.agent_mode, "consent": self.auto_consent},
            "guardrail_status": guardrail_status,
            "risk_summary": {
                "positions": len(state.futures_positions),
                "danger": len(danger),
                "nearest_distance_pct": round(
                    min((p.distance_pct for p in state.futures_positions), default=1.0)
                    * 100,
                    2,
                ),
                "top_risk": risk_summary["top_risk"],
                "positions": risk_summary["positions"],
            },
            "audit": [a.to_dict() for a in self.audit[-30:]],
            "ts": utcnow().isoformat(),
        }

    # ------------------------------------------------------------ agent entry
    async def handle_message(self, message: str) -> AgentResponse:
        intent_raw = await parse_intent_llm(
            self.config, message, self._llm_context()
        )
        intent = intent_raw.get("intent", "chat")
        await self.refresh()

        if intent in ("risk", "check"):
            return await self._do_risk_report()
        if intent in ("protect", "de_risk"):
            return await self._do_de_risk()
        if intent == "status":
            return await self._do_status()
        if intent == "market":
            return await self.market_analysis(intent_raw.get("symbol"))
        if intent == "add_condition":
            return await self._do_add_condition(intent_raw.get("condition", {}))
        if intent == "open":
            return await self._console_open(intent_raw)
        if intent == "close":
            return await self._console_close(intent_raw)
        if intent == "add_margin":
            return await self._console_add_margin(intent_raw)
        if intent == "release":
            return await self._console_release(intent_raw)
        if intent == "funds":
            return await self._console_funds(intent_raw)
        if intent == "agent_auto":
            return await self._console_agent_auto(message)
        if intent == "agent_manual":
            return self.set_agent_mode("manual")
        if intent == "rail_edit":
            return self._console_rail_edit(intent_raw)
        # free-form chat: the LLM (when configured) supplies a grounded answer;
        # otherwise we fall back to a compact status + command menu
        if intent == "chat":
            reply = intent_raw.get("reply") if isinstance(intent_raw, dict) else None
            if isinstance(reply, str) and reply.strip():
                return AgentResponse(
                    message=reply.strip(), intent="chat", state=self.snapshot()
                )
            return await self._do_chat(message)
        return await self._do_chat(message)

    # ----------------------------------------------- console command actions
    async def _console_open(self, p: dict[str, Any]) -> AgentResponse:
        """“open long BTC 500 at 10x” -> guardrailed OPEN proposal/order."""
        symbol = str(p.get("symbol") or "").strip().upper()
        if not symbol:
            return AgentResponse(
                message="Which coin? Say e.g. “open long BTC with 500 margin at 10x”.",
                intent="open",
                state=self.snapshot(),
            )
        margin = _as_float(p.get("margin_usdt") or p.get("amount_usdt"))
        if margin is None or margin <= 0:
            return AgentResponse(
                message="How much margin? Say e.g. “open long BTC 500 at 10x”.",
                intent="open",
                state=self.snapshot(),
            )
        side = str(p.get("side", "LONG")).upper()
        if side not in ("LONG", "SHORT"):
            return AgentResponse(
                message="Say “long” or “short”, e.g. “open long BTC 500 at 10x”.",
                intent="open",
                state=self.snapshot(),
            )
        lev = _as_float(p.get("leverage")) or 10.0
        return await self.queue_open(symbol, side, margin, lev)

    async def _console_close(self, p: dict[str, Any]) -> AgentResponse:
        symbol = str(p.get("symbol") or "").strip().upper()
        state = self.account()
        if str(p.get("close_all") or "").lower() in ("true", "1", "yes", "all"):
            symbol = ""
        if not symbol:
            open_syms = [fp.symbol for fp in state.futures_positions]
            if not open_syms:
                return AgentResponse(
                    message="No open positions to close.",
                    intent="close",
                    state=self.snapshot(),
                )
            responses = [
                await self.queue_close(s, close_all=True) for s in open_syms
            ]
            lines = [r.message for r in responses]
            return AgentResponse(
                message="\n".join(lines),
                intent="close",
                state=self.snapshot(),
            )
        if not any(fp.symbol == symbol for fp in state.futures_positions):
            return AgentResponse(
                message=f"No open {symbol} position to close.",
                intent="close",
                state=self.snapshot(),
            )
        return await self.queue_close(symbol, close_all=True)

    async def _console_add_margin(self, p: dict[str, Any]) -> AgentResponse:
        symbol = str(p.get("symbol") or "").strip().upper()
        amount = _as_float(p.get("amount_usdt") or p.get("margin_usdt"))
        if not symbol or amount is None or amount <= 0:
            return AgentResponse(
                message="Say e.g. “add 300 margin to BTC”.",
                intent="add_margin",
                state=self.snapshot(),
            )
        return await self.add_margin_to_position(symbol, amount)

    async def _console_release(self, p: dict[str, Any]) -> AgentResponse:
        symbol = str(p.get("symbol") or "").strip().upper()
        if not symbol:
            return AgentResponse(
                message="Say e.g. “release margin on BTC”.",
                intent="release",
                state=self.snapshot(),
            )
        return await self.release_margin_to_spot(symbol)

    async def _console_funds(self, p: dict[str, Any]) -> AgentResponse:
        direction = str(p.get("direction") or "").lower()
        if direction not in ("to_futures", "to_spot"):
            return AgentResponse(
                message='Say e.g. “move 500 to futures” or “move 500 to spot”.',
                intent="funds",
                state=self.snapshot(),
            )
        amount = _as_float(p.get("amount_usdt"))
        return await self.transfer_balance(direction, amount)

    async def _console_agent_auto(self, message: str) -> AgentResponse:
        m = message.lower()
        consent_words = ("consent", "understand", "confirm", "enable", "yes", "accept")
        if any(w in m for w in consent_words):
            return self.set_agent_mode("auto", consent=True)
        return AgentResponse(
            message=AUTO_MODE_NOTICE
            + "\n\nReply “I understand — enable auto” to confirm and switch.",
            intent="agent",
            state=self.snapshot(),
        )

    def _console_rail_edit(self, p: dict[str, Any]) -> AgentResponse:
        key = p.get("rail_key")
        value = p.get("rail_value")
        if not key or value is None:
            return AgentResponse(
                message="Say e.g. “set danger zone to 5%”, “watch zone to 8%” or "
                "“de-risk target to 15%”.",
                intent="rail_edit",
                state=self.snapshot(),
            )
        if key not in EDITABLE_RAIL_KEYS:
            return AgentResponse(
                message=f"“{key}” isn't an editable guardrail. Editable: danger zone, watch zone, "
                "and de-risk target.",
                intent="rail_edit",
                state=self.snapshot(),
            )
        resp = self.update_guardrails({key: value})
        return AgentResponse(
            message=resp.message,
            intent="guardrails",
            state=self.snapshot(),
        )

    # ------------------------------------------------- funds: free balance <-> spot
    async def transfer_balance(
        self,
        direction: str,
        amount: Optional[float] = None,
    ) -> AgentResponse:
        """Move free balance between spot and the futures wallet (both ways).

        Only *free* balance moves: spot cash <-> futures-wallet USDT. No open
        position's margin is touched, so liquidation protection is unchanged.
        ``amount=None`` moves the whole source balance.
        """
        direction = (direction or "").lower()
        if direction not in ("to_futures", "to_spot"):
            return AgentResponse(
                message="Direction must be “to_futures” or “to_spot”.",
                intent="funds",
                state=self.snapshot(),
            )
        await self.refresh()
        state = self.account()
        if direction == "to_futures":
            source = "spot cash"
            have = state.cash_usdt
            tkind = TransferKind.DEPOSIT_FUTURES
        else:
            source = "free futures balance"
            have = state.futures_wallet_usdt
            tkind = TransferKind.RETURN_WALLET
        if amount is None or amount <= 0:
            amount = have
        if amount <= 1e-9:
            return AgentResponse(
                message=f"Nothing to move — {source} is empty.",
                intent="funds",
                state=self.snapshot(),
            )
        if amount > have + 1e-6:
            return AgentResponse(
                message=f"Only ${have:,.2f} available in {source} — you asked for ${amount:,.2f}.",
                intent="funds",
                state=self.snapshot(),
            )
        reasons: list[str] = []
        if tkind == TransferKind.DEPOSIT_FUTURES:
            r = check_spot_funds_sufficient(state.cash_usdt, amount)
        else:
            r = check_futures_funds_sufficient(state.futures_wallet_usdt, amount)
        if not r.ok:
            reasons.append(r.reason)
        if reasons:
            self.log(AuditLevel.GUARDRAIL, "fund_move_blocked", "; ".join(reasons))
            return AgentResponse(
                message="⛔ Fund move blocked: " + " ".join(reasons),
                intent="funds",
                state=self.snapshot(),
            )
        verb = "Move" if tkind == TransferKind.RETURN_WALLET else "Deposit"
        where = (
            "back to spot cash"
            if tkind == TransferKind.RETURN_WALLET
            else "into the free futures wallet"
        )
        prop = TradeProposal(
            id=uuid.uuid4().hex[:10],
            symbol="USDT",
            side=Side.BUY,
            kind=ProposalKind.TRANSFER,
            transfer_kind=tkind,
            est_value_usdt=round(amount, 2),
            est_quantity=0.0,
            est_price=0.0,
            reason=(
                f"{verb} ${amount:,.2f} of {source} {where}. This is free balance — "
                "no open position's margin is touched, so liquidation protection "
                "is unchanged."
            ),
        )
        self.proposals[prop.id] = prop
        self.log(
            AuditLevel.INFO,
            "fund_move_proposed",
            f"{tkind.value} ${amount:,.2f}",
            prop.id,
        )
        if self._auto_active():
            self.log(
                AuditLevel.ACTION,
                "auto_mode",
                f"automatic mode executed {tkind.value} ${amount:,.2f}",
                prop.id,
            )
            resp = self.decide(prop.id, True)
            return AgentResponse(
                message="⚡ " + resp.message + " (automatic mode)",
                intent="funds",
                state=self.snapshot(),
            )
        return AgentResponse(
            message=f"📝 {prop.reason} Approve to execute.",
            intent="funds",
            state=self.snapshot(),
            proposals=[prop.to_dict()],
        )

    async def add_margin_to_position(self, symbol: str, amount: float) -> AgentResponse:
        """Plain “add $X margin to symbol” — no danger needed. Guardrailed the
        same way as the de-risk Option B (cash-backed, size-capped by cash)."""
        symbol = symbol.upper()
        await self.refresh()
        state = self.account()
        if not any(fp.symbol == symbol for fp in state.futures_positions):
            return AgentResponse(
                message=f"No open {symbol} position to add margin to.",
                intent="add_margin",
                state=self.snapshot(),
            )
        prop = TradeProposal(
            id=uuid.uuid4().hex[:10],
            symbol=symbol,
            side=Side.BUY,
            kind=ProposalKind.TRANSFER,
            transfer_kind=TransferKind.ADD_MARGIN,
            est_value_usdt=round(amount, 2),
            est_quantity=0.0,
            est_price=0.0,
            reason=(
                f"Add ${amount:,.2f} of margin to the open {symbol} position — "
                f"moves the liquidation price further away and lowers effective "
                "leverage."
            ),
        )
        allowed, reasons = self._check_de_risk(prop, state)
        if not allowed:
            self.log(
                AuditLevel.GUARDRAIL, "margin_blocked", "; ".join(reasons), prop.id
            )
            return AgentResponse(
                message="⛔ Margin add blocked: " + " ".join(reasons),
                intent="add_margin",
                state=self.snapshot(),
            )
        self.proposals[prop.id] = prop
        self.log(
            AuditLevel.INFO,
            "margin_proposed",
            f"{symbol} ${amount:,.2f}",
            prop.id,
        )
        if self._auto_active():
            self.log(
                AuditLevel.ACTION,
                "auto_mode",
                f"automatic mode added ${amount:,.2f} margin to {symbol}",
                prop.id,
            )
            resp = self.decide(prop.id, True)
            return AgentResponse(
                message="⚡ " + resp.message + " (automatic mode)",
                intent="add_margin",
                state=self.snapshot(),
            )
        return AgentResponse(
            message=f"📝 Add ${amount:,.2f} margin to {symbol}? Approve to execute.",
            intent="add_margin",
            state=self.snapshot(),
            proposals=[prop.to_dict()],
        )

    async def check_conditions(self) -> list[AgentResponse]:
        """Evaluate armed conditions against live prices (called periodically).

        Edge-triggered: a condition fires *once* when the price first crosses
        its trigger, then re-arms only after the price comes back the other
        way. Staying past the trigger never re-proposes or re-executes — that
        would otherwise spam the approval queue (and drain cash in auto mode)
        on every 2s poll while a coin simply trades beyond the price.
        """
        if not self.conditions:
            return []
        try:
            await self.refresh()
        except Exception:
            return []
        state = self.account()
        responses: list[AgentResponse] = []
        for cond in list(self.conditions.values()):
            if not cond.active:
                continue
            price = self._price_for(state, cond.symbol)
            if price is None or price <= 0:
                try:
                    price = (await self.market.get_ticker(cond.symbol)).price
                except Exception:
                    price = 0.0
            if price is None or price <= 0:
                continue
            hit = (cond.op == ConditionOp.ABOVE and price >= cond.price) or (
                cond.op == ConditionOp.BELOW and price <= cond.price
            )
            if hit and cond._was_hit:
                continue  # already fired on this crossing — do not re-fire
            cond._was_hit = hit
            if not hit:
                continue  # armed again for the next crossing
            cond.fires += 1
            cond.last_fired_at = utcnow()
            self.log(
                AuditLevel.ACTION,
                "condition_fired",
                f"{cond.symbol} {cond.op.value} {cond.price} @ {price:.2f}",
            )
            resp = await self._propose_condition_trade(cond, price)
            if resp:
                responses.append(resp)
        return responses

    # -------------------------------------------------------------- intents
    async def _do_risk_report(self) -> AgentResponse:
        state = self.account()
        msg = narrate_risk_report(state, self.guardrails)
        return AgentResponse(message=msg, intent="risk", state=self.snapshot())

    async def _do_status(self) -> AgentResponse:
        state = self.account()
        lines = ["📊 Portfolio status"]
        lines.append(
            f"  Spot: ${state.spot_value_usdt:,.2f} (cash ${state.cash_usdt:,.2f})"
        )
        lines.append(
            f"  Futures equity: ${state.futures_equity_usdt:,.2f} "
            f"(wallet ${state.futures_wallet_usdt:,.2f} + margin ${sum(p.margin_usdt for p in state.futures_positions):,.2f})"
        )
        drawdown = (
            (state.total_value_usdt / state.peak_value_usdt - 1.0) * 100
            if state.peak_value_usdt
            else 0.0
        )
        lines.append(
            f"  Total: ${state.total_value_usdt:,.2f} · drawdown {drawdown:.1f}%"
        )
        if state.futures_positions:
            lines.append("  Open positions:")
            for p in state.futures_positions:
                lines.append(
                    f"    • {p.side.value} {p.symbol} {p.effective_leverage:.1f}x — liq {p.liq_price:,.0f} "
                    f"({p.distance_pct * 100:.1f}% away, {p.risk.value})"
                )
        else:
            lines.append("  No open futures positions.")
        lines.append(
            "Armed conditions: "
            + (
                ", ".join(c.symbol for c in self.conditions.values() if c.active)
                or "none"
            )
        )
        return AgentResponse(
            message="\n".join(lines), intent="status", state=self.snapshot()
        )

    async def _do_de_risk(self) -> AgentResponse:
        """Scan open positions; for every one in DANGER, propose a guardrailed
        de-risk (reduce position or add margin)."""
        state = self.account()
        danger = [p for p in state.futures_positions if p.risk.value == "DANGER"]
        if not danger:
            watch = [p for p in state.futures_positions if p.risk.value == "WATCH"]
            if watch:
                return AgentResponse(
                    message="No position is in the danger zone right now, but these need watching:\n"
                    + "\n".join(
                        f"  • {p.side.value} {p.symbol}: {p.distance_pct * 100:.1f}% from liquidation"
                        for p in watch
                    ),
                    intent="de_risk",
                    state=self.snapshot(),
                )
            return AgentResponse(
                message="All positions are clear — nothing needs de-risking right now. "
                "I'll keep watching around the clock.",
                intent="de_risk",
                state=self.snapshot(),
            )

        before_ids = set(self.proposals)
        for pos in danger:
            self._propose_de_risk_for(pos, state)
        new_pending = [
            p
            for p in self.proposals.values()
            if p.id not in before_ids and p.status == ProposalStatus.PENDING
        ]
        self.log(
            AuditLevel.ACTION,
            "de_risk_scan",
            f"{len(danger)} position(s) in danger, {len(new_pending)} guardrailed proposal(s) queued",
        )
        if self._auto_active() and new_pending:
            return self._auto_exec_de_risk(new_pending)
        return AgentResponse(
            message=narrate_de_risk(danger, new_pending),
            intent="de_risk",
            state=self.snapshot(),
            proposals=[
                p.to_dict()
                for p in self.proposals.values()
                if p.status == ProposalStatus.PENDING
            ],
        )

    def _auto_exec_de_risk(
        self,
        new_pending: list[TradeProposal],
        *,
        event: str = "auto_mode",
        headline: str = (
            "⚡ Automatic mode — the guardian de-risked your position(s) without waiting "
            "for approval"
        ),
    ) -> AgentResponse:
        """Execute guardrailed de-risk proposals autonomously.

        Prefers the reduce (deleverage, no extra capital) and only adds margin
        when a reduce is not possible — never both. Rejects the unchosen option
        so nothing confusing lingers in the queue.
        """
        chosen = [p for p in new_pending if p.kind == ProposalKind.TRADE]
        if not chosen:
            chosen = new_pending
        chosen_ids = {p.id for p in chosen}
        self.log(
            AuditLevel.ACTION,
            event,
            f"automatic mode executed {len(chosen)} de-risk action(s) without approval",
        )
        done = []
        for prop in chosen:
            self.decide(prop.id, True)
            done.append(
                f"  • {prop.kind.value} {prop.symbol} ≈ ${prop.est_value_usdt:,.2f}"
            )
        # reject the options the agent didn't take so nothing confusing lingers
        for prop in new_pending:
            if prop.id not in chosen_ids and prop.status == ProposalStatus.PENDING:
                self.decide(prop.id, False)
        return AgentResponse(
            message=(
                headline
                + ":\n"
                + "\n".join(done)
                + "\n\nEvery action cleared the guardrails "
                "and is in the audit trail. Switch to Manual to go back to approving each step."
            ),
            intent="de_risk",
            state=self.snapshot(),
        )

    async def auto_protect_scan(self) -> list[AgentResponse]:
        """Proactive guardian sweep — the reason automatic mode exists.

        While automatic mode is active (consent given), this runs on the server
        timer and acts *without any prompt*: any position that slips into the
        danger zone is de-risked straight to the configured headroom. Manual
        mode returns immediately; the chat-driven "protect my positions" path
        stays for humans who want to trigger the same scan themselves.

        Retries are throttled per symbol so a blocked attempt (e.g. no cash)
        can never spam the audit trail every sweep.
        """
        if not self._auto_active():
            return []
        try:
            await self.refresh()
        except Exception:
            return []
        state = self.account()
        now = utcnow()
        responses: list[AgentResponse] = []
        for pos in state.futures_positions:
            if pos.risk.value != "DANGER":
                continue
            sym = pos.symbol
            last = self._auto_risk_attempt.get(sym)
            if last is not None and (now - last).total_seconds() < self._auto_risk_retry_s:
                continue
            if any(
                pp.symbol == sym and pp.status == ProposalStatus.PENDING
                for pp in self.proposals.values()
            ):
                continue
            self._auto_risk_attempt[sym] = now
            before = set(self.proposals)
            self._propose_de_risk_for(pos, state)
            new_pending = [
                p
                for p in self.proposals.values()
                if p.id not in before and p.status == ProposalStatus.PENDING
            ]
            if not new_pending:
                continue  # every option was blocked; reasons are already audited
            self.log(
                AuditLevel.ACTION,
                "auto_protect_scan",
                f"{sym} entered the danger zone ({pos.distance_pct * 100:.1f}% from liq) — "
                f"auto de-risk engaged",
            )
            responses.append(
                self._auto_exec_de_risk(
                    new_pending,
                    event="auto_de_risk",
                    headline=(
                        "⚡ Guardian auto-protect — "
                        + sym
                        + " reached the danger zone and was de-risked without waiting for approval"
                    ),
                )
            )
        return responses

    async def monitor_positions(self) -> list[AgentResponse]:
        """Always-on per-position guardian loop — runs in BOTH manual and auto.

        Runs on the server timer while the app is up, so the guardian tracks
        every open position continuously rather than waiting to be asked:

          - keeps a per-symbol risk-zone map (OK / WATCH / DANGER) and logs a
            visible event on every zone change (and on first sighting of a
            non-OK position), so its attention is never silent;
          - when a position reaches DANGER it responds immediately:
            manual mode queues a guardrailed de-risk plan for approval, auto
            mode executes it (reduce-first, no extra capital). Blocked or
            already-pending positions are throttled per symbol so nothing
            spams the audit trail.
        """
        if not self.account().futures_positions:
            # nothing open — stay quiet and keep the zone map clean
            self._monitor_zone.clear()
            return []
        try:
            await self.refresh()
        except Exception:
            return []
        state = self.account()
        now = utcnow()
        responses: list[AgentResponse] = []
        live_symbols = {p.symbol for p in state.futures_positions}
        # forget positions that have been closed so a future re-entry re-baselines
        for sym in list(self._monitor_zone):
            if sym not in live_symbols:
                self._monitor_zone.pop(sym, None)
        for pos in state.futures_positions:
            sym = pos.symbol
            zone = pos.risk.value
            prev = self._monitor_zone.get(sym)
            if prev != zone:
                if prev is None:
                    # first sighting: say we're watching it (skip calm OKs)
                    if zone != "OK":
                        self.log(
                            AuditLevel.INFO,
                            "monitor_zone",
                            f"{sym} {zone} · {pos.distance_pct * 100:.1f}% from liq — "
                            f"monitoring active",
                        )
                else:
                    self.log(
                        AuditLevel.INFO,
                        "monitor_zone",
                        f"{sym} {prev}→{zone} · {pos.distance_pct * 100:.1f}% from liq",
                    )
                self._monitor_zone[sym] = zone

            if zone != "DANGER":
                continue
            last = self._auto_risk_attempt.get(sym)
            if last is not None and (now - last).total_seconds() < self._auto_risk_retry_s:
                continue
            if any(
                pp.symbol == sym and pp.status == ProposalStatus.PENDING
                for pp in self.proposals.values()
            ):
                continue  # a plan for this symbol is already awaiting the user
            self._auto_risk_attempt[sym] = now
            before = set(self.proposals)
            self._propose_de_risk_for(pos, state)
            new_pending = [
                p
                for p in self.proposals.values()
                if p.id not in before and p.status == ProposalStatus.PENDING
            ]
            if not new_pending:
                continue  # every option was blocked; reasons are already audited
            if self._auto_active():
                self.log(
                    AuditLevel.ACTION,
                    "auto_protect_scan",
                    f"{sym} entered the danger zone ({pos.distance_pct * 100:.1f}% from liq) — "
                    f"auto de-risk engaged",
                )
                responses.append(
                    self._auto_exec_de_risk(
                        new_pending,
                        event="auto_de_risk",
                        headline=(
                            "⚡ Guardian auto-protect — "
                            + sym
                            + " reached the danger zone and was de-risked without waiting for approval"
                        ),
                    )
                )
            else:
                self.log(
                    AuditLevel.ACTION,
                    "de_risk_alert",
                    f"{sym} reached the danger zone ({pos.distance_pct * 100:.1f}% from liq) — "
                    f"de-risk plan queued for your approval",
                )
        return responses

    def _propose_de_risk_for(
        self, pos: FuturesPosition, state: AccountState
    ) -> Optional[TradeProposal]:
        """Offer both de-risk options for a danger position: add margin, or
        deleverage (close part, keep the margin). The user picks."""
        target = self.guardrails.liq_target_dist_pct
        mark = pos.mark_price
        cut = required_cut(pos, mark, target)
        add = required_margin(pos, mark, target)

        if add <= 0 and cut <= 0:
            return None  # already safe (shouldn't happen for DANGER)

        base = pos.symbol.replace("USDT", "")
        created = False
        if cut > 0:
            eff_lev = (pos.quantity - cut) * mark / pos.margin_usdt
            prop = TradeProposal(
                id=uuid.uuid4().hex[:10],
                symbol=pos.symbol,
                side=Side.SELL,
                kind=ProposalKind.TRADE,
                est_value_usdt=round(cut * mark, 2),
                est_quantity=round(cut, 8),
                est_price=mark,
                reason=(
                    f"{pos.side.value} {pos.symbol} is {pos.distance_pct * 100:.1f}% from liquidation "
                    f"(liq {pos.liq_price:,.0f}). Option A — reduce: close {cut:,.6f} {base} "
                    f"≈ ${cut * mark:,.2f} and keep the margin on the rest, dropping effective leverage "
                    f"from {pos.leverage:.0f}x to ~{eff_lev:.1f}x with ~{target * 100:.0f}% headroom."
                ),
            )
            if self._queue_de_risk(prop, state):
                created = True
        if add > 0:
            prop = TradeProposal(
                id=uuid.uuid4().hex[:10],
                symbol=pos.symbol,
                side=Side.BUY,
                kind=ProposalKind.TRANSFER,
                est_value_usdt=round(add, 2),
                est_quantity=0.0,
                est_price=0.0,
                reason=(
                    f"{pos.side.value} {pos.symbol} is {pos.distance_pct * 100:.1f}% from liquidation "
                    f"(liq {pos.liq_price:,.0f}). Option B — add margin: ${add:,.2f} moves the liq price "
                    f"to ~{target * 100:.0f}% headroom and keeps the full position."
                ),
            )
            if self._queue_de_risk(prop, state):
                created = True
        return prop if created else None

    def _queue_de_risk(self, prop: TradeProposal, state: AccountState) -> bool:
        allowed, reasons = self._check_de_risk(prop, state)
        if not allowed:
            self.log(
                AuditLevel.GUARDRAIL,
                "de_risk_blocked",
                f"{prop.symbol}: " + "; ".join(reasons),
                prop.id,
            )
            return False
        self.proposals[prop.id] = prop
        self.log(
            AuditLevel.INFO,
            "de_risk_proposed",
            f"{prop.kind.value} {prop.symbol} ${prop.est_value_usdt:,.2f}",
            prop.id,
        )
        return True

    def _check_de_risk(
        self, prop: TradeProposal, state: AccountState
    ) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        for result in (
            check_futures_position_exists(True, prop.symbol),
            check_symbol_allowed(prop.symbol, self.guardrails),
        ):
            if not result.ok:
                reasons.append(result.reason)
        if prop.kind == ProposalKind.TRANSFER and prop.transfer_kind != TransferKind.RELEASE_MARGIN:
            # ADD_MARGIN / legacy transfers take spot cash -> the futures side.
            reasons += [
                r.reason
                for r in [
                    check_spot_funds_sufficient(
                        state.cash_usdt, prop.est_value_usdt
                    )
                ]
                if not r.ok
            ]
        # REDUCE (SELL) and RELEASE_MARGIN shrink risk; protective action is
        # deliberately never capped by portfolio size or a calendar window.
        return not reasons, reasons

    # -------------------------------------------------------------- conditions
    async def _do_add_condition(self, cond: dict[str, Any]) -> AgentResponse:
        try:
            c = Condition(
                id=uuid.uuid4().hex[:10],
                symbol=cond.get("symbol", "BTCUSDT").upper(),
                op=ConditionOp(cond.get("op", "BELOW").upper()),
                price=float(cond.get("price", 0)),
                side=Side(cond.get("side", "BUY").upper()),
                amount_usdt=float(cond.get("amount_usdt", 500)),
                note=cond.get("note", ""),
            )
        except (ValueError, TypeError):
            return AgentResponse(
                message="Couldn't parse that condition. Try: “if BTC drops below 70,000, add 400 margin”.",
                intent="add_condition",
                state=self.snapshot(),
            )
        self.conditions[c.id] = c
        self.log(AuditLevel.ACTION, "condition_added", c.symbol)
        return AgentResponse(
            message=narrate_condition_added(c),
            intent="add_condition",
            state=self.snapshot(),
        )

    async def _propose_condition_trade(
        self, cond: Condition, price: float
    ) -> AgentResponse:
        state = self.account()
        has_pos = any(
            fp.symbol == cond.symbol for fp in state.futures_positions
        )
        if not has_pos:
            self.log(
                AuditLevel.GUARDRAIL,
                "condition_blocked",
                f"no open {cond.symbol} position to protect",
            )
            return AgentResponse(
                message=f"🚨 Condition fired on {cond.symbol} @ {price:,.2f}, but there is "
                f"no open {cond.symbol} position to protect.",
                intent="condition",
                state=self.snapshot(),
            )
        amount = round(float(cond.amount_usdt), 2)
        if cond.side == Side.BUY:
            # BUY condition = "add margin" to the open position. This is a real
            # margin transfer (spot cash -> position margin), never a sell.
            prop = TradeProposal(
                id=uuid.uuid4().hex[:10],
                symbol=cond.symbol,
                side=Side.BUY,
                kind=ProposalKind.TRANSFER,
                transfer_kind=TransferKind.ADD_MARGIN,
                est_value_usdt=amount,
                est_quantity=0.0,
                est_price=0.0,
                reason=(
                    f"Condition fired: {cond.symbol} {cond.op.value} {cond.price:,.2f} "
                    f"→ add ${amount:,.2f} margin"
                ),
                created_at=utcnow(),
            )
        else:
            # SELL condition = reduce the open position by ~that notional.
            prop = TradeProposal(
                id=uuid.uuid4().hex[:10],
                symbol=cond.symbol,
                side=Side.SELL,
                kind=ProposalKind.TRADE,
                est_value_usdt=amount,
                est_quantity=amount / price if price else 0.0,
                est_price=price,
                reason=(
                    f"Condition fired: {cond.symbol} {cond.op.value} {cond.price:,.2f} "
                    f"→ reduce ≈ ${amount:,.2f}"
                ),
                created_at=utcnow(),
            )
        allowed, reasons = self._check_de_risk(prop, state)
        if not allowed:
            self.log(
                AuditLevel.GUARDRAIL, "condition_blocked", "; ".join(reasons), prop.id
            )
            return AgentResponse(
                message=f"🚨 Condition fired on {cond.symbol} @ {price:,.2f}, but guardrails blocked the action: "
                + " ".join(reasons),
                intent="condition",
                state=self.snapshot(),
            )
        self.proposals[prop.id] = prop
        if self._auto_active():
            self.log(
                AuditLevel.ACTION,
                "auto_mode",
                f"automatic mode executed armed condition {cond.symbol} {cond.op.value} {cond.price}",
            )
            resp = self.decide(prop.id, True)
            return AgentResponse(
                message="⚡ "
                + narrate_condition_fired(cond, price)
                + "\n"
                + resp.message
                + " (automatic mode)",
                intent="condition",
                state=self.snapshot(),
            )
        msg = narrate_condition_fired(cond, price)
        return AgentResponse(
            message=msg,
            intent="condition",
            state=self.snapshot(),
            proposals=[prop.to_dict()],
        )

    async def _do_chat(self, message: str) -> AgentResponse:
        state = self.account()
        lines = [
            f"Here's where we stand: total ${state.total_value_usdt:,.2f}, "
            f"{len(state.futures_positions)} futures position(s) open, "
            f"{sum(1 for p in state.futures_positions if p.risk.value == 'DANGER')} in the danger zone.",
            "",
            "I take plain-language commands. Try:",
            "  • “open long BTC 500 at 10x”  /  “open short ETH 300 at 20x”",
            "  • “close BTC”  /  “close all my positions”",
            "  • “add 300 margin to BTC”  /  “release margin on BTC”",
            "  • “move 500 to futures”  /  “move 500 to spot”",
            "  • “protect my positions”  /  “risk report”  /  “status”",
            "  • “analyze BTC”  /  “if BTC drops below 60000 add 400 margin”",
            "  • “set danger zone to 5%”  /  “watch zone to 8%”  /  “de-risk target to 15%”",
            "  • “auto mode” (consent first)  /  “manual mode”",
        ]
        return AgentResponse(
            message="\n".join(lines), intent="chat", state=self.snapshot()
        )

    def _llm_context(self) -> dict[str, Any]:
        """Compact, serializable portfolio context for the LLM console brain —
        lets open-ended questions be answered from the real account state."""
        try:
            st = self.account()
        except Exception:
            return {}
        return {
            "agent_mode": self.agent_mode,
            "total_value_usdt": round(st.total_value_usdt, 2),
            "spot_cash_usdt": round(st.cash_usdt, 2),
            "futures_wallet_usdt": round(st.futures_wallet_usdt, 2),
            "open_positions": [
                {
                    "symbol": p.symbol,
                    "side": p.side.value,
                    "quantity": p.quantity,
                    "margin_usdt": round(p.margin_usdt, 2),
                    "liq_price": p.liq_price,
                    "distance_pct": round(p.distance_pct * 100, 1),
                    "risk": p.risk.value,
                }
                for p in st.futures_positions
            ],
            "pending_proposals": sum(
                1 for p in self.proposals.values() if p.status.value == "PENDING"
            ),
            "armed_conditions": len(self.conditions),
            "guardrail_values": self.guardrail_status().get("values", {}),
        }

    # ------------------------------------------------------------- approvals
    def decide(self, proposal_id: str, approve: bool) -> AgentResponse:
        prop = self.proposals.get(proposal_id)
        if prop is None:
            return AgentResponse(
                message=f"No pending proposal {proposal_id}. It may already be decided.",
                intent="decide",
                state=self.snapshot(),
            )
        if prop.status != ProposalStatus.PENDING:
            return AgentResponse(
                message=f"Proposal {proposal_id} is already {prop.status.value}.",
                intent="decide",
                state=self.snapshot(),
            )
        if not approve:
            prop.status = ProposalStatus.REJECTED
            prop.decided_at = utcnow()
            self.log(AuditLevel.INFO, "proposal_rejected", prop.symbol, prop.id)
            return AgentResponse(
                message=f"Rejected the {prop.kind.value} on {prop.symbol}. No execution. ✅",
                intent="decide",
                state=self.snapshot(),
            )

        prop.status = ProposalStatus.APPROVED
        prop.decided_at = utcnow()
        self.log(
            AuditLevel.ACTION,
            "proposal_approved",
            f"{prop.kind.value} {prop.symbol} {prop.est_value_usdt:,.2f}",
            prop.id,
        )
        try:
            result = self.adapter.execute(prop)
        except Exception as exc:
            prop.status = ProposalStatus.FAILED
            self.log(AuditLevel.ERROR, "execution_error", str(exc), prop.id)
            return AgentResponse(
                message=f"❌ Execution error: {exc}. No position changed.",
                intent="decide",
                state=self.snapshot(),
            )
        if result.ok:
            prop.status = ProposalStatus.EXECUTED
            prop.executed_price = result.executed_price
            prop.executed_value_usdt = result.executed_value_usdt
            self.log(
                AuditLevel.ACTION,
                "executed",
                f"{prop.kind.value} {prop.symbol} {result.executed_value_usdt:,.2f}",
                prop.id,
            )
            summary = None
            if prop.kind == ProposalKind.OPEN:
                side_word = "LONG" if result.side == Side.BUY else "SHORT"
                summary = (
                    f"Opened {side_word} {result.symbol} @ {result.executed_price:,.2f} "
                    f"with ${prop.open_margin_usdt:,.2f} margin at {prop.open_leverage:.0f}x "
                    f"(notional ${result.executed_value_usdt:,.2f})"
                )
            elif prop.kind == ProposalKind.CLOSE:
                summary = f"Closed {result.symbol} ≈ ${result.executed_value_usdt:,.2f} @ {result.executed_price:,.2f}"
            elif prop.kind == ProposalKind.TRANSFER:
                tkind = prop.transfer_kind or TransferKind.ADD_MARGIN
                if tkind == TransferKind.RETURN_WALLET:
                    summary = (
                        f"Moved ${result.executed_value_usdt:,.2f} of free futures "
                        f"balance back to spot cash"
                    )
                elif tkind == TransferKind.RELEASE_MARGIN:
                    summary = (
                        f"Released ${result.executed_value_usdt:,.2f} excess margin "
                        f"on {result.symbol} back to spot cash"
                    )
                elif tkind == TransferKind.DEPOSIT_FUTURES:
                    summary = (
                        f"Deposited ${result.executed_value_usdt:,.2f} of spot cash "
                        f"into the free futures wallet"
                    )
                else:
                    summary = f"Added ${result.executed_value_usdt:,.2f} margin to {result.symbol}"
            elif prop.kind == ProposalKind.TRADE and prop.side == Side.SELL:
                summary = f"Reduced {result.symbol}: sold {result.executed_value_usdt / result.executed_price:,.6f} ≈ ${result.executed_value_usdt:,.2f}"
            if summary is not None:
                message = f"✅ {summary}"
                if result.message:
                    message += f" — {result.message}"
                return AgentResponse(
                    message=message, intent="decide", state=self.snapshot()
                )
            return AgentResponse(
                message=narrate_execution(result),
                intent="decide",
                state=self.snapshot(),
            )
        prop.status = ProposalStatus.FAILED
        self.log(AuditLevel.WARN, "execution_rejected", result.message, prop.id)
        return AgentResponse(
            message=f"Execution declined: {result.message}",
            intent="decide",
            state=self.snapshot(),
        )

    def approve_all(self) -> list[AgentResponse]:
        out = []
        for pid in list(self.proposals.keys()):
            prop = self.proposals.get(pid)
            if prop and prop.status == ProposalStatus.PENDING:
                out.append(self.decide(pid, True))
        return out

    # ---------------------------------------------------------------- control
    def reset(self) -> AgentResponse:
        reset = getattr(self.adapter, "reset", None)
        if reset is None:
            return AgentResponse(
                message="Live mode can't reset the account — manage the sub-account in Binance.",
                intent="reset",
                state=self.snapshot(),
            )
        reset()
        self.proposals.clear()
        self.conditions.clear()
        self._monitor_zone.clear()
        self._auto_risk_attempt.clear()
        self.log(AuditLevel.ACTION, "reset", "demo account reset")
        return AgentResponse(
            message="♻️ Demo account reset to fresh cash. Ready when you are.",
            intent="reset",
            state=self.snapshot(),
        )

    # ------------------------------------------------------- market analysis
    def watch_symbols(self) -> list[str]:
        """The 4 coins the ticker strip + analysis watch."""
        defaults = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
        allow = self.guardrails.symbol_allowlist
        return allow if allow else defaults

    async def watch_tickers(self) -> dict[str, dict]:
        """Fresh price+24h for the live header strip (fast refresh)."""
        try:
            tickers = await self.market.get_watch_tickers(self.watch_symbols())
            return {
                s: {
                    "symbol": s,
                    "price": t.price,
                    "change_24h_pct": t.change_24h_pct,
                }
                for s, t in tickers.items()
            }
        except Exception:
            return {}

    def _auto_active(self) -> bool:
        """Automatic mode is on only after the user accepted the risk notice."""
        return self.agent_mode == "auto" and self.auto_consent

    def set_agent_mode(self, mode: str, consent: bool = False) -> AgentResponse:
        """Switch between manual (approve-everything) and automatic agent.

        Automatic mode requires explicit consent because the agent then acts on
        the user's behalf. Consent is one-way: switching back to manual revokes
        it, and re-enabling automatic mode asks for consent again.
        """
        mode = (mode or "manual").strip().lower()
        if mode not in ("manual", "auto"):
            return AgentResponse(
                message='Agent mode must be "manual" or "auto".',
                intent="agent",
                state=self.snapshot(),
            )
        if mode == "auto":
            if not consent:
                return AgentResponse(
                    message=AUTO_MODE_NOTICE
                    + "\n\nConsent is required before the automatic agent "
                    "can act on your behalf.",
                    intent="agent",
                    state=self.snapshot(),
                    options=["consent"],
                )
            self.agent_mode = "auto"
            self.auto_consent = True
            self.log(
                AuditLevel.ACTION,
                "agent_mode_auto",
                "automatic agent enabled after explicit consent",
            )
            return AgentResponse(
                message=AUTO_MODE_NOTICE
                + "\n\n⚡ Automatic mode is ON. I will act on your behalf "
                "(guardrails still apply). Switch back to Manual at any time.",
                intent="agent",
                state=self.snapshot(),
            )
        if self.agent_mode != "manual":
            self.log(
                AuditLevel.ACTION,
                "agent_mode_manual",
                "switched back to manual (approval) mode",
            )
        self.agent_mode = "manual"
        self.auto_consent = False
        return AgentResponse(
            message="🧭 Manual mode — every action waits for your approval before executing.",
            intent="agent",
            state=self.snapshot(),
        )

    def _num(self, v: float) -> str:
        if v is None:
            return "—"
        return f"{v:,.0f}" if v >= 1000 else f"{v:,.2f}"

    async def market_analysis(self, symbol: Optional[str] = None) -> AgentResponse:
        """Ask which coin to analyze, then produce a deep single-coin report.

        Without a symbol this returns a prompt with the analyzable coins; the
        UI turns those into quick-choice buttons. With a symbol it reads the
        tape (trend, momentum, volatility, levels) and adds a guardrail-aware
        game plan — including what the guardian sees if you hold that coin.
        """
        await self.refresh()
        allowed = self.watch_symbols()
        if not symbol or not str(symbol).strip():
            return AgentResponse(
                message="Which coin do you want me to analyze? Pick one and I'll pull the tape: "
                "trend, momentum, volatility, key support/resistance and what the guardian "
                "would do with it.",
                intent="market",
                state=self.snapshot(),
                options=list(allowed),
            )
        sym = str(symbol).strip().upper()
        if not sym.endswith("USDT"):
            sym = sym + "USDT"
        if sym not in allowed:
            return AgentResponse(
                message="I can only analyze the coins on your watchlist: "
                + ", ".join(c.replace("USDT", "") for c in allowed)
                + ". Add a symbol to the allowlist if you want another one.",
                intent="market",
                state=self.snapshot(),
                options=list(allowed),
            )
        analyze = getattr(self.market, "analyze_symbol", None)
        if analyze is None:
            return AgentResponse(
                message=f"Can't analyze {sym} right now — the market client has no analysis feed.",
                intent="market",
                state=self.snapshot(),
                options=list(allowed),
            )
        try:
            d = await analyze(sym)
        except Exception as exc:
            return AgentResponse(
                message=f"⚠️ Analysis feed error for {sym}: {exc}",
                intent="market",
                state=self.snapshot(),
                options=list(allowed),
            )
        if d.get("note") == "not enough market data":
            return AgentResponse(
                message=f"Not enough market data to analyze {sym} yet.",
                intent="market",
                state=self.snapshot(),
                options=list(allowed),
            )
        b = d["base"]
        price = d["price"]
        icon = {"BULLISH": "▲", "BEARISH": "▼", "NEUTRAL": "•"}.get(d["signal"], "•")
        sign = {"BULLISH": "bullish", "BEARISH": "bearish", "NEUTRAL": "sideways"}[
            d["signal"]
        ]

        lines = [
            f"📊 {b}/USDT market analysis  {icon}",
            f"  Price: ${self._num(price)}  ({d['change_24h_pct']:+.2f}% 24h)"
            + (
                f" · {d['change_7d_pct']:+.2f}% 7d"
                if d.get("change_7d_pct") is not None
                else ""
            ),
        ]
        lines.append(
            f"  Range: 24h ${self._num(d['low_24h'])} – ${self._num(d['high_24h'])}"
            + (
                f" · 7d ${self._num(d['low_7d'])} – ${self._num(d['high_7d'])}"
                if d.get("high_7d")
                else ""
            )
        )
        # trend & momentum lines
        ma_dir = (
            "rising"
            if d["ma7_slope_pct"] > 0.05
            else ("falling" if d["ma7_slope_pct"] < -0.05 else "flat")
        )
        lines.append(
            f"  Trend: {d['signal']} — price vs MA7 ${self._num(d['ma7'])} vs MA25 ${self._num(d['ma25'])} "
            f"(MA7 {ma_dir}, {d['ma7_slope_pct']:+.2f}%/h)"
        )
        regime = d["regime"]
        rsi_extra = (
            " — stretched, pullback risk"
            if regime == "overbought"
            else (" — beaten down, bounce risk" if regime == "oversold" else "")
        )
        lines.append(f"  Momentum: RSI14 {d['rsi14']}{rsi_extra}")
        lines.append(
            f"  Volatility: ATR(14h) ≈ {d['atr14_pct']:.2f}% of price per hour"
        )
        lines.append(
            f"  Levels: support ≈ ${self._num(d['support'])} ({d['support_scope']}) · "
            f"resistance ≈ ${self._num(d['resistance'])} ({d['resistance_scope']})"
        )

        # game plan
        sup, res = d["support"], d["resistance"]
        if regime == "overbought":
            plan = (
                "Do not chase here. Momentum is overheated — wait for a pullback toward "
                f"support (${self._num(sup)}) or the MA7 (${self._num(d['ma7'])}), and keep stops tight."
            )
        elif regime == "oversold":
            plan = (
                "Market is washed out. Watch for a reclaim of the MA7 "
                f"(${self._num(d['ma7'])}) before buying; a failed bounce under "
                f"${self._num(sup)} keeps the bearish case."
            )
        elif d["signal"] == "BULLISH":
            plan = (
                f"Uptrend intact above MA7/MA25. Buy-the-dip candidates near "
                f"${self._num(sup)}; a stop below ${self._num(sup)} and targets toward "
                f"${self._num(res)} fit the trend. Scale in — don't go all-in at once."
            )
        elif d["signal"] == "BEARISH":
            plan = (
                f"Downtrend below MA7/MA25. Rallies toward ${self._num(res)} are shorts "
                f"candidates with a stop above it; first downside target ${self._num(sup)}."
            )
        else:
            plan = (
                f"Trend is unclear between ${self._num(sup)} and ${self._num(res)}. "
                "Trade the range edges with defined stops, or stand aside until it picks a side."
            )
        lines.append("  Plan: " + plan)

        # guardian context for an open position on this coin
        pos = next(
            (p for p in self.account().futures_positions if p.symbol == sym), None
        )
        if pos:
            lines.append(
                f"  Your position: {pos.side.value} {pos.quantity:,.4f} {b} @ {pos.entry_price:,.2f} · "
                f"mark {pos.mark_price:,.2f} · liq {pos.liq_price:,.2f} "
                f"({pos.distance_pct * 100:.1f}% away, {pos.risk.value}). "
                + (
                    "Guardian: consider a stop below the level above and let the position run."
                    if pos.risk.value == "OK"
                    else "Guardian: this position is too close to liquidation — reduce or add margin now."
                )
            )
        else:
            lines.append(
                f"  Your book: no open {b} position — {sign} momentum noted for your next entry."
            )
        lines.append("")
        lines.append(
            "Indicators come from Binance public hourly klines (RSI/MA/ATR/range math). "
            "Educational read, not a promise — prices can always surprise."
        )
        return AgentResponse(
            message="\n".join(lines), intent="market", state=self.snapshot()
        )

    # ------------------------------------------------------------ trade flows
    async def queue_open(
        self,
        symbol: str,
        side: str,  # "LONG" | "SHORT"
        margin_usdt: float,
        leverage: float,
    ) -> AgentResponse:
        """Queue a guardrailed OPEN proposal for a new leveraged position."""
        symbol = symbol.upper()
        await self.refresh()
        state = self.account()
        results: list[str] = []

        def _note(ok: bool, msg: str) -> bool:
            if not ok:
                results.append(msg)
            return ok

        _note(
            check_symbol_allowed(symbol, self.guardrails).ok,
            f"{symbol} is not on the approved list",
        )
        _note(margin_usdt > 0, "margin must be a positive amount")
        _note(
            leverage > 0,
            "leverage must be a positive number of x",
        )
        _note(
            state.cash_usdt >= margin_usdt,
            f"not enough cash: need ${margin_usdt:,.0f}, have ${state.cash_usdt:,.0f}",
        )
        side_enum = Side.BUY if side.upper() == "LONG" else Side.SELL
        existing = next(
            (x for x in state.futures_positions if x.symbol == symbol), None
        )
        if existing is not None:
            want = PositionSide.LONG if side_enum == Side.BUY else PositionSide.SHORT
            if existing.side != want:
                _note(
                    False,
                    f"opposite {existing.side.value} position open on {symbol} — close it first",
                )
        if results:
            self.log(AuditLevel.GUARDRAIL, "open_blocked", "; ".join(results))
            return AgentResponse(
                message="⛔ Open blocked:\n  • " + "\n  • ".join(results),
                intent="open",
                state=self.snapshot(),
            )
        try:
            price = (await self.market.get_ticker(symbol)).price
        except Exception as exc:
            return AgentResponse(
                message=f"⚠️ Couldn't fetch {symbol} price: {exc}",
                intent="open",
                state=self.snapshot(),
            )
        prop = TradeProposal(
            id=uuid.uuid4().hex[:10],
            symbol=symbol,
            side=side_enum,
            kind=ProposalKind.OPEN,
            est_value_usdt=round(margin_usdt * leverage, 2),
            est_quantity=round(margin_usdt * leverage / price, 8),
            est_price=price,
            open_margin_usdt=margin_usdt,
            open_leverage=leverage,
            reason=f"Open {side.upper()} {symbol} at {leverage:.0f}x with ${margin_usdt:,.0f} margin "
            f"(entry ~${price:,.0f}). Margin comes from spot cash.",
        )
        self.proposals[prop.id] = prop
        self.log(
            AuditLevel.INFO,
            "open_proposed",
            f"{side.upper()} {symbol} {leverage:.0f}x ${margin_usdt:,.0f}",
            prop.id,
        )
        if self._auto_active():
            self.log(
                AuditLevel.ACTION,
                "auto_mode",
                f"automatic mode executed open {side.upper()} {symbol}",
                prop.id,
            )
            resp = self.decide(prop.id, True)
            return AgentResponse(
                message="⚡ "
                + resp.message
                + " (automatic mode — your order was executed without approval)",
                intent="open",
                state=self.snapshot(),
            )
        return AgentResponse(
            message=f"📝 Open {side.upper()} {symbol} {leverage:.0f}x — ${margin_usdt:,.0f} margin (notional ${margin_usdt * leverage:,.0f}). Approve to execute.",
            intent="open",
            state=self.snapshot(),
            proposals=[prop.to_dict()],
        )

    async def queue_close(
        self, symbol: str, *, close_all: bool = True, notional_usdt: float = 0.0
    ) -> AgentResponse:
        """Queue a CLOSE proposal for a position (manual exit or TP/SL)."""
        symbol = symbol.upper()
        await self.refresh()
        state = self.account()
        pos = next((x for x in state.futures_positions if x.symbol == symbol), None)
        if pos is None:
            return AgentResponse(
                message=f"No open {symbol} position to close.",
                intent="close",
                state=self.snapshot(),
            )
        mark = pos.mark_price
        qty = (
            pos.quantity
            if close_all
            else min(notional_usdt / mark if mark else 0.0, pos.quantity)
        )
        if qty <= 0:
            return AgentResponse(
                message=f"Nothing to close on {symbol}.",
                intent="close",
                state=self.snapshot(),
            )
        side = Side.BUY if pos.side == PositionSide.SHORT else Side.SELL  # opposite
        prop = TradeProposal(
            id=uuid.uuid4().hex[:10],
            symbol=symbol,
            side=side,
            kind=ProposalKind.CLOSE,
            close_all=close_all,
            keep_margin=False,
            est_value_usdt=round(qty * mark, 2),
            est_quantity=round(qty, 8),
            est_price=mark,
            reason=(
                "Close entire position"
                if close_all
                else f"Close {qty:,.6f} {symbol.replace('USDT', '')} ≈ ${qty * mark:,.2f}"
            ),
        )
        # closing reduces risk — never calendar-capped: the guardian must be
        # able to protect (or let the human exit) at any time.
        self.proposals[prop.id] = prop
        self.log(
            AuditLevel.INFO,
            "close_proposed",
            f"{symbol} {'all' if close_all else qty}",
            prop.id,
        )
        if self._auto_active():
            self.log(
                AuditLevel.ACTION,
                "auto_mode",
                f"automatic mode executed close {symbol}",
                prop.id,
            )
            resp = self.decide(prop.id, True)
            return AgentResponse(
                message="⚡ " + resp.message + " (automatic mode)",
                intent="close",
                state=self.snapshot(),
            )
        return AgentResponse(
            message=f"📝 {prop.reason} at ≈${mark:,.2f}. Approve to execute.",
            intent="close",
            state=self.snapshot(),
            proposals=[prop.to_dict()],
        )

    # ------------------------------------------------- funds (futures -> spot)
    def _releasable_margin(self, pos: FuturesPosition) -> Optional[float]:
        """USDT that can be safely pulled off an open position right now and
        moved back to spot, or None when nothing can be released.

        Releasing is only offered while the position keeps at least the
        configured de-risk headroom (see ``risk.releasable_margin``), and only
        if the de-risk target is above the watch zone (otherwise "releasing to
        target" could itself leave the position in a warning state).
        """
        target = self.guardrails.liq_target_dist_pct
        if target <= self.guardrails.liq_warn_pct:
            return None
        mark = pos.mark_price or pos.entry_price
        if pos.quantity <= 0 or pos.margin_usdt <= 0 or mark <= 0:
            return None
        rel = releasable_margin(pos, mark, target)
        if rel <= 0:
            return None
        return round(rel, 2)

    async def return_futures_wallet_to_spot(self) -> AgentResponse:
        """Move the entire free futures-wallet balance back to spot cash.

        Free balance in the futures wallet is not allocated to any isolated
        position's margin, so returning it never weakens protection — it just
        puts idle USDT back where it is visible and usable.
        """
        await self.refresh()
        state = self.account()
        amount = round(state.futures_wallet_usdt, 2)
        if amount <= 1e-9:
            return AgentResponse(
                message="Nothing to return — the free futures balance is empty.",
                intent="return",
                state=self.snapshot(),
            )
        prop = TradeProposal(
            id=uuid.uuid4().hex[:10],
            symbol="USDT",
            side=Side.BUY,
            kind=ProposalKind.TRANSFER,
            transfer_kind=TransferKind.RETURN_WALLET,
            est_value_usdt=amount,
            est_quantity=0.0,
            est_price=0.0,
            reason=(
                f"Move the free futures-wallet balance of ${amount:,.2f} back to "
                "spot cash. This is idle money — no open position is using it — "
                "so returning it changes nothing about liquidation protection."
            ),
        )
        self.proposals[prop.id] = prop
        self.log(
            AuditLevel.INFO,
            "return_proposed",
            f"futures wallet ${amount:,.2f} -> spot",
            prop.id,
        )
        if self._auto_active():
            self.log(
                AuditLevel.ACTION,
                "auto_mode",
                "automatic mode returned the futures wallet to spot",
                prop.id,
            )
            resp = self.decide(prop.id, True)
            return AgentResponse(
                message="⚡ " + resp.message + " (automatic mode)",
                intent="return",
                state=self.snapshot(),
            )
        return AgentResponse(
            message=(
                f"📝 Move ${amount:,.2f} of free futures balance back to spot cash? "
                "Approve to execute."
            ),
            intent="return",
            state=self.snapshot(),
            proposals=[prop.to_dict()],
        )

    async def release_margin_to_spot(self, symbol: str) -> AgentResponse:
        """Pull the excess margin off an open position back to spot.

        Bounded so the position keeps at least the configured de-risk headroom —
        releasing can only ever leave the position back at its normal protected
        state, never in danger.
        """
        symbol = symbol.upper()
        await self.refresh()
        state = self.account()
        pos = next(
            (p for p in state.futures_positions if p.symbol == symbol), None
        )
        if pos is None:
            return AgentResponse(
                message=f"No open {symbol} position to release margin from.",
                intent="release",
                state=self.snapshot(),
            )
        target = self.guardrails.liq_target_dist_pct
        if target <= self.guardrails.liq_warn_pct:
            return AgentResponse(
                message=(
                    f"Release suspended: the de-risk target ({target * 100:.0f}%) "
                    f"must stay above the watch zone "
                    f"({self.guardrails.liq_warn_pct * 100:.0f}%) so releasing margin "
                    f"can never push {symbol} into a warning state."
                ),
                intent="release",
                state=self.snapshot(),
            )
        amount = self._releasable_margin(pos)
        if amount is None:
            return AgentResponse(
                message=(
                    f"{symbol} currently holds no excess margin to release — its "
                    "margin is exactly what keeps it at the de-risk headroom right now."
                ),
                intent="release",
                state=self.snapshot(),
            )
        base = symbol.replace("USDT", "")
        mark = pos.mark_price or pos.entry_price
        prop = TradeProposal(
            id=uuid.uuid4().hex[:10],
            symbol=symbol,
            side=Side.BUY,
            kind=ProposalKind.TRANSFER,
            transfer_kind=TransferKind.RELEASE_MARGIN,
            est_value_usdt=amount,
            est_quantity=0.0,
            est_price=mark,
            reason=(
                f"{pos.side.value} {symbol} sits {pos.distance_pct * 100:.1f}% from "
                f"liquidation with ${pos.margin_usdt:,.2f} margin. The margin above "
                f"what keeps {target * 100:.0f}% headroom is idle insurance — "
                f"release ${amount:,.2f} back to spot and the position keeps "
                f"≥ {target * 100:.0f}% headroom."
            ),
        )
        self.proposals[prop.id] = prop
        self.log(
            AuditLevel.INFO,
            "release_proposed",
            f"{symbol} margin ${amount:,.2f} -> spot",
            prop.id,
        )
        if self._auto_active():
            self.log(
                AuditLevel.ACTION,
                "auto_mode",
                f"automatic mode released {symbol} excess margin to spot",
                prop.id,
            )
            resp = self.decide(prop.id, True)
            return AgentResponse(
                message="⚡ " + resp.message + " (automatic mode)",
                intent="release",
                state=self.snapshot(),
            )
        return AgentResponse(
            message=(
                f"📝 Release ${amount:,.2f} excess margin on {symbol} back to spot "
                f"(keeps ≥ {target * 100:.0f}% liquidation headroom)? Approve to execute."
            ),
            intent="release",
            state=self.snapshot(),
            proposals=[prop.to_dict()],
        )

    def set_tp_sl(self, symbol: str, tp: float = 0.0, sl: float = 0.0) -> AgentResponse:
        symbol = symbol.upper()
        ok = getattr(self.adapter, "set_tp_sl", None)
        if ok is None:
            return AgentResponse(
                message="Live adapter doesn't support TP/SL in sim.",
                intent="tpsl",
                state=self.snapshot(),
            )
        if not ok(symbol, tp, sl):
            return AgentResponse(
                message=f"No open {symbol} position.",
                intent="tpsl",
                state=self.snapshot(),
            )
        pos = next(
            (x for x in self.account().futures_positions if x.symbol == symbol), None
        )
        bits = []
        if tp:
            bits.append(f"TP ${tp:,.0f}")
        if sl:
            bits.append(f"SL ${sl:,.0f}")
        self.log(AuditLevel.ACTION, "tp_sl_set", f"{symbol} {' '.join(bits)}")
        return AgentResponse(
            message=f"🎯 {symbol}: armed "
            + (" + ".join(bits) if bits else "nothing — cleared")
            + (f". Current mark ${pos.mark_price:,.0f}." if pos else ""),
            intent="tpsl",
            state=self.snapshot(),
        )

    async def scan_tp_sl(self) -> list[AgentResponse]:
        """Periodic TP/SL sweep.

        Arming a take-profit or stop-loss is *pre-authorization* to exit, so
        when a level is crossed the position is closed right away in the
        simulator — no extra approval round-trip, exactly like an exchange-side
        stop order. Protective exits are never calendar-capped — the guardian
        must always be allowed to exit.
        """
        await self.refresh()
        state = self.account()
        responses: list[AgentResponse] = []
        for pos in state.futures_positions:
            mark = pos.mark_price
            hit_tp = (
                pos.take_profit_price > 0
                and not pos.tp_hit
                and (
                    mark >= pos.take_profit_price
                    if pos.side == PositionSide.LONG
                    else mark <= pos.take_profit_price
                )
            )
            hit_sl = (
                pos.stop_loss_price > 0
                and not pos.sl_hit
                and (
                    mark <= pos.stop_loss_price
                    if pos.side == PositionSide.LONG
                    else mark >= pos.stop_loss_price
                )
            )
            if not (hit_tp or hit_sl):
                continue
            kind_word = "stop-loss" if hit_sl else "take-profit"
            level = pos.stop_loss_price if hit_sl else pos.take_profit_price
            # don't re-fire while a close attempt is already pending/executing
            if any(
                pp.symbol == pos.symbol and pp.status == ProposalStatus.PENDING
                for pp in self.proposals.values()
            ):
                continue
            prop = TradeProposal(
                id=uuid.uuid4().hex[:10],
                symbol=pos.symbol,
                side=Side.BUY if pos.side == PositionSide.SHORT else Side.SELL,
                kind=ProposalKind.CLOSE,
                close_all=True,
                keep_margin=False,
                est_value_usdt=round(pos.quantity * mark, 2),
                est_quantity=round(pos.quantity, 8),
                est_price=mark,
                reason=f"{kind_word.title()} hit ({pos.symbol} @ ${mark:,.0f} vs level ${level:,.0f}) — auto-close",
            )
            self.log(AuditLevel.ACTION, "tp_sl_fired", prop.reason, prop.id)
            try:
                result = self.adapter.execute(prop)
            except Exception as exc:
                self.log(AuditLevel.ERROR, "tp_sl_execution_error", str(exc), prop.id)
                responses.append(
                    AgentResponse(
                        message=f"❌ {kind_word.title()} crossed on {pos.symbol} but the close failed: {exc}",
                        intent="tpsl",
                        state=self.snapshot(),
                    )
                )
                continue
            if result.ok:
                # mark so a (fresh) re-armed level on a new position is tracked cleanly
                pos.tp_hit, pos.sl_hit = True, True
                self.log(
                    AuditLevel.ACTION,
                    "executed",
                    f"CLOSE {pos.symbol} {result.executed_value_usdt:,.2f}",
                    prop.id,
                )
                responses.append(
                    AgentResponse(
                        message=f"✅ {kind_word.title()} executed on {pos.symbol} — closed ≈ ${result.executed_value_usdt:,.2f} @ {result.executed_price:,.2f}.",
                        intent="tpsl",
                        state=self.snapshot(),
                    )
                )
            else:
                self.log(
                    AuditLevel.WARN, "tp_sl_execution_rejected", result.message, prop.id
                )
                responses.append(
                    AgentResponse(
                        message=f"⚠️ {pos.symbol} {kind_word} crossed but the close was declined: {result.message}",
                        intent="tpsl",
                        state=self.snapshot(),
                    )
                )
        return responses

    # ------------------------------------------------------ guardrail editing
    def guardrail_status(self, state: Optional[AccountState] = None) -> dict:
        """Current editable guardrail profile."""
        g = self.guardrails
        return {
            "values": {
                "liq_warn_pct": g.liq_warn_pct,
                "liq_danger_pct": g.liq_danger_pct,
                "liq_target_dist_pct": g.liq_target_dist_pct,
            },
            "symbol_allowlist": list(g.symbol_allowlist),
        }

    def update_guardrails(self, updates: dict[str, Any]) -> AgentResponse:
        """Update the runtime-editable risk profile.

        Every numeric limit is sanity-bounded (RAIL_BOUNDS) so the agent can
        never be driven into a nonsense value, and edits persist across
        restarts. There is deliberately no per-day trade budget here: the
        guardian only ever protects, so its actions must never be capped by a
        calendar window.
        """
        g = self.guardrails
        changed: list[str] = []
        refused: list[str] = []

        for k in EDITABLE_RAIL_KEYS:
            if k not in updates:
                continue
            try:
                val = float(updates[k])
            except (TypeError, ValueError):
                refused.append(f"{k}: {updates[k]!r} is not a number")
                continue
            if not math.isfinite(val):
                refused.append(f"{k}: {val} is not a finite number")
                continue
            lo, hi = RAIL_BOUNDS.get(k, (float("-inf"), float("inf")))
            if not (lo <= val <= hi):
                if math.isfinite(hi):
                    refused.append(
                        f"{k}: {self._fmt_val(k, val)} must stay within "
                        f"{self._fmt_val(k, lo)} – {self._fmt_val(k, hi)}"
                    )
                else:
                    refused.append(
                        f"{k}: {self._fmt_val(k, val)} must be >= {self._fmt_val(k, lo)}"
                    )
                continue
            cur = getattr(g, k)
            if abs(val - float(cur)) < 1e-12:
                continue  # no-op edit
            setattr(g, k, val)
            changed.append(f"{k}={self._fmt_val(k, val)}")

        # --- symbol allowlist (hygiene list, not a risk ceiling) -----------
        if isinstance(updates.get("symbol_allowlist"), list):
            clean = [
                str(s).strip().upper()
                for s in updates["symbol_allowlist"]
                if str(s).strip()
            ]
            if clean != list(g.symbol_allowlist):
                g.symbol_allowlist = clean
                changed.append("symbol_allowlist=…")

        if changed:
            self._save_guardrail_profile()
            self.log(AuditLevel.ACTION, "guardrails_updated", ", ".join(changed))

        parts: list[str] = []
        if changed:
            parts.append("✅ " + ", ".join(changed))
        if refused:
            parts.append("⛔ Refused " + str(len(refused)) + " edit(s): " + "; ".join(refused))
        return AgentResponse(
            message="\n".join(parts) if parts else "No changes.",
            intent="guardrails",
            state=self.snapshot(),
        )

    @staticmethod
    def _fmt_val(key: str, v: float) -> str:
        """Human-friendly formatting for a guardrail value (messages/audit)."""
        if key.endswith("_pct"):
            return f"{v * 100:g}%"
        return f"{v:g}"

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _price_for(state: AccountState, symbol: str) -> Optional[float]:
        for p in state.positions:
            if p.symbol == symbol:
                return p.price
        for p in state.futures_positions:
            if p.symbol == symbol:
                return p.mark_price
        return None
