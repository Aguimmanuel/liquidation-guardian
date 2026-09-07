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
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from app.adapters.base import ExecutionAdapter
from app.agent.guardrails import (
    GuardrailConfig,
    check_cash_sufficient,
    check_daily_trade_count,
    check_futures_position_exists,
    check_max_trade_size,
    check_min_trade_value,
    check_symbol_allowed,
    check_transfer_sufficient,
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
    required_cut,
    required_margin,
)
from app.config import AppConfig
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


AUTO_MODE_NOTICE = (
    "⚠️ Automatic agent — read before enabling.\n\n"
    "When automatic mode is on, the agent acts on your behalf: it can reduce risk, "
    "add margin, close positions, and execute the conditions you arm, without asking "
    "you to approve each step. Guardrails still apply, but guardrails cannot guarantee "
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
        self._load_guardrail_lock()

    # --------------------------------------------- guardrail lock persistence
    def _lock_path(self):
        from pathlib import Path

        root = getattr(self.config, "state_file", "data/sim_state.json")
        base = getattr(self.config, "guardrail_state_file", "data/guardrail_state.json")
        # keep it relative to the same folder as the sim state (repo root at runtime)
        path = Path(base)
        if not path.is_absolute():
            path = Path.cwd() / base
        return path

    def _save_guardrail_lock(self) -> None:
        g = self.guardrails
        try:
            payload = {
                "daily_trades_locked": bool(g.daily_trades_locked),
                "daily_lock_at": g.daily_lock_at.isoformat()
                if g.daily_lock_at
                else None,
            }
            path = self._lock_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload))
        except Exception:
            pass  # persistence is best-effort; never break the agent for it

    def _load_guardrail_lock(self) -> None:
        g = self.guardrails
        try:
            path = self._lock_path()
            if not path.exists():
                return
            payload = json.loads(path.read_text())
            if payload.get("daily_trades_locked") and payload.get("daily_lock_at"):
                at = datetime.fromisoformat(payload["daily_lock_at"])
                # a lock that already ran its 24h no longer applies
                if (datetime.now(timezone.utc) - at).total_seconds() < 86400:
                    g.daily_trades_locked = True
                    g.daily_lock_at = at
                else:
                    path.unlink(missing_ok=True)
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
        return {
            "mode": self.config.mode,
            "config": self.config.to_dict(),
            "guardrails": self.guardrails.to_dict(),
            "account": state.to_dict(),
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
        intent_raw = await parse_intent_llm(self.config, message, {})
        intent = intent_raw.get("intent", "chat")
        await self.refresh()

        if intent in ("check", "risk"):
            return await self._do_risk_report()
        if intent in ("protect", "de_risk"):
            return await self._do_de_risk()
        if intent == "status":
            return await self._do_status()
        if intent == "market":
            return await self.market_analysis(intent_raw.get("symbol"))
        if intent == "add_condition":
            return await self._do_add_condition(intent_raw.get("condition", {}))
        return await self._do_chat(message)

    async def check_conditions(self) -> list[AgentResponse]:
        """Evaluate armed conditions against live prices (called periodically)."""
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
            if hit:
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
            f"  Total: ${state.total_value_usdt:,.2f} · drawdown {drawdown:.1f}% · "
            f"trades today {state.trade_count_today}/{self.guardrails.max_daily_trades}"
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
            # prefer the reduce (deleverage, no extra capital) and only add
            # margin when a reduce is not possible — never stack both.
            chosen = [p for p in new_pending if p.kind == ProposalKind.TRADE]
            if not chosen:
                chosen = new_pending
            chosen_ids = {p.id for p in chosen}
            self.log(
                AuditLevel.ACTION,
                "auto_mode",
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
                    "⚡ Automatic mode — the guardian de-risked your position(s) without waiting "
                    "for approval:\n"
                    + "\n".join(done)
                    + "\n\nEvery action cleared the guardrails "
                    "and is in the audit trail. Switch to Manual to go back to approving each step."
                ),
                intent="de_risk",
                state=self.snapshot(),
            )
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
            check_min_trade_value(prop.est_value_usdt, self.guardrails),
            check_daily_trade_count(state.trade_count_today, self.guardrails),
        ):
            if not result.ok:
                reasons.append(result.reason)
        if prop.kind == ProposalKind.TRANSFER:
            reasons += [
                r.reason
                for r in [
                    check_transfer_sufficient(
                        state.cash_usdt, prop.est_value_usdt, self.guardrails
                    )
                ]
                if not r.ok
            ]
        elif prop.side == Side.SELL:
            # REDUCE is de-risking (closing exposure) — exempt from the order-size
            # cap, exactly like sells in a risk-reduction flow. The notional being
            # closed is the user's own position, and the action shrinks risk.
            reasons += []
        else:
            reasons += [
                r.reason
                for r in [
                    check_max_trade_size(
                        prop.est_value_usdt, state.total_value_usdt, self.guardrails
                    )
                ]
                if not r.ok
            ]
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
        total = state.total_value_usdt
        prop = TradeProposal(
            id=uuid.uuid4().hex[:10],
            symbol=cond.symbol,
            side=cond.side,
            kind=ProposalKind.TRADE,
            est_value_usdt=min(cond.amount_usdt, total * self.guardrails.max_trade_pct),
            est_quantity=cond.amount_usdt / price if price else 0.0,
            est_price=price,
            reason=f"Condition triggered: {cond.symbol} {cond.op.value} {cond.price:,.2f}",
            created_at=utcnow(),
        )
        from app.agent.guardrails import evaluate_proposal

        allowed, results = evaluate_proposal(prop, state, self.guardrails)
        if not allowed:
            reasons = [r.reason for r in results if not r.ok]
            self.log(
                AuditLevel.GUARDRAIL, "condition_blocked", "; ".join(reasons), prop.id
            )
            return AgentResponse(
                message=f"🚨 Condition fired on {cond.symbol} @ {price:,.2f}, but guardrails blocked the trade: "
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
        base = (
            f"Here's where we stand: total ${state.total_value_usdt:,.2f}, "
            f"{len(state.futures_positions)} futures position(s) open, "
            f"{sum(1 for p in state.futures_positions if p.risk.value == 'DANGER')} in the danger zone. "
            "Try “risk report”, “protect my positions”, or “status”."
        )
        return AgentResponse(message=base, intent="chat", state=self.snapshot())

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
        _note(
            margin_usdt >= self.guardrails.min_trade_value_usdt,
            f"margin ${margin_usdt:,.0f} is below the ${self.guardrails.min_trade_value_usdt:,.0f} minimum",
        )
        _note(
            0 < leverage <= self.guardrails.max_leverage,
            f"leverage must be between 1x and {self.guardrails.max_leverage:.0f}x",
        )
        _note(
            state.cash_usdt >= margin_usdt,
            f"not enough cash: need ${margin_usdt:,.0f}, have ${state.cash_usdt:,.0f}",
        )
        _note(
            check_daily_trade_count(state.trade_count_today, self.guardrails).ok,
            f"daily trade budget used ({state.trade_count_today}/{self.guardrails.max_daily_trades})",
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
        # closing reduces risk — exempt from the size cap but respects budget
        reasons = []
        if not check_daily_trade_count(state.trade_count_today, self.guardrails).ok:
            reasons.append(
                f"daily trade budget used ({state.trade_count_today}/{self.guardrails.max_daily_trades})"
            )
        if reasons:
            self.log(AuditLevel.GUARDRAIL, "close_blocked", "; ".join(reasons), prop.id)
            return AgentResponse(
                message="⛔ Close blocked: " + " ".join(reasons),
                intent="close",
                state=self.snapshot(),
            )
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
        stop order. Protective exits are exempt from the daily-trade budget.
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
        """Current editable guardrails + 24h budget info."""
        g = self.guardrails
        if state is None:
            state = self.account()
        # auto-expire a stale 24h lock whenever the status is read
        if g.daily_trades_locked and g.daily_lock_at:
            if (datetime.now(timezone.utc) - g.daily_lock_at).total_seconds() >= 86400:
                g.daily_trades_locked = False
                g.daily_lock_at = None
                try:
                    self._lock_path().unlink(missing_ok=True)
                except Exception:
                    pass
        window_start = state.daily_window_start
        remaining = 0.0
        if window_start:
            remaining = max(
                0.0,
                86400.0 - (datetime.now(timezone.utc) - window_start).total_seconds(),
            )
        lock_remaining = 0
        if g.daily_trades_locked and g.daily_lock_at:
            lock_remaining = max(
                0,
                int(
                    86400
                    - (datetime.now(timezone.utc) - g.daily_lock_at).total_seconds()
                ),
            )
        return {
            "values": {
                "liq_warn_pct": g.liq_warn_pct,
                "liq_danger_pct": g.liq_danger_pct,
                "liq_target_dist_pct": g.liq_target_dist_pct,
                "max_trade_pct": g.max_trade_pct,
                "min_trade_value_usdt": g.min_trade_value_usdt,
                "cooldown_seconds": g.cooldown_seconds,
                "max_daily_trades": g.max_daily_trades,
                "max_leverage": g.max_leverage,
                "symbol_allowlist": g.symbol_allowlist,
            },
            "locked": g.daily_trades_locked,
            "lock_until": g.daily_lock_at.isoformat() if g.daily_lock_at else None,
            "lock_remaining_seconds": lock_remaining,
            "budget": {
                "trades_used": state.trade_count_today,
                "trades_cap": g.max_daily_trades,
                "window_remaining_seconds": int(remaining),
                "window_start": window_start.isoformat() if window_start else None,
            },
        }

    def update_guardrails(self, updates: dict[str, Any]) -> AgentResponse:
        """Update guardrail values.

        The daily trade cap can be *locked*: once locked it stays locked until
        the 24-hour window has fully elapsed — there is no early unlock, by
        design, because the whole point is a commitment you can't talk yourself
        out of. (A stale lock is auto-expired when read after its 24h.)
        """
        g = self.guardrails
        gstate = self.guardrail_status()

        if gstate["locked"] and "daily_trades_locked" in updates:
            want_lock = bool(updates["daily_trades_locked"])
            if want_lock:
                return AgentResponse(
                    message="🔒 The daily trade budget is already locked for this 24-hour window. "
                    "It releases automatically when the window ends.",
                    intent="guardrails",
                    state=self.snapshot(),
                )
            return AgentResponse(
                message="🔒 The daily trade budget is locked for this 24-hour window and cannot be "
                "unlocked early — that lock is the whole point. It releases automatically "
                "when the window ends.",
                intent="guardrails",
                state=self.snapshot(),
            )
        if gstate["locked"] and "max_daily_trades" in updates:
            return AgentResponse(
                message="🔒 Daily trade budget is locked for this 24-hour window — the cap and the "
                "lock release together when the window ends.",
                intent="guardrails",
                state=self.snapshot(),
            )

        changed = []
        numeric_keys = [
            "liq_warn_pct",
            "liq_danger_pct",
            "liq_target_dist_pct",
            "max_trade_pct",
            "min_trade_value_usdt",
            "cooldown_seconds",
            "max_daily_trades",
            "max_leverage",
        ]
        for k in numeric_keys:
            if k in updates:
                val = float(updates[k])
                setattr(g, k, val)
                changed.append(f"{k}={val}")
        if "daily_trades_locked" in updates and bool(updates["daily_trades_locked"]):
            if not gstate["locked"]:
                g.daily_trades_locked = True
                g.daily_lock_at = datetime.now(timezone.utc)
                changed.append("daily budget LOCKED for 24h")
                self._save_guardrail_lock()
        self.log(AuditLevel.ACTION, "guardrails_updated", ", ".join(changed))
        return AgentResponse(
            message="✅ Guardrails updated: "
            + (", ".join(changed) if changed else "no changes"),
            intent="guardrails",
            state=self.snapshot(),
        )

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
