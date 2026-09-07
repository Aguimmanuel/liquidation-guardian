"""Feature tests: dynamic leverage opens, TP/SL auto execution, always-on
guardian monitoring, two-way fund movement (futures -> spot), and guardrail
editing. The old per-day trade budget + 24h profile lock were removed on
purpose: a calendar cap must never freeze protective actions.

All tests run fully offline behind a fake market feed.
"""
from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from app.adapters.sim import SimAdapter
from app.agent.orchestrator import Orchestrator
from app.config import AppConfig, GuardrailConfig
from app.models import Ticker, utcnow


def run(coro):
    return asyncio.run(coro)


class FakeMarket:
    """Minimal async market feed whose prices we control in tests."""

    def __init__(self) -> None:
        self.prices = {s: 1000.0 for s in ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]}
        self.change = 0.0

    def _ticker(self, symbol: str) -> Ticker:
        return Ticker(symbol=symbol, price=self.prices.get(symbol, 100.0), change_24h_pct=self.change)

    async def get_ticker(self, symbol: str) -> Ticker:
        return self._ticker(symbol)

    async def get_tickers(self, symbols: list[str]) -> dict[str, Ticker]:
        return {s: self._ticker(s) for s in symbols}

    async def get_watch_tickers(self, symbols: list[str]) -> dict[str, Ticker]:
        return {s: self._ticker(s) for s in symbols}

    async def get_price(self, symbol: str) -> float:
        return self.prices.get(symbol, 100.0)

    async def get_klines(self, symbol: str, interval: str = "1h", limit: int = 96) -> list[dict]:
        return [{"close": self.prices.get(symbol, 100.0)} for _ in range(limit)]

    async def trend_summary(self, symbol: str, bars: int = 96) -> dict:
        price = self.prices.get(symbol, 100.0)
        return {
            "symbol": symbol,
            "price": price,
            "change_24h_pct": self.change,
            "rsi14": 55.0,
            "ma7": price,
            "ma25": price,
            "signal": "BULLISH",
            "note": "",
        }

    async def analyze_symbol(self, symbol: str, bars: int = 168) -> dict:
        price = self.prices.get(symbol, 100.0)
        return {
            "symbol": symbol,
            "base": symbol.replace("USDT", ""),
            "price": price,
            "change_24h_pct": self.change,
            "change_7d_pct": self.change,
            "high_24h": price * 1.02,
            "low_24h": price * 0.98,
            "high_7d": price * 1.05,
            "low_7d": price * 0.95,
            "rsi14": 55.0,
            "ma7": price * 1.01,
            "ma25": price * 0.99,
            "ma7_slope_pct": 0.1,
            "atr14_pct": 0.8,
            "support": price * 0.98,
            "support_scope": "24h",
            "resistance": price * 1.02,
            "resistance_scope": "24h",
            "regime": "neutral",
            "signal": "BULLISH",
        }

    async def aclose(self) -> None:
        pass


@pytest.fixture()
def stack(tmp_path):
    """A ready orchestrator + fake market + clean sim state file."""
    state_file = str(tmp_path / "state.json")
    cfg = AppConfig(
        mode="sim",
        default_initial_cash=10_000.0,
        market_ttl_seconds=1,
        ticker_ttl_seconds=1,
        state_file=state_file,
        guardrail_state_file=str(tmp_path / "guardrail_state.json"),
    )
    guard = GuardrailConfig()
    market = FakeMarket()
    sim = SimAdapter(cfg, guard, market, state_file=state_file)
    sim.reset(cfg.default_initial_cash)
    orch = Orchestrator(cfg, guard, sim, market)
    return orch, market, sim


def _open_long(orch, symbol="BTCUSDT", margin=500.0, leverage=10.0):
    resp = run(orch.queue_open(symbol, "LONG", margin, leverage))
    assert resp.proposals, resp.message
    decided = orch.decide(resp.proposals[0]["id"], True)
    assert "Opened LONG" in decided.message, decided.message
    return decided


def test_open_approve_executes_long(stack):
    orch, market, sim = stack
    _open_long(orch, margin=500.0, leverage=10.0)
    acct = orch.account()
    pos = next(p for p in acct.futures_positions if p.symbol == "BTCUSDT")
    assert pos.side.value == "LONG"
    assert pos.quantity == pytest.approx(5000.0 / 1000.0)     # notional / entry
    assert pos.margin_usdt == pytest.approx(500.0)
    assert acct.cash_usdt == pytest.approx(9500.0)            # margin taken from spot


def test_open_dynamic_leverage_matters(stack):
    orch, market, sim = stack
    _open_long(orch, margin=500.0, leverage=25.0)
    pos = next(p for p in orch.account().futures_positions if p.symbol == "BTCUSDT")
    assert pos.notional_usdt == pytest.approx(500 * 25.0)
    assert pos.leverage == pytest.approx(25.0)


def test_open_blocked_by_leverage_cap_and_cash(stack):
    orch, market, sim = stack
    blocked = run(orch.queue_open("BTCUSDT", "LONG", 500.0, 999.0))
    assert "blocked" in blocked.message.lower()
    broke = run(orch.queue_open("ETHUSDT", "LONG", 99_999.0, 10.0))
    assert "blocked" in broke.message.lower()


def test_opposite_side_open_blocked(stack):
    orch, market, sim = stack
    _open_long(orch)
    blocked = run(orch.queue_open("BTCUSDT", "SHORT", 300.0, 10.0))
    assert "opposite" in blocked.message.lower()


def test_same_side_open_averages_entry(stack):
    orch, market, sim = stack
    _open_long(orch, margin=500.0, leverage=10.0)
    market.prices["BTCUSDT"] = 1100.0
    _open_long(orch, margin=500.0, leverage=10.0)
    acct = orch.account()
    pos = next(p for p in acct.futures_positions if p.symbol == "BTCUSDT")
    # still one position; averaged entry between 1000 and 1100
    assert len([p for p in acct.futures_positions if p.symbol == "BTCUSDT"]) == 1
    assert pos.entry_price == pytest.approx((1000 * 5 + 1100 * (5000 / 1100)) / (5 + 5000 / 1100))
    assert pos.margin_usdt == pytest.approx(1000.0)


def test_stop_loss_auto_closes_long(stack):
    orch, market, sim = stack
    _open_long(orch, margin=500.0, leverage=10.0)          # entry at 1000, qty 5
    orch.set_tp_sl("BTCUSDT", 1030.0, 980.0)
    market.prices["BTCUSDT"] = 975.0                        # through the stop
    fired = run(orch.scan_tp_sl())
    assert len(fired) == 1
    acct = orch.account()
    assert not [p for p in acct.futures_positions if p.symbol == "BTCUSDT"]
    assert any(a.event == "tp_sl_fired" for a in orch.audit)
    # realized loss ≈ (975 - 1000) * 5, minus fees
    assert acct.realized_pnl_usdt < 0


def test_take_profit_auto_closes_long(stack):
    orch, market, sim = stack
    _open_long(orch, margin=500.0, leverage=10.0)
    orch.set_tp_sl("BTCUSDT", 1030.0, 0.0)
    market.prices["BTCUSDT"] = 1040.0                       # through the target
    fired = run(orch.scan_tp_sl())
    assert len(fired) == 1
    acct = orch.account()
    assert not [p for p in acct.futures_positions if p.symbol == "BTCUSDT"]
    assert acct.realized_pnl_usdt > 0


def test_tp_sl_short_side_logic(stack):
    orch, market, sim = stack
    resp = run(orch.queue_open("ETHUSDT", "SHORT", 400.0, 10.0))
    orch.decide(resp.proposals[0]["id"], True)
    orch.set_tp_sl("ETHUSDT", 980.0, 1030.0)  # short: TP below, SL above
    market.prices["ETHUSDT"] = 1050.0                       # through the short stop
    fired = run(orch.scan_tp_sl())
    assert fired, "short stop-loss should fire"
    acct = orch.account()
    assert not [p for p in acct.futures_positions if p.symbol == "ETHUSDT"]


def test_funds_wallet_roundtrip_returns_to_spot(stack):
    """Closing a position parks its margin + PnL in the futures wallet; the
    wallet can now be moved back to spot (it is idle money, never part of an
    open position's margin)."""
    orch, market, sim = stack
    _open_long(orch, margin=500.0, leverage=10.0)          # spot cash 9500, wallet 0
    assert orch.account().futures_wallet_usdt == pytest.approx(0.0)
    close = run(orch.queue_close("BTCUSDT", close_all=True))
    orch.decide(close.proposals[0]["id"], True)
    acct1 = orch.account()
    assert not acct1.futures_positions
    assert acct1.futures_wallet_usdt == pytest.approx(500.0, abs=0.05)
    ret = run(orch.return_futures_wallet_to_spot())
    assert ret.proposals and ret.proposals[0]["transfer_kind"] == "RETURN_WALLET"
    orch.decide(ret.proposals[0]["id"], True)
    acct2 = orch.account()
    assert acct2.futures_wallet_usdt == pytest.approx(0.0)
    assert acct2.cash_usdt == pytest.approx(10000.0, abs=0.05)
    assert any(a.event == "return_proposed" for a in orch.audit)


def test_release_excess_margin_back_to_spot(stack):
    """Margin added by a de-risk can be pulled back off an open position once
    it is no longer needed, bounded so the position keeps its de-risk
    headroom and stays in the OK zone."""
    from app.models import TransferKind

    orch, market, sim = stack
    b = run(orch.queue_open("BTCUSDT", "LONG", 500.0, 10.0))
    orch.decide(b.proposals[0]["id"], True)                # WATCH ~9.5% from liq
    # drive the price down so the position lands in DANGER -> monitor queues
    # the two-option de-risk plan (reduce OR add margin)
    market.prices["BTCUSDT"] = 920.0
    run(orch.monitor_positions())
    add = [p for p in orch.proposals.values() if p.status.value == "PENDING"
           and p.kind.value == "TRANSFER"]
    assert add, "DANGER must queue an add-margin (Option B) transfer"
    orch.decide(add[0].id, True)                           # top margin up to headroom
    pos = next(p for p in orch.account().futures_positions if p.symbol == "BTCUSDT")
    assert pos.risk.value == "OK"
    margin_after_topup = pos.margin_usdt                   # capture as a float
    # price recovers -> that margin is now excess and should be releaseable
    market.prices["BTCUSDT"] = 1000.0
    run(orch.refresh())
    snap = orch.snapshot()
    shown = next(p for p in snap["account"]["futures_positions"] if p["symbol"] == "BTCUSDT")
    assert shown["releasable_margin_usdt"] is not None and shown["releasable_margin_usdt"] > 100.0

    cash_before = snap["account"]["cash_usdt"]
    ret = run(orch.release_margin_to_spot("BTCUSDT"))
    assert ret.proposals and ret.proposals[0]["transfer_kind"] == TransferKind.RELEASE_MARGIN.value
    released = ret.proposals[0]["est_value_usdt"]
    orch.decide(ret.proposals[0]["id"], True)
    acct2 = orch.account()
    pos2 = next(p for p in acct2.futures_positions if p.symbol == "BTCUSDT")
    assert acct2.cash_usdt == pytest.approx(cash_before + released, abs=0.05)
    assert acct2.futures_wallet_usdt == pytest.approx(0.0)
    assert pos2.risk.value == "OK"                          # still protected
    assert pos2.margin_usdt < margin_after_topup            # some margin came home
    assert any(a.event == "release_proposed" for a in orch.audit)


def test_release_refused_when_nothing_excess_and_target_misconfigured(stack):
    """Release is bounded: nothing to release when the position still needs
    margin, and the action is refused entirely if the de-risk target is not
    above the watch zone (it could otherwise push a position into warning)."""
    orch, market, sim = stack
    _open_long(orch, margin=500.0, leverage=10.0)           # WATCH: needs MORE margin
    ret = run(orch.release_margin_to_spot("BTCUSDT"))
    assert not ret.proposals
    assert "no excess margin" in ret.message.lower()
    # a nonsense config (target <= watch zone) must suspend releases outright
    _open_long(orch, symbol="ETHUSDT", margin=500.0, leverage=10.0)
    orch.update_guardrails({"liq_target_dist_pct": 0.08})   # below watch 0.10
    ret2 = run(orch.release_margin_to_spot("ETHUSDT"))
    assert not ret2.proposals
    assert "de-risk target" in ret2.message


def test_guardrail_edits_surface_in_snapshot(stack):
    orch, market, sim = stack
    orch.update_guardrails({"liq_danger_pct": 0.05, "max_leverage": 20})
    gs = orch.guardrail_status()
    assert gs["values"]["liq_danger_pct"] == pytest.approx(0.05)
    assert gs["values"]["max_leverage"] == pytest.approx(20.0)
    snap = orch.snapshot()
    assert snap["guardrails"]["max_leverage"] == 20.0


def test_market_analysis_prompts_then_reports(stack):
    orch, market, sim = stack
    # without a symbol -> prompt offering coins
    prompt = run(orch.market_analysis())
    assert prompt.intent == "market"
    assert prompt.options and "BTCUSDT" in prompt.options
    assert "Which coin" in prompt.message

    # with a symbol -> deep single-coin report
    deep = run(orch.market_analysis("BTCUSDT"))
    assert "market analysis" in deep.message.lower()
    for needle in ("Trend", "RSI", "Volatility", "support", "resistance", "Plan", "BULLISH"):
        assert needle in deep.message, needle

    # chat intents route symbol through
    parsed = run(orch.handle_message("analyze btc"))
    assert parsed.intent == "market"
    assert "BTC" in parsed.message




# ---------------------------------------------------------------------------
# Automatic agent mode
# ---------------------------------------------------------------------------


def test_manual_mode_queues_for_approval(stack):
    orch, market, sim = stack
    assert orch.agent_mode == "manual"
    resp = run(orch.queue_open("BTCUSDT", "LONG", 500.0, 10.0))
    assert resp.proposals, "manual mode must queue a proposal"
    assert not orch.account().futures_positions, "nothing should execute before approval"


def test_auto_mode_requires_consent(stack):
    orch, market, sim = stack
    denied = orch.set_agent_mode("auto", consent=False)
    assert "consent" in denied.message.lower() or "read before enabling" in denied.message.lower()
    assert orch.agent_mode == "manual"
    assert orch.snapshot()["agent"]["mode"] == "manual"


def test_auto_mode_executes_open_without_approval(stack):
    orch, market, sim = stack
    resp = orch.set_agent_mode("auto", consent=True)
    assert orch.agent_mode == "auto"
    assert orch._auto_active()
    assert "Automatic mode is ON" in resp.message
    # opening a trade executes immediately (no pending queue)
    out = run(orch.queue_open("BTCUSDT", "LONG", 500.0, 10.0))
    assert not out.proposals, "auto mode should not leave pending proposals"
    pos = next(p for p in orch.account().futures_positions if p.symbol == "BTCUSDT")
    assert pos.side.value == "LONG"
    assert orch.account().cash_usdt == pytest.approx(9500.0)
    # going back to manual revokes consent
    back = orch.set_agent_mode("manual", consent=False)
    assert orch.agent_mode == "manual" and not orch._auto_active()


def test_auto_mode_executes_de_risk_on_danger(stack):
    orch, market, sim = stack
    opened = run(orch.queue_open("BTCUSDT", "LONG", 600.0, 20.0))
    orch.decide(opened.proposals[0]["id"], True)
    pos = next(p for p in orch.account().futures_positions if p.symbol == "BTCUSDT")
    assert pos.risk.value == "DANGER"  # 20x puts liquidation ~5% away
    orch.set_agent_mode("auto", consent=True)
    out = run(orch.handle_message("protect my positions"))
    assert "Automatic mode" in out.message
    assert not [p for p in orch.proposals.values() if p.status.value == "PENDING"]
    assert any(a.event == "auto_mode" for a in orch.audit)
    healed = next(p for p in orch.account().futures_positions if p.symbol == "BTCUSDT")
    assert healed.risk.value != "DANGER" or healed.quantity < pos.quantity


def test_daily_budget_and_24h_lock_are_gone(stack):
    """The per-day trade budget and the 24h profile lock were removed: a
    calendar cap must never freeze protective actions, and stale lock requests
    are inert."""
    orch, market, sim = stack
    assert not hasattr(orch.guardrails, "max_daily_trades")
    assert not hasattr(orch.guardrails, "daily_trades_locked")
    gs = orch.guardrail_status()
    for stale in ("budget", "locked", "lock_until", "lock_remaining_seconds"):
        assert stale not in gs, stale
    assert "max_daily_trades" not in gs["values"]
    # stale requests (cap / lock) are inert, never fatal, never engage anything
    resp = orch.update_guardrails({"max_daily_trades": 3, "daily_trades_locked": True})
    assert "No changes" in resp.message
    assert not any(a.event == "guardrails_updated" for a in orch.audit)
    # and a DANGER position still gets protected: no cap, no gate in the way
    _open_danger_long(orch)
    orch.set_agent_mode("auto", consent=True)
    responses = run(orch.monitor_positions())
    assert responses, "guardian must still de-risk without any daily budget"
    healed = next(p for p in orch.account().futures_positions if p.symbol == "BTCUSDT")
    assert healed.risk.value != "DANGER"


def test_guardrail_edits_reject_out_of_bounds(stack):
    orch, market, sim = stack
    bad = orch.update_guardrails({"max_leverage": 9999.0})
    assert "must stay within" in bad.message
    assert orch.guardrails.max_leverage == 50.0
    nan = orch.update_guardrails({"liq_danger_pct": float("nan")})
    assert "not a finite number" in nan.message


def test_profile_values_persist_across_restart(stack, tmp_path):
    """Edits survive a restart (env defaults no longer win)."""
    orch, market, sim = stack
    orch.update_guardrails({"liq_danger_pct": 0.05, "max_leverage": 25.0, "cooldown_seconds": 1800.0})

    from app.agent.orchestrator import Orchestrator
    from app.config import GuardrailConfig

    g2 = GuardrailConfig()  # fresh env defaults (liq_danger 0.06, lev 50, cooldown 3600)
    orch2 = Orchestrator(orch.config, g2, sim, market)
    assert g2.liq_danger_pct == pytest.approx(0.05)
    assert g2.max_leverage == pytest.approx(25.0)
    assert g2.cooldown_seconds == pytest.approx(1800.0)


def _open_danger_long(orch):
    """Open a LONG that lands squarely in the danger zone (~5% from liq)."""
    opened = run(orch.queue_open("BTCUSDT", "LONG", 600.0, 20.0))
    orch.decide(opened.proposals[0]["id"], True)
    pos = next(p for p in orch.account().futures_positions if p.symbol == "BTCUSDT")
    assert pos.risk.value == "DANGER"
    return pos


def test_auto_protect_scan_does_nothing_in_manual_mode(stack):
    orch, market, sim = stack
    _open_danger_long(orch)
    assert orch.agent_mode == "manual"
    responses = run(orch.auto_protect_scan())
    assert responses == []
    # position untouched, nothing queued
    assert not [p for p in orch.proposals.values() if p.status.value == "PENDING"]


def test_auto_protect_scan_derisks_danger_without_prompt(stack):
    """The core proactive behavior: automatic mode de-risks a position that
    slips into the danger zone with nobody asking it to."""
    orch, market, sim = stack
    pos = _open_danger_long(orch)
    quantity_before = pos.quantity
    orch.set_agent_mode("auto", consent=True)

    responses = run(orch.auto_protect_scan())
    assert responses, "guardian must act on a DANGER position by itself"
    assert any(a.event == "auto_protect_scan" for a in orch.audit)
    assert any(a.event == "auto_de_risk" for a in orch.audit)
    assert not [p for p in orch.proposals.values() if p.status.value == "PENDING"]
    healed = next(p for p in orch.account().futures_positions if p.symbol == "BTCUSDT")
    assert healed.quantity < quantity_before          # reduced, not margin-added
    assert healed.risk.value != "DANGER"              # back to safe headroom


def test_auto_protect_throttles_blocked_attempts(stack):
    """A de-risk blocked by guardrails (e.g. symbol off the allowlist) must not
    retry every sweep — the guardian never skips guardrails, and never spams."""
    orch, market, sim = stack
    _open_danger_long(orch)
    orch.update_guardrails({"symbol_allowlist": ["ETHUSDT"]})  # BTC de-risk now blocked
    orch.set_agent_mode("auto", consent=True)
    run(orch.monitor_positions())
    blocked_first = [a for a in orch.audit if a.event == "de_risk_blocked"]
    assert blocked_first, "guardrails must still block an off-list de-risk"
    pos = next(p for p in orch.account().futures_positions if p.symbol == "BTCUSDT")
    assert pos.quantity == pytest.approx(12000.0 / 1000.0)     # untouched
    run(orch.monitor_positions())                              # immediately again -> throttled
    blocked_second = [a for a in orch.audit if a.event == "de_risk_blocked"]
    assert len(blocked_second) == len(blocked_first), \
        "second attempt must be throttled, not re-logged"


def test_monitor_positions_queues_de_risk_plan_in_manual_mode(stack):
    """Manual mode now escalates on its own: a DANGER position gets a de-risk
    plan queued (approval still required) without pressing Protect."""
    orch, market, sim = stack
    pos = _open_danger_long(orch)
    qty_before = pos.quantity
    responses = run(orch.monitor_positions())
    assert not responses, "manual mode must not execute on its own"
    pending = [p for p in orch.proposals.values() if p.status.value == "PENDING"]
    assert pending, "DANGER must auto-queue a de-risk plan in manual mode"
    healed = next(p for p in orch.account().futures_positions if p.symbol == "BTCUSDT")
    assert healed.quantity == qty_before, "nothing should execute without approval"
    assert any(a.event == "de_risk_alert" for a in orch.audit)
    assert any(a.event == "monitor_zone" for a in orch.audit)


def test_monitor_positions_executes_in_auto_mode(stack):
    """Auto mode keeps executing DANGER de-risks, now via the unified monitor."""
    orch, market, sim = stack
    orch.set_agent_mode("auto", consent=True)
    opened = run(orch.queue_open("BTCUSDT", "LONG", 600.0, 20.0))
    assert not opened.proposals, "auto mode opens immediately"
    pos = next(p for p in orch.account().futures_positions if p.symbol == "BTCUSDT")
    assert pos.risk.value == "DANGER"
    qty_before = pos.quantity
    responses = run(orch.monitor_positions())
    assert responses, "auto mode must de-risk on its own"
    healed = next(p for p in orch.account().futures_positions if p.symbol == "BTCUSDT")
    assert healed.quantity < qty_before
    assert any(a.event == "auto_de_risk" for a in orch.audit)


def test_monitor_positions_does_not_requeue_while_pending(stack):
    """A queued-but-undecided plan must not be re-queued every sweep."""
    orch, market, sim = stack
    _open_danger_long(orch)
    run(orch.monitor_positions())
    n1 = len([p for p in orch.proposals.values() if p.status.value == "PENDING"])
    alerts1 = len([a for a in orch.audit if a.event == "de_risk_alert"])
    assert n1 > 0
    run(orch.monitor_positions())  # immediately again
    n2 = len([p for p in orch.proposals.values() if p.status.value == "PENDING"])
    alerts2 = len([a for a in orch.audit if a.event == "de_risk_alert"])
    assert (n2, alerts2) == (n1, alerts1), "must be throttled while pending"


def test_monitor_visibility_watch_logged_ok_silent(stack):
    """The monitor narrates positions it is watching (WATCH+) and stays quiet
    on calm OK positions, and never queues anything below DANGER."""
    orch, market, sim = stack
    # 10x BTC -> WATCH (~9.5% from liq); 2x ETH -> OK (~49% from liq)
    b = run(orch.queue_open("BTCUSDT", "LONG", 500.0, 10.0))
    orch.decide(b.proposals[0]["id"], True)
    e = run(orch.queue_open("ETHUSDT", "LONG", 500.0, 2.0))
    orch.decide(e.proposals[0]["id"], True)
    run(orch.monitor_positions())
    zone_events = [a.detail for a in orch.audit if a.event == "monitor_zone"]
    assert any("ETHUSDT" not in d and "BTCUSDT" in d and "WATCH" in d for d in zone_events)
    assert not any("ETHUSDT" in d for d in zone_events), "calm OK positions stay quiet"
    assert not [p for p in orch.proposals.values() if p.status.value == "PENDING"]
