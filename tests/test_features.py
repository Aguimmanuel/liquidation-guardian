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


def _fund(orch, amount):
    """Deposit spot cash into the free futures wallet (Binance model: money
    must live futures-side before it can be used as margin)."""
    r = run(orch.transfer_balance("to_futures", amount))
    assert r.proposals, r.message
    orch.decide(r.proposals[0]["id"], True)


def _open_long(orch, symbol="BTCUSDT", margin=500.0, leverage=10.0, extra_fund=0.0):
    """Open a futures position the Binance way: fund the futures wallet first
    (margin + optional extra so later margin adds have free balance), then
    open. Margin is drawn from the futures wallet, never from spot."""
    _fund(orch, margin + extra_fund)
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
    # 500 deposited to futures first: spot falls by 500, the open spends the
    # free futures balance (wallet back to 0), NOT spot cash
    assert acct.cash_usdt == pytest.approx(9500.0)
    assert acct.futures_wallet_usdt == pytest.approx(0.0)


def test_open_dynamic_leverage_matters(stack):
    orch, market, sim = stack
    _open_long(orch, margin=500.0, leverage=25.0)
    pos = next(p for p in orch.account().futures_positions if p.symbol == "BTCUSDT")
    assert pos.notional_usdt == pytest.approx(500 * 25.0)
    assert pos.leverage == pytest.approx(25.0)


def test_open_rejects_invalid_margin_and_insufficient_cash(stack):
    """A degenerate open is refused (no dust floor / leverage cap any more, but
    a zero margin or unpayable margin can never go through)."""
    orch, market, sim = stack
    bad = run(orch.queue_open("BTCUSDT", "LONG", 0.0, 10.0))
    assert "blocked" in bad.message.lower()
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
    _fund(orch, 400.0)
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
    # close pays the taker fee on the closed notional ($5,000 x 0.04% = $2),
    # so the wallet gets margin 500 - fee 2 = 498
    assert acct1.futures_wallet_usdt == pytest.approx(498.0, abs=0.05)
    ret = run(orch.return_futures_wallet_to_spot())
    assert ret.proposals and ret.proposals[0]["transfer_kind"] == "RETURN_WALLET"
    orch.decide(ret.proposals[0]["id"], True)
    acct2 = orch.account()
    assert acct2.futures_wallet_usdt == pytest.approx(0.0)
    assert acct2.cash_usdt == pytest.approx(9998.0, abs=0.05)  # 9500 spot + 498 returned
    assert any(a.event == "return_proposed" for a in orch.audit)


def test_release_excess_margin_lands_in_futures_wallet(stack):
    """Margin added by a de-risk can be pulled back off an open position once
    it is no longer needed, bounded so the position keeps its de-risk
    headroom and stays in the OK zone. Under the Binance model the released
    margin returns to the free futures wallet (not spot)."""
    from app.models import TransferKind

    orch, market, sim = stack
    # fund 500 margin + 8000 spare futures balance so the danger top-up has
    # money to draw from and there is free balance to release into later
    _open_long(orch, margin=500.0, leverage=10.0, extra_fund=8000.0)
    # position opens WATCH (~9.5% from liq)
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

    wallet_before = snap["account"]["futures_wallet_usdt"]
    cash_before = snap["account"]["cash_usdt"]
    ret = run(orch.release_margin_to_wallet("BTCUSDT"))
    assert ret.proposals and ret.proposals[0]["transfer_kind"] == TransferKind.RELEASE_MARGIN.value
    released = ret.proposals[0]["est_value_usdt"]
    orch.decide(ret.proposals[0]["id"], True)
    acct2 = orch.account()
    pos2 = next(p for p in acct2.futures_positions if p.symbol == "BTCUSDT")
    assert acct2.cash_usdt == pytest.approx(cash_before, abs=0.05)   # spot untouched
    assert acct2.futures_wallet_usdt == pytest.approx(wallet_before + released, abs=0.05)
    assert pos2.risk.value == "OK"                          # still protected
    assert pos2.margin_usdt < margin_after_topup            # some margin came home
    assert any(a.event == "release_proposed" for a in orch.audit)


def test_release_refused_when_nothing_excess_and_target_misconfigured(stack):
    """Release is bounded: nothing to release when the position still needs
    margin, and the action is refused entirely if the de-risk target is not
    above the watch zone (it could otherwise push a position into warning)."""
    orch, market, sim = stack
    _open_long(orch, margin=500.0, leverage=10.0)           # WATCH: needs MORE margin
    ret = run(orch.release_margin_to_wallet("BTCUSDT"))
    assert not ret.proposals
    assert "no excess margin" in ret.message.lower()
    # a nonsense config (target <= watch zone) must suspend releases outright.
    # B11 now refuses such an edit through update_guardrails, so we simulate
    # the only remaining source: a legacy/corrupt profile loaded straight into
    # the config object (the orchestrator still defends itself).
    _open_long(orch, symbol="ETHUSDT", margin=500.0, leverage=10.0)
    orch.guardrails.liq_target_dist_pct = 0.08              # below watch 0.10
    ret2 = run(orch.release_margin_to_wallet("ETHUSDT"))
    assert not ret2.proposals
    assert "de-risk target" in ret2.message


def test_guardrail_edits_surface_in_snapshot(stack):
    orch, market, sim = stack
    orch.update_guardrails({"liq_danger_pct": 0.05, "liq_target_dist_pct": 0.20})
    gs = orch.guardrail_status()
    assert gs["values"]["liq_danger_pct"] == pytest.approx(0.05)
    assert gs["values"]["liq_target_dist_pct"] == pytest.approx(0.20)
    snap = orch.snapshot()
    assert snap["guardrails"]["liq_target_dist_pct"] == pytest.approx(0.20)


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
    _fund(orch, 500.0)                                        # fund futures first
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
    _fund(orch, 500.0)                                        # fund futures first
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
    pos = _open_danger_long(orch)                             # 20x, funded first
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
    bad = orch.update_guardrails({"liq_target_dist_pct": 5.0})  # > 1.0 = 100% bound
    assert "must stay within" in bad.message
    assert orch.guardrails.liq_target_dist_pct == pytest.approx(0.15)
    nan = orch.update_guardrails({"liq_danger_pct": float("nan")})
    assert "not a finite number" in nan.message


def test_guardrail_edit_refuses_inverted_rail_order(stack):
    """B11: no edit may break 0 < danger < warn < target < 1 — an inverted
    profile would silently flip every zone test."""
    orch, market, sim = stack
    assert orch.guardrails.liq_danger_pct < orch.guardrails.liq_warn_pct < orch.guardrails.liq_target_dist_pct
    bad = orch.update_guardrails({"liq_target_dist_pct": 0.08})  # below warn 0.10
    assert "danger < warn < target" in bad.message
    assert orch.guardrails.liq_target_dist_pct == pytest.approx(0.15), "edit refused"
    bad2 = orch.update_guardrails({"liq_danger_pct": 0.12})      # above warn 0.10
    assert "danger < warn < target" in bad2.message
    assert orch.guardrails.liq_danger_pct == pytest.approx(0.06)
    # combined valid edit across all three rails is fine
    ok = orch.update_guardrails({
        "liq_danger_pct": 0.04, "liq_warn_pct": 0.08, "liq_target_dist_pct": 0.20,
    })
    assert "Refused" not in ok.message
    assert orch.guardrails.liq_danger_pct == pytest.approx(0.04)
    assert orch.guardrails.liq_warn_pct == pytest.approx(0.08)
    assert orch.guardrails.liq_target_dist_pct == pytest.approx(0.20)


def test_inverted_profile_clamped_on_load(stack, tmp_path):
    """B11: a persisted profile that breaks the rail ordering is clamped back
    to the safe defaults on startup — never booted into inverted zone logic."""
    orch, market, sim = stack
    # write an inverted profile straight to the guardrail state file (as an
    # older/corrupt file could), then boot a fresh orchestrator on it
    path = orch.config.guardrail_state_file
    import json as _json
    from pathlib import Path as _Path
    _Path(path).write_text(_json.dumps({
        "values": {
            "liq_danger_pct": 0.18,        # > warn 0.10 and > target 0.08
            "liq_warn_pct": 0.10,
            "liq_target_dist_pct": 0.08,
        },
        "symbol_allowlist": ["BTCUSDT", "ETHUSDT"],
    }))
    from app.agent.orchestrator import Orchestrator
    from app.config import GuardrailConfig
    g2 = GuardrailConfig()
    orch2 = Orchestrator(orch.config, g2, sim, market)
    assert g2.liq_danger_pct < g2.liq_warn_pct < g2.liq_target_dist_pct
    assert g2.liq_danger_pct == pytest.approx(0.06), "clamped to the safe default"
    assert any(a.event == "guardrails_clamped" for a in orch2.audit)


def test_profile_values_persist_across_restart(stack, tmp_path):
    """Edits survive a restart (env defaults no longer win)."""
    orch, market, sim = stack
    orch.update_guardrails({"liq_danger_pct": 0.05, "liq_target_dist_pct": 0.22})

    from app.agent.orchestrator import Orchestrator
    from app.config import GuardrailConfig

    g2 = GuardrailConfig()  # fresh env defaults (danger 0.06, target 0.15)
    orch2 = Orchestrator(orch.config, g2, sim, market)
    assert g2.liq_danger_pct == pytest.approx(0.05)
    assert g2.liq_target_dist_pct == pytest.approx(0.22)


def _open_danger_long(orch):
    """Open a LONG that lands squarely in the danger zone (~5% from liq)."""
    _fund(orch, 600.0)                                        # fund futures first
    opened = run(orch.queue_open("BTCUSDT", "LONG", 600.0, 20.0))
    orch.decide(opened.proposals[0]["id"], True)
    pos = next(p for p in orch.account().futures_positions if p.symbol == "BTCUSDT")
    assert pos.risk.value == "DANGER"
    return pos


def test_auto_mode_message_explains_reduce_over_add_margin(stack):
    """#3: in automatic mode the deterministic rule must prefer the reduce
    (no extra capital) whenever one is viable, and the response has to explain
    the trade-off in plain English — never a silent pick, never both options."""
    orch, market, sim = stack
    # fund margin + spare balance so BOTH de-risk options are on the table
    _fund(orch, 1200.0)
    opened = run(orch.queue_open("BTCUSDT", "LONG", 500.0, 10.0))
    orch.decide(opened.proposals[0]["id"], True)
    market.prices["BTCUSDT"] = 920.0                   # push LONG into DANGER
    orch.set_agent_mode("auto", consent=True)

    responses = run(orch.monitor_positions())
    assert responses, "guardian must act on a DANGER position by itself"
    healed = next(p for p in orch.account().futures_positions if p.symbol == "BTCUSDT")
    assert healed.quantity < 5000.0 / 920.0, "reduce-first: quantity must drop"
    text = responses[0].message
    assert "Why:" in text and "REDUCE over add-margin" in text
    assert "no extra capital" in text
    # the unchosen add-margin option was rejected, not left pending
    assert not [p for p in orch.proposals.values() if p.status.value == "PENDING"]
    assert any(a.event == "auto_de_risk" for a in orch.audit)


def test_auto_mode_message_adds_margin_when_reduce_not_viable(stack):
    """#3: when no viable reduce exists for a symbol (the engine would omit
    Option A), the deterministic rule chooses add-margin and says why."""
    orch, market, sim = stack
    _open_long(orch, margin=500.0, leverage=10.0, extra_fund=700.0)
    market.prices["BTCUSDT"] = 920.0                    # LONG slips toward liq
    run(orch.monitor_positions())                       # manual: queue both options
    pend = {p.kind.value: p for p in orch.proposals.values()
            if p.status.value == "PENDING"}
    assert "TRADE" in pend and "TRANSFER" in pend
    # hand the auto-chooser ONLY the margin option (reduce omitted = not viable)
    resp = orch._auto_exec_de_risk([pend["TRANSFER"]])
    assert "ADD-MARGIN over reduce" in resp.message
    assert "partial close can't restore" in resp.message
    pos = next(p for p in orch.account().futures_positions if p.symbol == "BTCUSDT")
    assert pos.margin_usdt == pytest.approx(500.0 + pend["TRANSFER"].est_value_usdt)
    assert pos.quantity == pytest.approx(5000.0 / 1000.0)  # opened @1000, full position kept


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
    _fund(orch, 600.0)                                    # fund futures first
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
    _open_long(orch, symbol="BTCUSDT", margin=500.0, leverage=10.0)
    _open_long(orch, symbol="ETHUSDT", margin=500.0, leverage=2.0)
    run(orch.monitor_positions())
    zone_events = [a.detail for a in orch.audit if a.event == "monitor_zone"]
    assert any("ETHUSDT" not in d and "BTCUSDT" in d and "WATCH" in d for d in zone_events)
    assert not any("ETHUSDT" in d for d in zone_events), "calm OK positions stay quiet"
    assert not [p for p in orch.proposals.values() if p.status.value == "PENDING"]


def test_b10_close_fills_at_current_mark_not_stale_estimate(stack):
    """B10: a proposal approved after the price has drifted must fill at the
    CURRENT mark, not the estimate captured at propose time. est_price stays
    on the proposal as the record of what was offered."""
    orch, market, sim = stack
    _open_long(orch, margin=500.0, leverage=10.0)        # 5 BTC @ 1000
    market.prices["BTCUSDT"] = 1100.0                    # move before proposing
    run(orch.refresh())                                  # quote the proposal at 1100
    run(orch.queue_close("BTCUSDT", close_all=True))
    prop = next(p for p in orch.proposals.values() if p.status.value == "PENDING")
    assert prop.est_price == pytest.approx(1100.0), "proposal priced at propose time"
    market.prices["BTCUSDT"] = 1200.0                    # drifts before approval
    run(orch.refresh())                                  # mark the current quote
    out = orch.decide(prop.id, True)
    assert "Closed BTCUSDT" in out.message and "@ 1,200" in out.message
    assert prop.executed_price == pytest.approx(1200.0)
    acct = orch.account()
    # fill @1200 on 5 BTC: pnl = +1000, fee = 5*1200*0.0004 = 2.40,
    # wallet = margin 500 + pnl 1000 - fee 2.40
    assert acct.futures_wallet_usdt == pytest.approx(1497.60, abs=0.01)
    assert acct.futures_positions == []


# ---------------------------------------------------------------------------
# Free-balance spot <-> futures transfers + console command flows
# ---------------------------------------------------------------------------


def test_funds_transfer_both_directions_by_amount(stack):
    orch, market, sim = stack
    r = run(orch.transfer_balance("to_futures", 2000.0))
    assert r.proposals and r.proposals[0]["transfer_kind"] == "DEPOSIT_FUTURES"
    orch.decide(r.proposals[0]["id"], True)
    acct = orch.account()
    assert acct.cash_usdt == pytest.approx(8000.0)
    assert acct.futures_wallet_usdt == pytest.approx(2000.0)

    r2 = run(orch.transfer_balance("to_spot", 500.0))
    assert r2.proposals and r2.proposals[0]["transfer_kind"] == "RETURN_WALLET"
    orch.decide(r2.proposals[0]["id"], True)
    acct = orch.account()
    assert acct.cash_usdt == pytest.approx(8500.0)
    assert acct.futures_wallet_usdt == pytest.approx(1500.0)
    assert any(a.event == "fund_move_proposed" for a in orch.audit)
    # the free balance is not margin — total value is unchanged by the moves
    assert acct.total_value_usdt == pytest.approx(10000.0)


def test_funds_transfer_refuses_more_than_available(stack):
    orch, market, sim = stack
    r = run(orch.transfer_balance("to_spot", 99999.0))
    assert not r.proposals
    assert "free futures balance" in r.message.lower()
    r2 = run(orch.transfer_balance("to_futures", 99999.0))
    assert not r2.proposals
    assert "spot cash" in r2.message.lower()


def test_funds_transfer_all_when_no_amount(stack):
    orch, market, sim = stack
    run(orch.transfer_balance("to_futures", 1200.0))
    prop = [p for p in orch.proposals.values() if p.status.value == "PENDING"][0]
    orch.decide(prop.id, True)
    # blank amount -> moves the whole source balance
    r = run(orch.transfer_balance("to_spot", None))
    orch.decide(r.proposals[0]["id"], True)
    acct = orch.account()
    assert acct.futures_wallet_usdt == pytest.approx(0.0)
    assert acct.cash_usdt == pytest.approx(10000.0)


def test_condition_buy_adds_margin_not_sells(stack):
    """Regression: a BUY ("add margin") condition on an open position must add
    margin — never sell quantity of the position, and never be blocked by a
    cooldown right after a trade."""
    orch, market, sim = stack
    # fund 500 margin + 600 spare futures balance (the condition top-up spends
    # free futures, not spot)
    _open_long(orch, margin=500.0, leverage=10.0, extra_fund=600.0)
    pos = next(p for p in orch.account().futures_positions if p.symbol == "BTCUSDT")
    assert pos.quantity == pytest.approx(5.0)
    # arm an add-margin condition below the current price and drop to fire it
    run(orch.add_condition({
        "symbol": "BTCUSDT", "op": "BELOW", "price": 800.0,
        "side": "BUY", "amount_usdt": 600.0, "note": "add margin",
    }))
    market.prices["BTCUSDT"] = 700.0
    run(orch.check_conditions())
    pend = [p for p in orch.proposals.values() if p.status.value == "PENDING"
            and p.kind.value == "TRANSFER"]
    assert pend, "condition should queue a TRANSFER add-margin"
    assert pend[0].transfer_kind.value == "ADD_MARGIN"
    orch.decide(pend[0].id, True)
    acct = orch.account()
    pos = next(p for p in acct.futures_positions if p.symbol == "BTCUSDT")
    assert pos.margin_usdt == pytest.approx(1100.0)        # +600 added
    assert pos.quantity == pytest.approx(5.0)              # quantity untouched
    assert acct.cash_usdt == pytest.approx(8900.0)         # 1100 went to futures
    assert acct.futures_wallet_usdt == pytest.approx(0.0)  # spare 600 was spent


def test_condition_buy_without_position_is_blocked(stack):
    orch, market, sim = stack
    run(orch.add_condition({
        "symbol": "BTCUSDT", "op": "BELOW", "price": 800.0,
        "side": "BUY", "amount_usdt": 300.0, "note": "",
    }))
    market.prices["BTCUSDT"] = 700.0
    run(orch.check_conditions())
    assert not [p for p in orch.proposals.values() if p.status.value == "PENDING"]
    assert any(a.event == "condition_blocked" for a in orch.audit)
    assert any(a.event == "condition_fired" for a in orch.audit)
    # one-shot + auto-close: even a blocked fire consumes the condition — it
    # is closed and removed so it can never retry or linger on the page
    assert not orch.conditions


def _arm_below(stack, price=800.0, amount=400.0):
    orch, market, sim = stack
    run(orch.add_condition({
        "symbol": "BTCUSDT", "op": "BELOW", "price": price,
        "side": "BUY", "amount_usdt": amount, "note": "",
    }))
    return orch, market, sim


def test_condition_fires_once_then_is_closed_and_removed(stack):
    """One-shot + auto-close: a condition fires once when the price crosses
    the trigger, queues its single proposal, and is then closed and removed —
    it never lingers on the page, never re-fires while price stays past the
    trigger, and never re-arms after price returns and crosses again."""
    orch, market, sim = stack
    _open_long(orch, margin=500.0, leverage=10.0, extra_fund=500.0)  # spare wallet
    orch, market, sim = _arm_below(stack)                  # BELOW 800, price 1000
    market.prices["BTCUSDT"] = 700.0                       # first crossing below
    run(orch.check_conditions())
    assert sum(1 for p in orch.proposals.values()
               if p.status.value == "PENDING" and p.kind.value == "TRANSFER") == 1
    assert not orch.conditions, "condition is closed and removed after its single fire"
    assert any(a.event == "condition_closed" for a in orch.audit)
    assert any(a.event == "condition_fired" for a in orch.audit)
    # market stays below: five more sweeps, zero new proposals
    for _ in range(5):
        run(orch.check_conditions())
    assert sum(1 for p in orch.proposals.values()
               if p.status.value == "PENDING" and p.kind.value == "TRANSFER") == 1
    # even after the price returns above and crosses below again, nothing fires
    market.prices["BTCUSDT"] = 2000.0
    run(orch.check_conditions())
    market.prices["BTCUSDT"] = 700.0
    run(orch.check_conditions())
    assert sum(1 for p in orch.proposals.values()
               if p.status.value == "PENDING" and p.kind.value == "TRANSFER") == 1
    assert not orch.conditions


def test_fired_condition_gone_from_snapshot(stack):
    """The page renders S.conditions straight from the snapshot — once a
    condition fires it must be absent there too, so a page refresh can never
    resurrect a spent condition."""
    orch, market, sim = stack
    _open_long(orch, margin=500.0, leverage=10.0, extra_fund=500.0)
    orch, market, sim = _arm_below(stack, price=800.0, amount=400.0)
    assert len(orch.snapshot()["conditions"]) == 1
    market.prices["BTCUSDT"] = 700.0
    run(orch.check_conditions())
    snap = orch.snapshot()
    assert snap["conditions"] == [], "spent condition must be absent from the snapshot"
    assert not orch.conditions


def test_condition_armed_while_price_already_past_waits_for_fresh_crossing(stack):
    """B12: arming a condition whose trigger the market has already crossed
    must NOT fire instantly — the latch is seeded at arm time, so the
    condition waits until the price leaves and re-enters the trigger."""
    orch, market, sim = stack
    _open_long(orch, margin=500.0, leverage=10.0, extra_fund=500.0)  # spare wallet
    # current BTC price is 1000; arm BELOW 1500 = already satisfied at arm time
    run(orch.add_condition({
        "symbol": "BTCUSDT", "op": "BELOW", "price": 1500.0,
        "side": "BUY", "amount_usdt": 300.0, "note": "",
    }))
    run(orch.check_conditions())                           # still below 1500: no fire
    cond = next(iter(orch.conditions.values()))
    assert cond.fires == 0 and cond.active is True
    market.prices["BTCUSDT"] = 2000.0                      # leaves the trigger
    run(orch.check_conditions())
    market.prices["BTCUSDT"] = 1400.0                      # fresh crossing below
    run(orch.check_conditions())
    assert sum(1 for p in orch.proposals.values()
               if p.status.value == "PENDING" and p.kind.value == "TRANSFER") == 1
    assert not orch.conditions, "fired condition is removed after its one shot"

def test_condition_defers_when_plan_pending_same_symbol(stack):
    """#1: a fired condition must not stack a second action on a symbol that
    already has a pending de-risk plan — the shot is consumed once (one-shot)
    but nothing is queued on top of the existing plan."""
    orch, market, sim = stack
    _open_long(orch, margin=500.0, leverage=10.0, extra_fund=700.0)
    market.prices["BTCUSDT"] = 920.0                     # into DANGER
    run(orch.monitor_positions())                        # manual: plan queued
    pend_before = [p for p in orch.proposals.values() if p.status.value == "PENDING"]
    assert pend_before, "de-risk plan must be pending first"
    orch, market, sim = _arm_below(stack, price=900.0, amount=300.0)
    market.prices["BTCUSDT"] = 850.0                     # fresh crossing below
    run(orch.check_conditions())
    assert not orch.conditions, "the fired condition is closed and removed even when it defers"
    pend_after = [p for p in orch.proposals.values() if p.status.value == "PENDING"]
    assert len(pend_after) == len(pend_before), "must not stack a second plan"
    assert any(a.event == "condition_fired" for a in orch.audit)
    assert any(a.event == "condition_deferred" for a in orch.audit)


def test_condition_rejects_invalid_price_or_amount(stack):
    """B12: conditions must carry a positive trigger price and a positive
    amount; nonsense payloads are refused with a readable message."""
    orch, market, sim = stack
    _open_long(orch, margin=500.0, leverage=10.0, extra_fund=500.0)
    r = run(orch.add_condition({
        "symbol": "BTCUSDT", "op": "BELOW", "price": 0.0,
        "side": "BUY", "amount_usdt": 300.0,
    }))
    assert r.intent == "add_condition" and "positive trigger price" in r.message
    assert not orch.conditions
    r2 = run(orch.add_condition({
        "symbol": "BTCUSDT", "op": "BELOW", "price": 800.0,
        "side": "BUY", "amount_usdt": -50.0,
    }))
    assert "positive USDT" in r2.message
    assert not orch.conditions


def test_condition_auto_mode_does_not_drain_funds_on_stale_trigger(stack):
    """In auto mode the worst failure is spending the futures balance each
    poll; the one-shot rule means a stuck-below price executes exactly once."""
    orch, market, sim = stack
    # fund 500 margin + 900 spare futures balance (500 cash stays in spot)
    _open_long(orch, margin=500.0, leverage=10.0, extra_fund=900.0)
    orch, market, sim = _arm_below(stack, price=800.0, amount=400.0)
    run(orch.handle_message("I understand - enable auto"))
    market.prices["BTCUSDT"] = 700.0
    run(orch.check_conditions())                           # auto executes one margin add
    acct = orch.account()
    pos = next(p for p in acct.futures_positions if p.symbol == "BTCUSDT")
    assert pos.margin_usdt == pytest.approx(900.0)         # exactly one +400
    for _ in range(3):
        run(orch.check_conditions())                       # stays below — nothing more
    acct = orch.account()
    pos = next(p for p in acct.futures_positions if p.symbol == "BTCUSDT")
    assert pos.margin_usdt == pytest.approx(900.0)
    assert not orch.conditions, "auto-fired condition is closed and removed after executing once"
    # the second +400 was never spent: 900 spare was deposited, 400 used
    assert acct.futures_wallet_usdt == pytest.approx(500.0)
    assert acct.cash_usdt == pytest.approx(8600.0)         # 1400 moved to futures


def test_console_open_margin_transfer_close_end_to_end(stack):
    """Plain-language console commands drive the whole lifecycle, Binance-style:
    funds are moved into the futures wallet first, then used as margin."""
    orch, market, sim = stack
    # 0) deposit into the futures wallet via chat (the required first step)
    r0 = run(orch.handle_message("move 2000 to futures"))
    assert r0.intent == "funds" and r0.proposals
    orch.decide(r0.proposals[0]["id"], True)
    assert orch.account().futures_wallet_usdt == pytest.approx(2000.0)

    # 1) open via chat (margin drawn from the futures wallet)
    r = run(orch.handle_message("open long BTC 500 at 10x"))
    assert r.intent == "open" and r.proposals, r.message
    assert r.proposals[0]["kind"] == "OPEN" and r.proposals[0]["symbol"] == "BTCUSDT"
    orch.decide(r.proposals[0]["id"], True)
    pos = next(p for p in orch.account().futures_positions if p.symbol == "BTCUSDT")
    assert pos.quantity == pytest.approx(5.0)
    assert orch.account().futures_wallet_usdt == pytest.approx(1500.0)

    # 2) add margin via chat (also from the futures wallet)
    r2 = run(orch.handle_message("add 200 margin to BTC"))
    assert r2.intent == "add_margin" and r2.proposals
    assert r2.proposals[0]["transfer_kind"] == "ADD_MARGIN"
    orch.decide(r2.proposals[0]["id"], True)
    pos = next(p for p in orch.account().futures_positions if p.symbol == "BTCUSDT")
    assert pos.margin_usdt == pytest.approx(700.0)
    assert orch.account().futures_wallet_usdt == pytest.approx(1300.0)

    # 3) deposit more free cash into the futures wallet via chat
    r3 = run(orch.handle_message("move 1000 to futures"))
    assert r3.intent == "funds" and r3.proposals
    orch.decide(r3.proposals[0]["id"], True)
    assert orch.account().futures_wallet_usdt == pytest.approx(2300.0)

    # 4) close via chat -> margin + PnL park back in the futures wallet
    r4 = run(orch.handle_message("close BTC"))
    assert r4.intent == "close" and r4.proposals
    orch.decide(r4.proposals[0]["id"], True)
    assert not [p for p in orch.account().futures_positions if p.symbol == "BTCUSDT"]
    # flat close at entry: margin 700 returns, minus $2 taker fee on $5,000 notional
    assert orch.account().futures_wallet_usdt == pytest.approx(2998.0, abs=0.5)


def test_console_rail_edits_and_agent_mode(stack):
    orch, market, sim = stack
    r = run(orch.handle_message("set danger zone to 5%"))
    assert r.intent in ("chat", "rail_edit", "guardrails")
    assert orch.guardrails.liq_danger_pct == pytest.approx(0.05)
    r2 = run(orch.handle_message("set de-risk target to 12%"))
    assert orch.guardrails.liq_target_dist_pct == pytest.approx(0.12)
    # auto needs explicit consent phrase
    r3 = run(orch.handle_message("auto mode"))
    assert orch.agent_mode == "manual"           # not enabled without consent
    assert "consent" in r3.message.lower() or "understand" in r3.message.lower()
    r4 = run(orch.handle_message("I understand - enable auto"))
    assert orch.agent_mode == "auto"
    run(orch.handle_message("manual mode"))
    assert orch.agent_mode == "manual"


def test_rails_snapshot_exact_shape(stack):
    orch, market, sim = stack
    gs = orch.guardrail_status()
    assert set(gs["values"].keys()) == {
        "liq_warn_pct", "liq_danger_pct", "liq_target_dist_pct",
    }
    for gone in ("max_trade_pct", "cooldown_seconds", "budget",
                 "min_trade_value_usdt", "max_leverage"):
        assert gone not in gs["values"], gone
    for gone in ("cooldown_seconds", "max_trade_pct",
                 "min_trade_value_usdt", "max_leverage"):
        assert not hasattr(orch.guardrails, gone), gone


def test_small_typed_funds_transfer_is_allowed(stack):
    """A typed, non-max amount works and no dust floor blocks it — regression
    guard for the retired min-action-value knob and the wallet transfer UI."""
    orch, market, sim = stack
    r = run(orch.transfer_balance("to_futures", 2.5))
    assert r.proposals and r.proposals[0]["transfer_kind"] == "DEPOSIT_FUTURES"
    assert r.proposals[0]["est_value_usdt"] == pytest.approx(2.5)
    orch.decide(r.proposals[0]["id"], True)
    acct = orch.account()
    assert acct.futures_wallet_usdt == pytest.approx(2.5)
    assert acct.cash_usdt == pytest.approx(9997.5)


# ---------------------------------------------------------------------------
# Binance funding model regressions: futures wallet is the margin source
# ---------------------------------------------------------------------------


def test_open_requires_deposit_to_futures_wallet_first(stack):
    """A fresh account starts with all $10k in spot and $0 in the futures
    wallet. Under the Binance model, opening before depositing is blocked with
    a clear message — it must never silently spot-fund the margin."""
    orch, market, sim = stack
    assert orch.account().cash_usdt == pytest.approx(10000.0)
    assert orch.account().futures_wallet_usdt == pytest.approx(0.0)
    resp = run(orch.queue_open("BTCUSDT", "LONG", 500.0, 10.0))
    assert not resp.proposals, "fresh account must not be able to open"
    assert "deposit from spot first" in resp.message, resp.message
    assert "futures" in resp.message.lower()
    # nothing moved anywhere
    acct = orch.account()
    assert acct.cash_usdt == pytest.approx(10000.0)
    assert acct.futures_wallet_usdt == pytest.approx(0.0)
    assert not acct.futures_positions


def test_add_margin_without_open_position_fails_cleanly(stack):
    """A stale add-margin approval (position closed before it executed) must
    fail loudly rather than silently 'succeeding' by moving nothing."""
    orch, market, sim = stack
    _open_long(orch, margin=500.0, leverage=10.0, extra_fund=200.0)
    # queue an add-margin while the position is open…
    r = run(orch.handle_message("add 200 margin to BTC"))
    assert r.intent == "add_margin" and r.proposals
    pending = r.proposals[0]["id"]
    # …then close the position before approving it
    c = run(orch.queue_close("BTCUSDT"))
    orch.decide(c.proposals[0]["id"], True)
    assert not [p for p in orch.account().futures_positions if p.symbol == "BTCUSDT"]
    acct = orch.account()
    wallet_before = acct.futures_wallet_usdt
    # approving the stale add-margin must fail without moving money
    out = orch.decide(pending, True)
    assert "no open BTCUSDT position" in out.message, out.message
    assert orch.account().futures_wallet_usdt == pytest.approx(wallet_before)


# ---------------------------------------------------------------------------
# Phase-1 review regressions (B1/B5/B6/B13/B14/B21)
# ---------------------------------------------------------------------------


def test_snapshot_risk_summary_has_distinct_open_count_and_ranked_list(stack):
    """B1: the snapshot must expose a numeric open_count AND the ranked risk
    list under separate keys — the old duplicate "positions" key let the list
    overwrite the count (which broke the UI's watch badge)."""
    orch, market, sim = stack
    snap = orch.snapshot()
    rs = snap["risk_summary"]
    assert rs["open_count"] == 0
    assert isinstance(rs["positions"], list) and rs["positions"] == []
    assert rs["danger"] == 0 and rs["watch_count"] == 0
    _open_danger_long(orch)
    rs = orch.snapshot()["risk_summary"]
    assert rs["open_count"] == 1
    assert len(rs["positions"]) == 1          # ranked list stays intact
    assert rs["positions"][0]["symbol"] == "BTCUSDT"
    assert rs["danger"] == 1


def test_open_rejects_leverage_outside_exchange_limits(stack):
    """B14: 1x..125x is a hard-coded exchange limit, not an editable knob —
    values outside it are refused with a clear message."""
    orch, market, sim = stack
    _fund(orch, 6000.0)
    for bad in (0.5, 126.0, 500.0):
        resp = run(orch.queue_open("BTCUSDT", "LONG", 500.0, bad))
        assert not resp.proposals, f"leverage {bad}x must be blocked"
        assert "125" in resp.message and "leverage" in resp.message.lower(), resp.message
    good = run(orch.queue_open("BTCUSDT", "LONG", 500.0, 125.0))
    assert good.proposals, "125x is at the exchange limit and must be accepted"


def test_pending_de_risk_plan_does_not_suppress_stop_loss(stack):
    """B5: a crossed stop-loss must execute even when a non-CLOSE proposal
    (e.g. the guardian's own de-risk add-margin plan) is still pending on the
    same symbol — only an in-flight CLOSE should block it."""
    orch, market, sim = stack
    _open_long(orch, margin=500.0, leverage=10.0, extra_fund=400.0)
    pos = next(p for p in orch.account().futures_positions if p.symbol == "BTCUSDT")
    sim.set_tp_sl("BTCUSDT", take_profit=0.0, stop_loss=950.0)
    # create a pending ADD_MARGIN (non-CLOSE) proposal on the same symbol
    r = run(orch.handle_message("add 200 margin to BTC"))
    assert r.intent == "add_margin" and r.proposals
    pending_id = r.proposals[0]["id"]
    assert any(
        p.id == pending_id and p.status.value == "PENDING" for p in orch.proposals.values()
    )
    # price crashes through the stop-loss
    market.prices["BTCUSDT"] = 940.0
    run(orch.scan_tp_sl())
    assert not [p for p in orch.account().futures_positions if p.symbol == "BTCUSDT"], (
        "stop-loss must fire despite the pending add-margin plan"
    )
    assert any(a.event == "tp_sl_fired" for a in orch.audit)


def test_close_fee_charged_on_notional_not_pnl(stack):
    """B6+B13: the reported fee on a futures close equals notional x rate (not
    PnL x rate) and the wallet is credited margin + pnl - that fee."""
    from app.models import ProposalKind

    orch, market, sim = stack
    _open_long(orch, margin=500.0, leverage=10.0)  # qty 5.0 @ price 1000
    pos = next(p for p in orch.account().futures_positions if p.symbol == "BTCUSDT")
    assert pos.quantity == pytest.approx(5.0)
    from app.models import TradeProposal as TP
    from app.models import Side

    close = TP(
        id="feee000001", symbol="BTCUSDT", side=Side.SELL,
        est_value_usdt=5500.0, est_quantity=5.0, est_price=1100.0,
        reason="fee regression", kind=ProposalKind.CLOSE, close_all=True,
        keep_margin=False,
    )
    result = sim.execute(close)
    assert result.ok, result.message
    # notional 5 * 1100 = 5500; taker fee 5500 * 0.0004 = 2.2 (NOT pnl 500 * 0.0004 = 0.2)
    assert result.fee_usdt == pytest.approx(2.2, abs=0.01)
    acct = orch.account()
    assert not acct.futures_positions
    # wallet = margin 500 + pnl (1100-1000)*5 = 500 - fee 2.2 = 997.8
    assert acct.futures_wallet_usdt == pytest.approx(997.8, abs=0.05)


def test_adding_to_position_blends_leverage_not_max(stack):
    """B21: adding margin/quantity at a lower leverage blends the position's
    effective leverage by notional instead of keeping max(old, new)."""
    orch, market, sim = stack
    _fund(orch, 1000.0)
    r1 = run(orch.queue_open("BTCUSDT", "LONG", 500.0, 20.0))   # notional 10k
    orch.decide(r1.proposals[0]["id"], True)
    _fund(orch, 1000.0)
    r2 = run(orch.queue_open("BTCUSDT", "LONG", 500.0, 10.0))   # notional 5k, same price
    orch.decide(r2.proposals[0]["id"], True)
    pos = next(p for p in orch.account().futures_positions if p.symbol == "BTCUSDT")
    assert pos.quantity == pytest.approx(15.0)          # 10 + 5 @ price 1000
    assert pos.margin_usdt == pytest.approx(1000.0)
    # blended: total notional 15,000 / margin 1,000 = 15x  (not max(20,10)=20)
    assert pos.leverage == pytest.approx(15.0, abs=0.01)
