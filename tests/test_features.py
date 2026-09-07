"""Feature tests for the item-7 build: dynamic leverage opens, TP/SL auto
execution, daily-window budget + lock, and guardrail editing.

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
    assert acct.trade_count_today == 1


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


def test_daily_budget_lock_is_irreversible_until_window(stack):
    orch, market, sim = stack
    orch.update_guardrails({"max_daily_trades": 2, "daily_trades_locked": True})
    gs = orch.guardrail_status()
    assert gs["locked"] is True
    assert gs["lock_remaining_seconds"] > 0

    # cannot raise the cap while locked
    blocked = orch.update_guardrails({"max_daily_trades": 99})
    assert "locked" in blocked.message.lower()
    # cannot unlock early
    blocked2 = orch.update_guardrails({"daily_trades_locked": False})
    assert "locked" in blocked2.message.lower()
    assert orch.guardrails.max_daily_trades == 2
    assert orch.guardrail_status()["locked"] is True

    # other guardrail keys remain editable while the budget is locked
    ok = orch.update_guardrails({"liq_warn_pct": 0.14})
    assert "updated" in ok.message.lower()
    assert orch.guardrails.liq_warn_pct == pytest.approx(0.14)

    # a 24h-old lock auto-expires on the next status read
    assert orch.guardrails.daily_lock_at is not None
    orch.guardrails.daily_lock_at = utcnow() - timedelta(hours=25)
    assert orch.guardrail_status()["locked"] is False
    changed = orch.update_guardrails({"max_daily_trades": 7})
    assert orch.guardrails.max_daily_trades == 7
    assert "updated" in changed.message.lower()

    # a fresh 24h window rolls the daily counter back to zero
    _open_long(orch)
    assert orch.account().trade_count_today == 1
    sim._load()
    assert sim._state is not None
    sim._state.daily_window_start = utcnow() - timedelta(hours=25)
    sim._save()
    assert orch.account().trade_count_today == 0


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


def test_guardrail_lock_persists_across_restart(stack, tmp_path):
    orch, market, sim = stack
    cfg = orch.config
    # lock it
    orch.update_guardrails({"max_daily_trades": 3, "daily_trades_locked": True})
    lock_at = orch.guardrails.daily_lock_at
    assert lock_at is not None
    # a "restart": fresh guardrails + fresh orchestrator over the same files
    from app.agent.orchestrator import Orchestrator
    from app.config import GuardrailConfig

    g2 = GuardrailConfig()
    orch2 = Orchestrator(cfg, g2, sim, market)
    assert g2.daily_trades_locked is True, "lock must survive a restart"
    assert g2.daily_lock_at == lock_at
    gs2 = orch2.guardrail_status()
    assert gs2["locked"] is True
    # cannot unlock even after restart
    msg = orch2.update_guardrails({"daily_trades_locked": False})
    assert "cannot be unlocked early" in msg.message
    # once expired it clears and editing returns
    g2.daily_lock_at = utcnow() - timedelta(hours=25)
    assert orch2.guardrail_status()["locked"] is False
    ok = orch2.update_guardrails({"max_daily_trades": 9})
    assert g2.max_daily_trades == 9
