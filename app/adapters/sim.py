"""Paper-trading adapter.

Runs the full agent loop against a simulated account (spot + USDⓈ-M futures)
priced with *real* Binance market data (keyless public feed) and simulated
taker fees. This is the default mode — the whole flow works with zero funds,
and the same code switches to the live MCP adapter by changing one env var.

State persists to a JSON file so the portfolio survives restarts.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.adapters.base import OrderResult
from app.config import AppConfig, GuardrailConfig
from app.market.client import MarketClient, MarketDataError
from app.models import (
    AccountState,
    FuturesPosition,
    Position,
    PositionSide,
    ProposalKind,
    Side,
    TradeProposal,
    TransferKind,
    utcnow,
)


class SimAdapter:
    mode = "sim"

    def __init__(
        self,
        config: AppConfig,
        guardrails: GuardrailConfig,
        market: MarketClient,
        state_file: Optional[str] = None,
    ):
        self.config = config
        self.guardrails = guardrails
        self.market = market
        self.state_file = Path(state_file or config.state_file)
        self._state: Optional[AccountState] = None
        self._loaded = False

    # ------------------------------------------------------------------ state
    def _load(self) -> None:
        if self._loaded:
            return
        if self.state_file.exists():
            try:
                raw = json.loads(self.state_file.read_text())
                self._state = self._from_dict(raw)
            except Exception:
                self._state = None
        if self._state is None:
            self._state = self._fresh()
        self._loaded = True

    def _fresh(self) -> AccountState:
        now = utcnow()
        return AccountState(
            total_value_usdt=self.config.default_initial_cash,
            cash_usdt=self.config.default_initial_cash,
            positions=[],
            spot_value_usdt=self.config.default_initial_cash,
            futures_wallet_usdt=0.0,
            futures_positions=[],
            peak_value_usdt=self.config.default_initial_cash,
            realized_pnl_usdt=0.0,
            last_trade_at=None,
            updated_at=now,
        )

    def _save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(self._to_dict(self._state), indent=2))

    def _to_dict(self, state: AccountState) -> dict:
        return {
            "cash_usdt": state.cash_usdt,
            "futures_wallet_usdt": state.futures_wallet_usdt,
            "peak_value_usdt": state.peak_value_usdt,
            "realized_pnl_usdt": state.realized_pnl_usdt,
            "last_trade_at": state.last_trade_at.isoformat() if state.last_trade_at else None,
            "positions": [
                {"symbol": p.symbol, "quantity": p.quantity, "price": p.price}
                for p in state.positions
            ],
            "futures_positions": [
                {
                    "symbol": p.symbol,
                    "side": p.side.value,
                    "entry_price": p.entry_price,
                    "quantity": p.quantity,
                    "leverage": p.leverage,
                    "margin_usdt": p.margin_usdt,
                    "mmr": p.mmr,
                    "mark_price": p.mark_price,
                    "take_profit_price": p.take_profit_price,
                    "stop_loss_price": p.stop_loss_price,
                }
                for p in state.futures_positions
            ],
        }

    def _from_dict(self, raw: dict) -> AccountState:
        last = datetime.fromisoformat(raw["last_trade_at"]) if raw.get("last_trade_at") else None
        futures = []
        for fp in raw.get("futures_positions", []):
            futures.append(
                FuturesPosition(
                    symbol=fp["symbol"],
                    side=PositionSide(fp["side"]),
                    entry_price=float(fp["entry_price"]),
                    quantity=float(fp["quantity"]),
                    leverage=float(fp["leverage"]),
                    margin_usdt=float(fp["margin_usdt"]),
                    mmr=float(fp.get("mmr", 0.005)),
                    mark_price=float(fp.get("mark_price", 0.0)),
                    take_profit_price=float(fp.get("take_profit_price", 0.0) or 0.0),
                    stop_loss_price=float(fp.get("stop_loss_price", 0.0) or 0.0),
                )
            )
        return AccountState(
            total_value_usdt=0.0,  # recomputed on next get_account
            cash_usdt=float(raw.get("cash_usdt", 0.0)),
            positions=[
                Position(
                    symbol=p["symbol"],
                    base=p["symbol"][: -len("USDT")] if p["symbol"].endswith("USDT") else p["symbol"],
                    quote="USDT",
                    quantity=float(p["quantity"]),
                    price=float(p["price"]),
                    value_usdt=0.0,
                    weight=0.0,
                )
                for p in raw.get("positions", [])
            ],
            futures_wallet_usdt=float(raw.get("futures_wallet_usdt", 0.0)),
            futures_positions=futures,
            peak_value_usdt=float(raw.get("peak_value_usdt", 0.0)),
            realized_pnl_usdt=float(raw.get("realized_pnl_usdt", 0.0)),
            last_trade_at=last,
            updated_at=utcnow(),
        )

    # ------------------------------------------------------------- account
    async def refresh_prices(self) -> None:
        """Mark spot and futures positions to market using the live feed."""
        self._load()
        state = self._state
        assert state is not None
        symbols = {p.symbol for p in state.positions} | {p.symbol for p in state.futures_positions}
        if not symbols:
            return
        try:
            # watch cache refreshes on the short ticker TTL so mark prices move
            # in near-real-time, not on the heavier market-data TTL
            tickers = await self.market.get_watch_tickers(list(symbols))
        except MarketDataError:
            return  # keep last known prices; a later refresh will catch up
        for pos in state.positions:
            t = tickers.get(pos.symbol)
            if t and t.price > 0:
                pos.price = t.price
                pos.value_usdt = pos.quantity * t.price
        for fp in state.futures_positions:
            t = tickers.get(fp.symbol)
            if t and t.price > 0:
                from app.agent.risk import evaluate_position

                evaluate_position(
                    fp,
                    t.price,
                    self.guardrails.liq_warn_pct,
                    self.guardrails.liq_danger_pct,
                )

    def get_account(self) -> AccountState:
        self._load()
        state = self._state
        assert state is not None
        spot = state.cash_usdt + sum(p.value_usdt for p in state.positions)
        state.spot_value_usdt = spot
        total = spot + state.futures_equity_usdt
        state.total_value_usdt = total
        if total > state.peak_value_usdt:
            state.peak_value_usdt = total
        for p in state.positions:
            p.weight = (p.value_usdt / total) if total > 0 else 0.0
        # keep derived risk fields fresh after any execution
        from app.agent.risk import evaluate_position

        for fp in state.futures_positions:
            if fp.mark_price > 0:
                evaluate_position(
                    fp, fp.mark_price,
                    self.guardrails.liq_warn_pct,
                    self.guardrails.liq_danger_pct,
                )
        state.updated_at = utcnow()
        return state

    # ------------------------------------------------------------- execute
    def execute(self, proposal: TradeProposal, *, bootstrap: bool = False) -> OrderResult:
        self._load()
        state = self._state
        assert state is not None

        if proposal.kind == ProposalKind.TRANSFER:
            return self._execute_transfer(proposal, bootstrap)
        if proposal.kind == ProposalKind.OPEN:
            return self._execute_open(proposal, bootstrap)

        price = proposal.est_price if proposal.est_price > 0 else proposal.executed_price or 0.0
        is_futures = self._has_futures_position(proposal.symbol)

        if is_futures or proposal.kind == ProposalKind.CLOSE:
            # futures path: CLOSE (reduce / exit) or TRADE reduce
            wallet_add = self._close_futures(
                proposal.symbol,
                proposal.est_quantity,
                price,
                self.guardrails.futures_fee_rate,
                keep_margin=proposal.keep_margin,
                close_all=proposal.close_all,
            )
            if wallet_add is None:
                return OrderResult(
                    ok=False, proposal_id=proposal.id, symbol=proposal.symbol,
                    side=proposal.side, executed_price=price, executed_value_usdt=0.0,
                    message="nothing to close at execution time",
                )
            state.futures_wallet_usdt += wallet_add
            executed = proposal.est_value_usdt
        elif proposal.side == Side.BUY:
            fee_rate = self.guardrails.fee_rate
            gross = proposal.est_value_usdt
            fee = gross * fee_rate
            if state.cash_usdt < gross + fee:
                return OrderResult(
                    ok=False, proposal_id=proposal.id, symbol=proposal.symbol,
                    side=Side.BUY, executed_price=price, executed_value_usdt=0.0,
                    message="insufficient spot balance at execution time",
                )
            state.cash_usdt -= gross + fee
            self._add_spot_position(proposal.symbol, gross / price, price)
            executed = gross
        else:  # SELL on spot
            fee_rate = self.guardrails.fee_rate
            qty = min(proposal.est_quantity, self._held_spot_qty(proposal.symbol))
            if qty <= 0:
                return OrderResult(
                    ok=False, proposal_id=proposal.id, symbol=proposal.symbol,
                    side=Side.SELL, executed_price=price, executed_value_usdt=0.0,
                    message="nothing to sell at execution time",
                )
            gross = qty * price
            fee = gross * fee_rate
            state.cash_usdt += gross - fee
            cost_basis = qty * self._spot_avg_cost(proposal.symbol)
            state.realized_pnl_usdt += gross - fee - cost_basis
            self._remove_spot_position(proposal.symbol, qty)
            executed = gross

        if not bootstrap:
            state.last_trade_at = utcnow()
        self._save()
        return OrderResult(
            ok=True,
            proposal_id=proposal.id,
            symbol=proposal.symbol,
            side=proposal.side,
            executed_price=price,
            executed_value_usdt=round(executed, 2),
            fee_usdt=round(executed * self.guardrails.futures_fee_rate, 2),
            message="simulated fill at mark price",
        )

    def _execute_transfer(self, proposal: TradeProposal, bootstrap: bool) -> OrderResult:
        """Move USDT between the spot wallet and the futures side.

        TransferKind decides the direction and source:
          ADD_MARGIN       free futures balance -> position margin (de-risk top-up),
          DEPOSIT_FUTURES  spot cash -> free futures-wallet balance (no position
                            is opened; funds just sit ready futures-side),
          RETURN_WALLET    futures wallet free balance -> spot,
          RELEASE_MARGIN   excess margin on an open position -> futures wallet
                            (bounded so the position keeps its de-risk headroom).
        """
        from app.agent.risk import evaluate_position, releasable_margin

        state = self._state
        assert state is not None
        kind = proposal.transfer_kind or TransferKind.ADD_MARGIN
        amount = proposal.est_value_usdt

        if kind in (TransferKind.RETURN_WALLET, TransferKind.DEPOSIT_FUTURES):
            if kind == TransferKind.RETURN_WALLET:
                take = min(amount, state.futures_wallet_usdt)
                if take < 1e-9:
                    return OrderResult(
                        ok=False, proposal_id=proposal.id, symbol=proposal.symbol,
                        side=proposal.side, executed_price=0.0, executed_value_usdt=0.0,
                        message="the futures wallet has no free balance to return",
                    )
                state.futures_wallet_usdt -= take
                state.cash_usdt += take
            else:  # DEPOSIT_FUTURES: spot cash -> free futures wallet
                if state.cash_usdt < amount - 1e-9:
                    return OrderResult(
                        ok=False, proposal_id=proposal.id, symbol=proposal.symbol,
                        side=proposal.side, executed_price=0.0, executed_value_usdt=0.0,
                        message=f"not enough spot cash: have ${state.cash_usdt:,.2f}",
                    )
                take = amount
                state.cash_usdt -= take
                state.futures_wallet_usdt += take
            if not bootstrap:
                state.last_trade_at = utcnow()
            self._save()
            word = "moved" if kind == TransferKind.RETURN_WALLET else "deposited"
            where = "the futures wallet back to spot" if kind == TransferKind.RETURN_WALLET else "into the free futures wallet"
            return OrderResult(
                ok=True,
                proposal_id=proposal.id,
                symbol=proposal.symbol,
                side=proposal.side,
                executed_price=0.0,
                executed_value_usdt=round(take, 2),
                fee_usdt=0.0,
                message=f"{word} ${take:,.2f} {where}",
            )

        if kind == TransferKind.RELEASE_MARGIN:
            fp = next(
                (p for p in state.futures_positions if p.symbol == proposal.symbol),
                None,
            )
            if fp is None:
                return OrderResult(
                    ok=False, proposal_id=proposal.id, symbol=proposal.symbol,
                    side=proposal.side, executed_price=0.0, executed_value_usdt=0.0,
                    message="no open position to release margin from",
                )
            target = self.guardrails.liq_target_dist_pct
            if target <= self.guardrails.liq_warn_pct:
                return OrderResult(
                    ok=False, proposal_id=proposal.id, symbol=proposal.symbol,
                    side=proposal.side, executed_price=0.0, executed_value_usdt=0.0,
                    message=(
                        "release suspended: the de-risk target must stay above the "
                        "watch zone so releasing margin can't push a position into risk"
                    ),
                )
            price = proposal.est_price if proposal.est_price > 0 else fp.mark_price
            if price <= 0:
                return OrderResult(
                    ok=False, proposal_id=proposal.id, symbol=proposal.symbol,
                    side=proposal.side, executed_price=0.0, executed_value_usdt=0.0,
                    message="no mark price available to size the margin release",
                )
            take = min(amount, releasable_margin(
                fp, price, target
            ))
            if take < 1e-9:
                return OrderResult(
                    ok=False, proposal_id=proposal.id, symbol=proposal.symbol,
                    side=proposal.side, executed_price=0.0, executed_value_usdt=0.0,
                    message="nothing to release — the position holds no excess margin right now",
                )
            fp.margin_usdt -= take
            state.futures_wallet_usdt += take
            evaluate_position(
                fp, fp.mark_price or price,
                self.guardrails.liq_warn_pct, self.guardrails.liq_danger_pct,
            )
            if not bootstrap:
                state.last_trade_at = utcnow()
            self._save()
            return OrderResult(
                ok=True,
                proposal_id=proposal.id,
                symbol=proposal.symbol,
                side=proposal.side,
                executed_price=0.0,
                executed_value_usdt=round(take, 2),
                fee_usdt=0.0,
                message=f"released ${take:,.2f} excess margin on {proposal.symbol} to the futures wallet",
            )

        # --- ADD_MARGIN (default / legacy): free futures balance -> position --
        fp = next((p for p in state.futures_positions if p.symbol == proposal.symbol), None)
        if fp is None:
            return OrderResult(
                ok=False, proposal_id=proposal.id, symbol=proposal.symbol,
                side=proposal.side, executed_price=0.0, executed_value_usdt=0.0,
                message=f"no open {proposal.symbol} position to add margin to — nothing was moved",
            )
        if state.futures_wallet_usdt < amount - 1e-9:
            return OrderResult(
                ok=False, proposal_id=proposal.id, symbol=proposal.symbol,
                side=proposal.side, executed_price=0.0, executed_value_usdt=0.0,
                message=f"not enough free futures balance: have ${state.futures_wallet_usdt:,.2f}",
            )
        state.futures_wallet_usdt -= amount
        fp.margin_usdt += amount
        if not bootstrap:
            state.last_trade_at = utcnow()
        self._save()
        return OrderResult(
            ok=True,
            proposal_id=proposal.id,
            symbol=proposal.symbol,
            side=proposal.side,
            executed_price=0.0,
            executed_value_usdt=round(amount, 2),
            fee_usdt=0.0,
            message=f"added ${amount:,.2f} margin to {proposal.symbol} from the futures balance",
        )

    def _execute_open(self, proposal: TradeProposal, bootstrap: bool) -> OrderResult:
        """Open a new leveraged position. Margin comes from the free balance of
        the USDⓈ-M futures wallet (Binance model: fund futures first, then
        trade)."""
        state = self._state
        assert state is not None
        margin = proposal.open_margin_usdt
        leverage = proposal.open_leverage or 10.0  # explicit or default 10x
        price = proposal.est_price if proposal.est_price > 0 else 0.0
        if price <= 0:
            return OrderResult(
                ok=False, proposal_id=proposal.id, symbol=proposal.symbol,
                side=proposal.side, executed_price=0.0, executed_value_usdt=0.0,
                message="no market price available to open",
            )
        if state.futures_wallet_usdt < margin - 1e-9:
            return OrderResult(
                ok=False, proposal_id=proposal.id, symbol=proposal.symbol,
                side=proposal.side, executed_price=0.0, executed_value_usdt=0.0,
                message=(
                    f"not enough free futures balance: need ${margin:,.2f}, "
                    f"have ${state.futures_wallet_usdt:,.2f} — deposit from spot first"
                ),
            )
        existing = next((p for p in state.futures_positions if p.symbol == proposal.symbol), None)
        want_side = PositionSide.LONG if proposal.side == Side.BUY else PositionSide.SHORT
        if existing is not None and existing.side != want_side:
            return OrderResult(
                ok=False, proposal_id=proposal.id, symbol=proposal.symbol,
                side=proposal.side, executed_price=0.0, executed_value_usdt=0.0,
                message=f"opposite {existing.side.value} position already open on {proposal.symbol} — close it first",
            )
        state.futures_wallet_usdt -= margin
        qty = (margin * leverage) / price
        if existing is not None:
            # add to existing same-side position (avg entry)
            total_margin = existing.margin_usdt + margin
            existing.entry_price = (existing.entry_price * existing.quantity + price * qty) / (existing.quantity + qty)
            existing.quantity += qty
            existing.margin_usdt = total_margin
            existing.leverage = max(existing.leverage, leverage)
            fp = existing
        else:
            fp = FuturesPosition(
                symbol=proposal.symbol,
                side=want_side,
                entry_price=price,
                quantity=qty,
                leverage=leverage,
                margin_usdt=margin,
                mmr=0.005,
                mark_price=price,
            )
            state.futures_positions.append(fp)
        from app.agent.risk import evaluate_position

        evaluate_position(fp, price, self.guardrails.liq_warn_pct, self.guardrails.liq_danger_pct)
        if not bootstrap:
            state.last_trade_at = utcnow()
        self._save()
        return OrderResult(
            ok=True,
            proposal_id=proposal.id,
            symbol=proposal.symbol,
            side=proposal.side,
            executed_price=price,
            executed_value_usdt=round(margin * leverage, 2),
            fee_usdt=0.0,
            message="simulated fill at mark price",
        )

    # ------------------------------------------------------------- helpers
    def _has_futures_position(self, symbol: str) -> bool:
        assert self._state is not None
        return any(p.symbol == symbol for p in self._state.futures_positions)

    def _held_spot_qty(self, symbol: str) -> float:
        assert self._state is not None
        for p in self._state.positions:
            if p.symbol == symbol:
                return p.quantity
        return 0.0

    def _spot_avg_cost(self, symbol: str) -> float:
        assert self._state is not None
        for p in self._state.positions:
            if p.symbol == symbol:
                return p.price
        return 0.0

    def _add_spot_position(self, symbol: str, qty: float, price: float) -> None:
        assert self._state is not None
        for p in self._state.positions:
            if p.symbol == symbol:
                new_qty = p.quantity + qty
                p.price = price
                p.quantity = new_qty
                p.value_usdt = new_qty * price
                return
        base = symbol[: -len("USDT")] if symbol.endswith("USDT") else symbol
        self._state.positions.append(
            Position(
                symbol=symbol, base=base, quote="USDT",
                quantity=qty, price=price, value_usdt=qty * price, weight=0.0,
            )
        )

    def _remove_spot_position(self, symbol: str, qty: float) -> None:
        assert self._state is not None
        for p in self._state.positions:
            if p.symbol == symbol:
                p.quantity -= qty
                p.value_usdt = max(0.0, p.value_usdt - qty * p.price)
                if p.quantity <= 1e-12:
                    self._state.positions.remove(p)
                return

    def _close_futures(
        self,
        symbol: str,
        qty: float,
        price: float,
        fee_rate: float,
        *,
        keep_margin: bool = True,
        close_all: bool = False,
    ):
        """Close qty (or all) of a futures position. Returns the amount of USDT
        that lands in the futures wallet, or None if there is nothing to close.

        Deleverage semantics (keep_margin=True): a partial close keeps the
        released margin on the remainder so the liquidation price actually
        moves — this is the guardian's "reduce". With keep_margin=False a
        partial close releases margin proportionally (classic take-profit).
        A full close always returns the whole margin plus realized PnL.
        """
        assert self._state is not None
        for fp in self._state.futures_positions:
            if fp.symbol == symbol:
                target = fp.quantity if close_all else min(qty or fp.quantity, fp.quantity)
                if target <= 0:
                    return None
                qty_before = fp.quantity
                closed = target
                if fp.side == PositionSide.LONG:
                    pnl = (price - fp.entry_price) * closed
                else:
                    pnl = (fp.entry_price - price) * closed
                pnl_net = pnl * (1.0 - fee_rate)
                self._state.realized_pnl_usdt += pnl_net
                if closed >= qty_before - 1e-12:
                    wallet_add = fp.margin_usdt + pnl_net
                    self._state.futures_positions.remove(fp)
                elif keep_margin:
                    fp.quantity = qty_before - closed
                    wallet_add = pnl_net  # margin stays on the remainder
                else:
                    # classic take-profit: release margin proportionally too
                    margin_before = fp.margin_usdt
                    fp.quantity = qty_before - closed
                    fp.margin_usdt = margin_before * (fp.quantity / qty_before)
                    wallet_add = pnl_net + (margin_before - fp.margin_usdt)
                return wallet_add
        return None

    def set_tp_sl(self, symbol: str, take_profit: float = 0.0, stop_loss: float = 0.0) -> bool:
        """Arm (or clear with 0) TP/SL prices on a position."""
        self._load()
        state = self._state
        assert state is not None
        for fp in state.futures_positions:
            if fp.symbol == symbol:
                fp.take_profit_price = take_profit or 0.0
                fp.stop_loss_price = stop_loss or 0.0
                fp.tp_hit = False
                fp.sl_hit = False
                self._save()
                return True
        return False

    # -------------------------------------------------------------- control
    def reset(self, cash: Optional[float] = None) -> AccountState:
        now = utcnow()
        self._state = AccountState(
            total_value_usdt=cash or self.config.default_initial_cash,
            cash_usdt=cash or self.config.default_initial_cash,
            positions=[],
            spot_value_usdt=cash or self.config.default_initial_cash,
            futures_wallet_usdt=0.0,
            futures_positions=[],
            peak_value_usdt=cash or self.config.default_initial_cash,
            realized_pnl_usdt=0.0,
            last_trade_at=None,
            updated_at=now,
        )
        self._loaded = True
        self._save()
        return self._state

    def close(self) -> None:
        pass
