"""API schema boundary tests (no server needed — pydantic validation).

B12: conditions require positive trigger price and positive amount.
B14: opens enforce the hard 1..125x exchange leverage limit.
"""

import pytest
from pydantic import ValidationError

from app.api.routes import ConditionIn, OpenIn


def _cond(**kw) -> ConditionIn:
    base = {"symbol": "BTCUSDT", "op": "BELOW", "price": 800.0,
            "side": "BUY", "amount_usdt": 300.0}
    base.update(kw)
    return ConditionIn(**base)


def test_condition_schema_rejects_nonpositive_price():
    for bad in (0, -5, -0.01):
        with pytest.raises(ValidationError):
            _cond(price=bad)


def test_condition_schema_rejects_nonpositive_amount():
    for bad in (0, -100):
        with pytest.raises(ValidationError):
            _cond(amount_usdt=bad)


def test_open_schema_enforces_exchange_leverage_limits():
    ok = OpenIn(symbol="BTCUSDT", side="LONG", margin_usdt=500.0, leverage=125.0)
    assert ok.leverage == 125.0
    ok2 = OpenIn(symbol="BTCUSDT", side="LONG", margin_usdt=500.0, leverage=1.0)
    assert ok2.leverage == 1.0
    for bad in (0, 0.5, 126.0, 500.0):
        with pytest.raises(ValidationError):
            OpenIn(symbol="BTCUSDT", side="LONG", margin_usdt=500.0, leverage=bad)
