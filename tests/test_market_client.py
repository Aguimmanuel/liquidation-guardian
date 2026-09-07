"""Pure-math regressions for the market-analysis helpers (no network)."""

from pytest import approx

from app.market.client import _change_pct_over, _wilder_rsi


def _reference_wilder(closes, period=14):
    """Reference Wilder RSI implemented straight from the definition, used to
    cross-check the shared helper."""
    if len(closes) <= period:
        return 50.0
    gains = losses = 0.0
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        gains += max(d, 0.0)
        losses += max(-d, 0.0)
    avg_g, avg_l = gains / period, losses / period
    for i in range(period + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        avg_g = (avg_g * (period - 1) + max(d, 0.0)) / period
        avg_l = (avg_l * (period - 1) + max(-d, 0.0)) / period
    if avg_l == 0:
        return 100.0
    if avg_g == 0:
        return 0.0
    return 100.0 - 100.0 / (1.0 + avg_g / avg_l)


def test_wilder_rsi_matches_reference_on_noisy_series():
    closes = [100.0]
    rng = 101.0
    for i in range(1, 120):
        rng = (rng * 1103 + 51749) % 100  # tiny deterministic PRNG
        closes.append(closes[-1] + ((rng / 100.0) - 0.5) * 2.0)
    assert _wilder_rsi(closes) == _reference_wilder(closes)


def test_wilder_rsi_never_divides_by_zero_on_straight_bars():
    """B7: 40 straight up-bars (zero losses) or down-bars must not crash."""
    up = [100 + i for i in range(40)]
    down = [400 - i for i in range(40)]
    assert _wilder_rsi(up) == 100.0
    assert _wilder_rsi(down) == 0.0
    assert _wilder_rsi([100.0] * 40) == 50.0  # flat series -> neutral


def test_wilder_rsi_is_truly_period_14_not_whole_window():
    """B8: RSI must react to the most recent volatility, not average the whole
    series. A distant rise followed by a fresh crash reads deep-oversold under
    true RSI(14); the old whole-window average diluted that crash away."""
    closes = [100.0]
    for _ in range(40):
        closes.append(closes[-1] + 0.5)      # a distant +20 rise
    for _ in range(100):
        closes.append(closes[-1] + 0.01)     # long calm
    for _ in range(14):
        closes.append(closes[-1] - 3.0)      # fresh crash
    for _ in range(3):
        closes.append(closes[-1] + 0.5)      # small bounce

    rsi = _wilder_rsi(closes)
    assert rsi < 15.0, f"expected deep oversold after a fresh crash, got {rsi}"

    def whole_window_bug(c):
        g = l = 0.0
        for i in range(1, len(c)):
            d = c[i] - c[i - 1]
            g += max(d, 0.0)
            l += max(-d, 0.0)
        n = len(c) - 1
        return 50.0 if (g + l) == 0 else 100.0 - 100.0 / (1.0 + (g / n) / (l / n))

    diluted = whole_window_bug(closes)
    assert diluted > 25.0, "old whole-window calc should read milder (diluted)"
    assert rsi < diluted - 10.0


def test_change_pct_over_uses_hourly_window_consistently():
    """B9: 24h change references the close 24 full hours back (index -25 on 1h
    bars) and both analysis paths share the same helper."""
    closes = [100.0] * 25 + [110.0]  # 25 closes: -25th == 100, last == 110
    assert _change_pct_over(closes, 24) == approx(10.0)
    assert _change_pct_over([100.0] * 24, 24) == 0.0  # not enough data
    assert _change_pct_over([100.0] * 25 + [99.0], 24) == approx(-1.0)
