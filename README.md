# Liquidation Guardian

A focused AI risk agent for [Binance Agent OS](https://www.binance.com/en/agent-os).
It watches open USDⓈ-M futures positions, computes their liquidation risk live,
and recommends de-risk actions before the exchange liquidates them. This is
intentionally a narrow agent: it helps protect capital, not a general-purpose
trading bot.

## What this agent does

This project is designed around one clear job: protect leveraged futures
positions from liquidation risk.

- **Monitors liquidation risk in real time.** BTC / ETH / BNB / SOL update
  continuously from the market feed, and each open position is scored as
  OK / WATCH / DANGER based on its distance to liquidation.
- **Explains the safest next step.** When a position gets dangerous, the agent
  proposes the two standard protection paths: reduce exposure or add margin.
- **Applies guardrails before action.** Every action is checked against size,
  wallet, leverage, cooldown, drawdown, and daily-budget rules.
- **Keeps the human in the loop.** Default mode waits for explicit approval.
  Automatic mode can act only after a clear consent notice and still obeys all
  guardrails.
- **Adds contextual analysis when useful.** A per-coin market read can help the
  operator understand trend, support/resistance, and whether risk should be
  reduced or preserved.

This scope is deliberate. The project is not trying to be a broad autonomous
trading system; it is an AI risk agent for futures protection and staged
execution planning.

## Why a "guardian" and not just an alert

A liquidation price is easy to compute; the hard part is what happens when you
get close to it. In a fast market, panic takes over — people average into a
losing position, add margin at the worst moment, or freeze. The guardian does
the math calmly and hands you a concrete, size-capped plan with the numbers.
The agent is a risk manager, not a gambler.

## Quickstart

```bash
pip install -r requirements.txt
python -m uvicorn app.main:app --port 8000
```

Open http://localhost:8000. The app starts with a $10,000 paper account (dark
theme is the default; the header toggle persists your choice).

**Demo mode:** This project is intentionally configured to run in `sim` mode.
It uses a paper account with live public market prices and does not require
Binance credentials. Keep `RS_MODE=sim` for demos, testing, and local use.

## Screenshots

![dashboard](docs/screenshots/dashboard.png)
![protect](docs/screenshots/protect.png)
![after reduce](docs/screenshots/after_reduce.png)

Suggested first run:

1. In the **Trade** panel pick BTC and open a Long — say 20x with $600 margin.
   Liquidation sits ~4.5% away, so the guardian flags it DANGER immediately.
   (Blank leverage defaults to 10x; your leverage-cap guardrail blocks more.)
2. In the positions table, arm a **take-profit** and a **stop-loss**.
3. Hit **📈 Market analysis** and pick a coin for the deep read.
4. Say `protect my positions` — the guardian proposes reduce and add-margin.
5. Approve the reduce: liquidation jumps from ~4.5% to ~15% headroom.
6. Open **Guardrails**, tighten the daily trade budget and **Lock for 24h**.
7. Try the **⚡ Auto** agent mode and read the consent notice before enabling.

## Live mode status

Live mode is not the supported project workflow yet. It requires a funded
Binance Agentic sub-account, MCP authentication, and verification against the
account response schema. Do not set `RS_MODE=live` unless you are developing
and testing the live adapter with a dedicated test account.

The live adapter (`app/adapters/mcp_live.py`) uses the official MCP Python SDK
against `https://agent.binance.com/mcp/agentic`. It discovers the account and
trade tools at runtime. Live execution is experimental and has not been
validated against every Binance MCP account schema. The simulator remains the
safe, supported path; it cannot move real funds or place real orders.

## Scripts and tests

```bash
python -m pytest tests/ -q            # 42 unit tests, no network needed
python scripts/backtest.py --days 90  # guardian vs no-guardian on real klines
python scripts/demo.py                # scripted walkthrough of the agent loop
```

The backtest pulls hourly klines from the public feed, picks the worst 5-day
crash in the period, opens a leveraged long at the top, and compares: hold
(no guardian), guardian that cuts, guardian that adds margin. It shows
liquidations avoided and capital preserved.

## Project layout

```
app/
├── agent/          # risk engine (liq math), guardrails, narration, orchestrator
├── adapters/       # sim (paper) and MCP live execution
├── market/         # keyless Binance market-data client with TTL cache
├── api/            # FastAPI routes
└── static/         # single-file web UI (inline CSS/JS, no external assets)
scripts/            # backtest + scripted demo
tests/              # unit tests for the risk engine, features, and parsers
docs/               # architecture notes, screenshots
```

## Configuration

Everything is environment-driven — see `.env.example` for the full list. The
main knobs:

| Var | Default | Meaning |
|---|---|---|
| `RS_MODE` | `sim` | `sim` or `live` |
| `RS_LIQ_WARN_PCT` | `0.10` | distance to liq below this = WATCH |
| `RS_LIQ_DANGER_PCT` | `0.06` | distance to liq below this = DANGER → de-risk proposed |
| `RS_LIQ_TARGET_DIST_PCT` | `0.15` | de-risk until 15% headroom |
| `RS_MAX_TRADE_PCT` | `0.10` | max order size as share of portfolio (buys/margin) |
| `RS_MIN_TRADE_VALUE_USDT` | `10.0` | ignore dust-size orders |
| `RS_MAX_DAILY_TRADES` | `10` | executions per 24-hour window (UI-editable + lockable) |
| `RS_MAX_LEVERAGE` | `50` | leverage cap when opening trades (UI-editable) |
| `RS_TICKER_TTL` | `1` | live-ticker refresh interval in seconds |
| `RS_SYMBOL_ALLOWLIST` | BTC,ETH,BNB,SOL | empty = allow any symbol |
| `OPENAI_API_KEY` | — | optional: LLM narration + intent parsing |

Guardrail values set from the UI apply for the running session (env vars are
the defaults); the 24-hour budget lock is persisted to `data/guardrail_state.json`
so it survives restarts and releases itself when its window completes.

## Notes

- The liquidation math models isolated margin without fees or funding (see
  `app/agent/risk.py` for the derivation). The live adapter reads real
  positions from the MCP account scope, so live numbers come from Binance
  itself.
- A naive "close part of the position" does *not* move the liquidation price —
  margin is released proportionally. That's why the guardian's reduce keeps the
  released margin on the remainder (deleveraging). The backtest demonstrates
  this.
- The automatic agent still respects every guardrail, but guardrails cannot
  guarantee profit or prevent losses — the consent notice says so, on purpose.
- The optional LLM layer only rewrites narration and parses free-text intents.
  All decisions come from the deterministic engine.
- Not financial advice. Leveraged trading can liquidate your entire margin.
  The backtest is illustrative, not a strategy claim.

## License

MIT — see [LICENSE](LICENSE).
