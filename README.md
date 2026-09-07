# Liquidation Guardian

A focused AI risk agent for [Binance Agent OS](https://www.binance.com/en/agent-os):
it watches open USDⓈ-M futures positions against live market data, computes the
real distance to each position's liquidation price, and proposes guardrailed
de-risk actions — reduce leverage or add margin — before the exchange
liquidates you. Intentionally narrow: a capital-protection agent for leveraged
futures, not a general-purpose trading bot.

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
  Automatic mode watches your open positions around the clock and, when one
  slips toward liquidation, de-risks it on its own — no prompt needed. It
  engages only after a clear consent notice and still obeys all guardrails.
- **Adds contextual analysis when useful.** A per-coin market read can help the
  operator understand trend, support/resistance, and whether risk should be
  reduced or preserved.
- **A risk profile you can lock for 24 hours.** Every limit — danger/warn
  zones, de-risk target, order size, leverage cap, cooldown, daily trade
  budget — is editable in the UI and persists across restarts. **Lock the
  profile** to commit: while locked you can still make any limit *stricter*,
  but loosening any limit is refused (a one-way ratchet) until the 24-hour
  window ends. There is no early unlock, by design.

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
./run.sh        # macOS / Linux
run.bat         # Windows
```

or by hand: `pip install -r requirements.txt`, then
`python -m uvicorn app.main:app --port 8000`.

Open http://localhost:8000. The app starts in **sim mode** with a $10,000 paper
account priced off live Binance market data — no Binance account or credentials
needed to explore every feature. The identical agent loop runs against the real
Binance Agent OS when you switch to live (see below).

## Screenshots

![dashboard](docs/screenshots/dashboard.png)
![protect](docs/screenshots/protect.png)
![after reduce](docs/screenshots/after_reduce.png)
![coin analysis](docs/screenshots/analysis.png)

Suggested first run:

1. In the **Trade** panel pick BTC and open a Long — say 20x with $600 margin.
   Liquidation sits ~4.5% away, so the guardian flags it DANGER immediately.
   (Blank leverage defaults to 10x; your leverage-cap guardrail blocks more.)
2. In the positions table, arm a **take-profit** and a **stop-loss**.
3. Hit **📈 Market analysis** and pick a coin for the deep read.
4. Say `protect my positions` — the guardian proposes reduce and add-margin.
5. Approve the reduce: liquidation jumps from ~4.5% to ~15% headroom.
6. Open **Guardrails**, tighten the limits and **Lock profile** — then try
   loosening one to watch the 24-hour one-way ratchet refuse it.
7. Try the **⚡ Auto** agent mode and read the consent notice before enabling.
   Let prices drift against your position: automatic mode de-risks it on its
   own, without you asking.

## Binance Agent OS / live mode

The same agent loop runs against the real Binance Agent OS MCP server —
`https://agent.binance.com/mcp/agentic` — through the official MCP Python SDK
(`app/adapters/mcp_live.py`). The adapter connects over streamable HTTP,
discovers the account and trade tools at runtime, and executes on a dedicated
**Agentic sub-account**, the isolation model Binance enforces: you fund the
sub-account manually, the agent has no withdrawal scope and can never pull
funds from your main account, and every non-read action goes through Binance's
confirm-before-execute flow.

Simulation is the default so the whole product runs with zero credentials and
zero real orders (paper fills at live prices). To trade live:

1. Create and fund an **Agentic sub-account** in Binance — the agent can only
   use funds already inside that sub-account.
2. `export RS_MODE=live` and start the app.
3. On first use the MCP client discovers Binance's OAuth authorization endpoint
   and opens the consent flow in your browser — or set `RS_MCP_ACCESS_TOKEN`
   to a pre-issued token to skip it. Approve the requested account/trade
   scopes, then confirm each order in Binance's confirm-before-execute screen.

The adapter logs the tools and scopes it discovers on startup, so you can
verify exactly what the live account exposes before placing any order. Start
with a small amount in the sub-account and work up.

## Scripts and tests

```bash
python -m pytest tests/ -q            # 49 unit tests, no network needed
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

Env vars are the boot defaults only. Edits made in the UI — plus any profile
lock — are persisted to `data/guardrail_state.json`, so your settings survive
restarts; a locked profile releases itself when its 24-hour window completes.

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
