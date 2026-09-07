# Architecture

## Design goals

1. **The agent is a risk manager, not a gambler.** It can propose anything its
   deterministic engine computes; a separate guardrail layer decides what may
   actually be queued for execution.
2. **Mirror Binance Agent OS's security model.** Dedicated Agentic sub-account,
   no withdrawal scope, confirm-before-execute. The demo enforces the same
   "human approves every write" flow that Binance enforces on the live path.
3. **Demoable anywhere.** SIM mode + keyless public market data means the full
   agent loop runs with zero funds and zero accounts.
4. **One loop, two execution backends.** The orchestrator depends on an
   `ExecutionAdapter` protocol; swapping the adapter swaps the execution
   destination, nothing else changes.

## Component map

```
app/
├── main.py                # FastAPI app factory, lifespan, wiring
├── config.py              # AppConfig + GuardrailConfig (env-driven)
├── models.py              # Ticker, FuturesPosition, AccountState, TradeProposal,
│                          #   Condition, AuditEntry, …
├── agent/
│   ├── risk.py            # liquidation math: liq price, distance, zones,
│   │                      #   required_cut (deleverage), required_margin
│   ├── guardrails.py      # deterministic risk checks + aggregate gate
│   ├── orchestrator.py    # agent loop: intents, de-risk, approvals, audit
│   ├── narration.py       # template + optional LLM prose
│   └── prompts.py         # LLM system prompts
├── market/
│   └── client.py          # keyless Binance market-data client (TTL cache)
├── adapters/
│   ├── base.py            # ExecutionAdapter protocol + OrderResult
│   ├── sim.py             # paper account (spot + futures), real prices;
│   │                      #   open/close/TP-SL execution, 24h trade window
│   └── mcp_live.py        # Binance MCP server (live Agentic sub-account)
├── api/routes.py          # REST surface for the UI
└── static/index.html      # single-file web UI (no CDN): trade panel, TP/SL,
│                          #   guardrail editor, agent-mode toggle, live ticker,
│                          #   per-coin analysis, onboarding tour
```

## The agent loop

```
observe ──► analyze ──► decide ──► narrate ──► approve? ──► execute ──► audit
   │            │           │            │           │
 live prices  liq distance  guardrail    plain-       human      SimAdapter or
 + positions  + risk zones  engine       language    yes/no     MCPLiveAdapter
```

- **observe** — `MarketClient` fetches tickers from the keyless public feed
  (`data-api.binance.vision`); positions are marked to market and their risk
  fields recomputed.
- **analyze** — `app/agent/risk.py` computes the isolated-margin liquidation
  price for long/short, the distance to it in percent, and assigns a risk
  zone. The 4 watch coins stream at a fast TTL for the live ticker strip, and
  armed conditions / TP-SL levels are swept against live prices on a timer.
- **decide** — for every DANGER position the orchestrator computes both
  de-risk options and queues them:
  - **Reduce** (`required_cut`): base qty to close so the remaining position
    sits at the target headroom — closing *with the released margin kept on
    the remainder* (deleveraging), because a naive proportional close leaves
    the liq price unchanged.
  - **Add margin** (`required_margin`): USDT to transfer so the position sits
    at the target headroom, keeping the full position.
  - Every proposal runs through the guardrail gate; blocked ones are logged
    with the reason.
  - **Open / close** go through the same gate: dynamic leverage and margin are
    validated (cap, cash sufficiency, symbol, opposite-side conflicts) and the
    resulting OPEN / CLOSE proposals still wait for explicit approval (or
    auto-execute in automatic mode).
  - **TP/SL exits** are different by design: arming a take-profit or stop-loss
    is pre-authorization to exit, so the simulator closes the position the
    moment the level is crossed (no extra approval round-trip, like an
    exchange-side stop order).
- **narrate** — template prose by default; optionally rephrased by an LLM
  (OpenAI-compatible) when `RS_NARRATION_LLM=true`. The LLM only explains; it
  never decides.
- **approve (manual mode, default)** — proposals queue with status `PENDING`;
  only an explicit approve executes them. This mirrors Binance's
  confirm-before-execute.
- **act (automatic mode)** — a header toggle switches the orchestrator to
  automatic mode, but only after the user accepts an on-screen notice that
  *agents can make mistakes*. Once consented, queued opens/closes and fired
  conditions execute immediately, and a background *auto-protect scan*
  (`orchestrator.auto_protect_scan`, run by the server autopilot) watches
  every open position: any that slips into the danger zone is de-risked on its
  own, with no prompt required (guardrails still gate everything; the de-risk
  prefers the *reduce* — no extra capital — and rejects the unchosen option,
  and retries are throttled per symbol so a blocked attempt can't spam the
  audit trail). Switching back to Manual revokes consent.
- **execute** — `SimAdapter` (paper fills at mark price, simulated fees) or
  `MCPLiveAdapter` (Binance MCP server, real sub-account).
- **audit** — every event appended to an in-memory audit trail, exposed via
  `/api/state` and shown in the UI.

## Liquidation math (`agent/risk.py`)

Isolated margin, no fees: liquidation occurs when
`margin + unrealized PnL = maintenance margin`, which solves to

    long:   P_liq = (entry·qty − margin) / (qty·(1 − MMR))
    short:  P_liq = (entry·qty + margin) / (qty·(1 + MMR))

At `margin = entry·qty/leverage` this reduces to the familiar
`P_liq ≈ entry·(1 ∓ 1/leverage)`.

**Key insight:** closing part of a position releases margin proportionally, so
a naive partial close does not move the liq price. The guardian's *reduce*
instead keeps the released margin on the remainder (deleveraging), and its
*add margin* injects capital — the two levers that actually change the liq
price. `required_cut` has a closed-form solution under the "keep margin"
assumption:

    Q' = margin / (entry − target_liq·(1 − MMR))     (long)
    cut = Q − Q'

## Guardrail engine (`agent/guardrails.py`)

Every proposal is scored by deterministic checks: symbol allowlist, min trade
value, max order size (for buys and margin adds — a *reduce* is exempt because
closing exposure is the safe direction, like sells), daily trade cap,
cooldown (autonomous actions only), leverage cap on opens, and
cash/transfer sufficiency. The thresholds are user-editable at runtime
(`app/agent/orchestrator.py` `update_guardrails`) and persisted to
`data/guardrail_state.json`, so a profile survives restarts. The whole profile
can be **locked for the current 24-hour window** as a self-discipline
tripwire: while locked, loosening *any* limit is refused (a one-way ratchet —
each limit has a known "loosening direction", see `LOOSEN_DIRECTION` in
`config.py`), tightening is always allowed, and there is no early unlock; the
lock auto-releases when the window completes.

## Intent handling

User messages are parsed to an intent (`protect | risk | status | market |
add_condition | chat`) by a deterministic keyword parser, with an optional LLM
parser when a key is configured. Decisions never depend on the LLM. The UI
drives opens, closes, TP/SL arming and guardrail edits through dedicated JSON
endpoints (`/api/trade/*`, `/api/tpsl`, `/api/guardrails`).

## Security model

- The demo writes to `data/sim_state.json` only.
- The live adapter uses the official Binance MCP server; Binance itself
  enforces: dedicated Agentic sub-account, no withdrawal scope, and
  confirm-before-execute on every non-read action.
- No API keys are stored by the app; the live path uses OAuth/Bearer provided
  at runtime via env (`RS_MCP_ACCESS_TOKEN`).

## Testing

`tests/` covers the risk engine and feature surface (49 tests, no network —
everything runs against a fake market feed): liq price for long and short,
distance and risk zones, `required_cut`/`required_margin`, guardrail semantics,
text parsers, plus feature tests for: dynamic-leverage opens (cap, cash,
opposite-side blocks, same-side averaging), TP/SL arming and automatic exits on
both sides, the 24-hour trade window rollover, the profile commitment lock
(loosening refused / tightening allowed, early unlock refused, expiry,
persistence across restart), and the proactive auto-protect scan (manual-mode
no-op, autonomous de-risk without a prompt, retry throttling, guardrails
respected).
