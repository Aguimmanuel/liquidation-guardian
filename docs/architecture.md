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
  conditions execute immediately, and the always-on guardian sweep
  (`orchestrator.monitor_positions`, run by the server autopilot every 2 s)
  watches every open position in *both* modes: any that slips into the danger
  zone is de-risked on its own, with no prompt required (guardrails still gate
  everything). The deterministic auto-chooser picks per symbol — reduce
  (no extra capital) whenever one is viable, add-margin only when a reduce
  can't restore the headroom target, never both — and the response explains
  the trade-off in plain English. Retries are throttled per symbol so a
  blocked attempt can't spam the audit trail. The sweep order is protective
  exits → monitor → conditions, so a condition never stacks a second action on
  a symbol that already has a pending plan. Switching back to Manual revokes
  consent.
  The same `monitor_positions()` loop runs while positions are
  open: it tracks each symbol's risk zone and logs zone changes, and on DANGER
  it queues a de-risk plan in manual mode (approval still required) or
  executes it in automatic mode — so the guardian escalates without being
  asked in either mode.
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

The guardrail engine keeps only checks that protect without ever freezing a
de-risk: symbol allowlist, spot-cash/transfer sufficiency, and position
existence. Removed on purpose — after an audit showed each was decorative for
a protection agent — the per-day trade budget, an order-size cap, a cooldown,
a dust minimum (min action value), and a leverage cap. None of them could
block a protective de-risk, but all of them *could* have frozen one mid-crash,
so the guardian is capped by nothing except cash reality and the symbol list.
The only user-editable knobs left in the UI are the three liquidation-zone
thresholds (danger zone, watch zone, de-risk target) — sanity-bounded
(`RAIL_BOUNDS`) and persisted to `data/guardrail_state.json`.

Money movement is two-way. Transfers are a single proposal kind with a
`transfer_kind` discriminator: `ADD_MARGIN` (futures wallet free balance →
position margin, the de-risk top-up), `DEPOSIT_FUTURES` (spot cash → free
futures wallet), `RETURN_WALLET` (free futures wallet → spot), and
`RELEASE_MARGIN` (excess margin on an open position → futures wallet). Margin
for opening and add-margin always comes out of the futures wallet — never out
of spot — matching Binance's USDTⓈ-M futures model; a fresh account holds all
its cash in spot until the user deposits into the futures wallet. Free-balance
moves never touch an open
position's margin; the release size is bounded by `risk.releasable_margin`,
which keeps the position at least at its de-risk headroom
(`liq_target_dist_pct`) — so moving money back out can never recreate the
liquidation risk the guardian exists to prevent. Releasing is refused
outright while the de-risk target is set at or below the watch zone.

## Intent handling

The console is a real command surface, not a fixed menu. Every message is
parsed by a deterministic keyword parser (`protect | risk | status | market |
add_condition | open | close | add_margin | release | funds | agent_auto |
agent_manual | rail_edit | chat`) — always available, no key needed. When an
OpenAI-compatible key is configured (`OPENAI_API_KEY`), an LLM does the same
job far more flexibly: it receives a compact portfolio snapshot as context,
maps open-ended commands to the exact same JSON intent schema, and answers
genuine questions with grounded plain-language replies. Any model endpoint
works (OpenAI, Groq, OpenRouter… or an Agent OS-hosted model), and decisions
never depend on the LLM — it only classifies and chats; every action still
flows through the guardrailed propose/approve pipeline, exactly as its button
does (`/api/trade/*`, `/api/return`, `/api/funds/transfer`, `/api/tpsl`,
`/api/guardrails`).

Price conditions are one-shot and auto-close: a condition fires *once* when
price first crosses the trigger, queues/executes its single action, and is
then removed from state entirely — it disappears from the page and can never
re-propose, re-execute, or re-arm, so a coin that stays past the trigger can
never spam the approval queue or drain cash in auto mode on every monitor
poll. The audit trail keeps the fired record. A BUY ("add margin") condition
becomes a real margin transfer to the open position (it never sells the
position); a SELL condition reduces it.

## Security model

- The demo writes to `data/sim_state.json` only.
- The live adapter uses the official Binance MCP server; Binance itself
  enforces: dedicated Agentic sub-account, no withdrawal scope, and
  confirm-before-execute on every non-read action.
- No API keys are stored by the app; the live path uses OAuth/Bearer provided
  at runtime via env (`RS_MCP_ACCESS_TOKEN`).

## Testing

`tests/` covers the risk engine and feature surface (101 tests, no network —
everything runs against a fake market feed): liq price for long and short,
distance and risk zones, `required_cut`/`required_margin`, `margin_for_target`/
`releasable_margin`, the guardrail checks, the plain-language console parser,
plus feature tests for: dynamic-leverage opens, TP/SL automatic exits, the
always-on monitor (manual escalation queue, autonomous auto de-risk, retry
throttling), the funds round trip in both directions (spot ↔ futures wallet by
amount, bounded per-position margin release), console-driven open → add-margin
→ transfer → close lifecycles, the fixed add-margin condition semantics, the
one-shot price conditions (fire once, then closed and removed from state and
page), and the deliberate absence of any per-day budget / order-size cap /
cooldown / dust floor / leverage cap.
