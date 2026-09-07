# Submission dossier — Liquidation Guardian · Binance Agent OS Mini Hackathon

Track A entry. Deadline **2026-09-08 23:59 UTC**. Repo head: Phase 3
(`tests/` = 100+ passing). This dossier maps every expected submission/survey
item to concrete evidence so the reply can be assembled in minutes.

> Public URL to confirm at publish time: `https://github.com/Aguimmanuel/liquidation-guardian`
> (push is intentionally pending — do not publish until you say "push").

---

## 1. The one-line pitch

> **Liquidation Guardian** — an AI agent that keeps leveraged futures positions
> from being liquidated: live distance-to-liquidation math on Binance prices,
> guardrailed de-risk proposals (reduce vs add margin) you approve — or an
> explicit, consent-gated Auto mode — plus one-shot price conditions and
> per-position TP/SL. Manual by default; built on Binance Agent OS's MCP server.

## 2. Track & eligibility check

- **Track A** ("Build an AI agent with Agent OS", $20k pool): video demo +
  public GitHub repo, replied to the announcement post. ✅ this repo + `DEMO_SCRIPT.md`.
- **Track B** (participation): connect MCPs **and trade** live — needs a funded
  Agentic sub-account; incidental, not claimed in SIM demos.
- Eligibility: Nigeria OK; **do not enter from US/UK/EEA/HK/SG** (per Binance rules).
- Entry mechanics: follow + repost @Binance; reply/quote with submission; complete survey.

## 3. Survey/field mapping (fill fast)

| Likely field | What to paste/link |
|---|---|
| Project name | Liquidation Guardian — risk-protection agent for Binance futures |
| Category / track | Track A — AI agent built with Agent OS |
| Repo | `https://github.com/Aguimmanuel/liquidation-guardian` (public after push) |
| Demo video | YouTube/Vimeo/Twitter link to the recording made from `DEMO_SCRIPT.md` |
| Elevator pitch | Section 1 above |
| What you built | Agent console + deterministic risk engine (liq math, OK/WATCH/DANGER zones, required_cut / required_margin, deleverage-by-keeping-margin), always-on monitor, guardrail checks, manual-by-default propose-and-approve flow, consent-gated auto mode with written rationale, one-shot price conditions, TP/SL auto-exit, fund transfer (spot↔futures wallet), live adapter to the official Binance Agent OS MCP server, SIM mode on live prices. |
| How it uses Agent OS | Live mode (`RS_MODE=live`) executes through the official MCP server (`app/adapters/mcp_live.py`, `mcp>=2.2`), inheriting Agentic sub-account isolation (no withdrawal scope) and Binance's confirm-before-execute. |
| Safety / consent story | Manual is the default; automatic mode engages only after an on-screen consent notice; decisions are deterministic + guardrailed and the console states the reasoning; de-risk actions are never blocked by budgets/cooldowns. |
| Repo quality evidence | `python -m pytest tests/ -q` → 100+ passing, no network; README with refreshed screenshots; `docs/architecture.md`; `DEMO_SCRIPT.md`. |

## 4. Evidence map (for judges)

- Risk math: `app/agent/risk.py` (isolated margin, tiered maintenance-margin by
  notional — representative Binance 2026 snapshot, `USD_M_MMR_TIERS`).
- Execution model: `app/adapters/sim.py` (paper, live prices, taker fee on
  notional) and `app/adapters/mcp_live.py` (real Agent OS MCP channel, kind-aware
  dispatch, explicit timeouts, documented live subset).
- Orchestration/safety: `app/agent/orchestrator.py` (monitor-first sweep,
  mutual exclusion between de-risk and conditions, deterministic auto rationale).
- UI: `app/static/index.html` (account bar with Fund transfer tile, Trade panel,
  positions + TP/SL, Price conditions with armed/fired chips, Pending Proposals,
  console, onboarding tour).
- Tests: `tests/test_*.py` (features, risk engine, live adapter surface, API
  schemas/auth, market math, parsers, guardrails).

## 5. Demo integrity notes (keep visible)

- Demo runs in **Sim mode** (paper, live prices) — safe to show, zero funds.
- The console is a **deterministic engine with optional LLM narration**
  (OpenAI-compatible via `OPENAI_API_KEY`); it is not the Agent OS product and
  does not claim Binance's model stack — it uses Agent OS's **MCP execution
  channel** in live mode.
- Backtest/charts are illustrative, never a strategy claim; README carries
  "not financial advice".
- Public screenshots in `docs/screenshots/` were refreshed against this build.

## 6. Publish checklist

- [ ] `git push` after your explicit go-ahead (no push has happened for Phases 1–3).
- [ ] Confirm the public repo URL renders (README images load).
- [ ] Record + host the video from `DEMO_SCRIPT.md` (< 2 min, H.264 MP4).
- [ ] Follow + repost the announcement; reply/quote with repo + video + pitch.
- [ ] Complete the survey before 2026-09-08 23:59 UTC.
- [ ] Eligibility spot-check once more on submit day (location rules can change).
