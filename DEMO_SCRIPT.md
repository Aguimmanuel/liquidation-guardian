# Demo script & recording runbook — Liquidation Guardian

**Track A submission asset.** ~90–120 s screen recording of the app in **Sim
mode** (paper account on live Binance prices — zero credentials, zero risk),
with a narration line for each beat and the exact clicks to reproduce it.
Works against this build as of Phase 3 (`tests/` = 100+ passing).

---

## Before you press record (5 min)

1. `pip install -r requirements.txt && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000`
2. Reset to the clean demo story: open the app, then
   `curl -X POST http://localhost:8000/api/reset -H "Content-Type: application/json" -d '{}'`
   — the account is back to **100% spot, $10,000, no positions** (fresh demo starts
   there, and the Trade flow will explain that margin must be funded first).
3. Make sure no onboarding tour is open (`Skip tour`), and the agent badge reads **SIM**.
4. Prices are **live**; the script is written to stay DANGER regardless of normal
   drift (step 3 opens deep in the zone). If a beat depends on a crash, don't wait —
   the guardian acts at *your* distance threshold, not a price level.

## Suggested narration (TTS or voiceover)

> "Most agents tell you when to buy. Liquidation Guardian is the one that keeps you
> from getting *wiped out*. It watches your open futures against live Binance prices,
> computes the real distance to your liquidation price in real time, and proposes the
> guardrailed move — reduce or add margin — before the exchange liquidates you.
> Every action waits for your approval, unless you opt into Auto mode with your eyes
> open. Manual by default. Built on Binance Agent OS's execution model."

---

## Beat 1 — Hook: liquidation risk is real (0–10 s)

1. Console: type `status` (or look at the stats row). Narrate the account bar:
   Total / Spot / Futures equity / nearest-liq / **Fund transfer**.
2. Keep the camera on the ticker — coins are streaming live.
3. **Narration:** "This is a $10,000 paper account at live prices. Total is all spot
   right now — the futures wallet is empty, because that's exactly how Binance
   futures work: you fund it first."

## Beat 2 — Fund the futures wallet + open a position (10–30 s)

1. In the **Fund transfer** box (account bar, rightmost tile): type `2,000`, `↓ to futures`.
2. Approve the pending proposal. Spot drops to ~$8,000, futures wallet shows $2,000.
3. In the **Trade** panel choose **BTC**; set leverage **20x** and margin **$600**, click
   **Open Long** → **Approve**.
4. Narrate the position row: entry, mark, **liquidation price**, distance **~4.5%**,
   badge **DANGER**.
   - The 2-second guardian sweep will queue a de-risk plan by itself — this is the
     "always-on guardian". Point at the 🛡️ badge count.
5. **Narration:** "20x on BTC puts liquidation about four and a half percent away.
   The guardian flagged it DANGER immediately and is already proposing a plan."

## Beat 3 — The protect moment (30–55 s)

1. The **Pending Proposals** card shows the queued de-risk (reduce or add margin).
   Click the **reduce** proposal → **Approve**.
2. Watch liquidation headroom jump: position shrinks a little, distance goes from
   **~4.5% → ~15%** (de-risk target), risk badge → **OK**.
3. **Narration (the thesis):** "Here's the insight: closing part of the position and
   *keeping the margin on the rest* deleverages you, so the liquidation price moves —
   not just the size. Reduce needs no extra capital, so that's what it proposed."

## Beat 4 — Conditions are one-shot; TP/SL auto-close (55–80 s)

1. Open **Price conditions** (+). Against the BTC position that is still open after
   Beat 3, arm a **one-shot** trigger within ~1% of the *live* price so a normal
   wiggle fires it, e.g. `BTCUSDT`, **ABOVE** `live+1%`, side **BUY** (add margin),
   amount `$300`. (The app shows the live price; if it doesn't fire in ~30 s, re-arm
   closer — a condition needs a fresh crossing past the trigger.)
2. Watch it fire once: chip flips **armed → fired · closed**, one proposal appears,
   and no further sweeps ever re-propose it.
3. Show **TP/SL**: in the position row arm a take-profit and a stop-loss.
4. **Narration:** "A condition fires exactly once, then closes itself — armed, fired,
   done. No loops, no surprise repeat orders. Take-profits and stop-losses are
   pre-authorized exits: when crossed, they close right away, exactly like an
   exchange-side stop."

## Beat 5 — Auto mode with the consent notice (80–105 s)

1. Flip the header toggle to **⚡ Auto**. Read the consent notice out loud
   ("agents can make mistakes…").
2. Drop a fresh 25–50x position (deep in DANGER) — the guardian de-risks it **without
   waiting**, and the console explains the choice in plain English
   ("Why: chose reduce over add-margin — no extra capital…").
3. Flip back to **Manual**.
4. **Narration:** "Automatic mode still obeys every guardrail, and it tells you why it
   chose what it chose. I can always flip back to approving every step."

## Optional beats (time permitting / if funded)

- **Backtest (illustrative):** `python scripts/backtest.py --days 90` and show the
  guardian-vs-hold chart for ~10 s (say "illustrative, not a strategy claim").
- **Real MCP beat:** if you can fund a tiny Agentic sub-account, set `RS_MODE=live`
  and record the actual **Binance confirm-before-execute** prompt appearing from our
  adapter — that single shot proves "connects to Agent OS for real".

## Wrap line

> "Most AI agents want to trade for you. This one stops you from getting liquidated —
> manual by default, automatic only with your eyes open, built on Binance Agent OS."
> + repo URL.

## Recording checklist

- [ ] 1080p+, whole browser window, natural tab focus; no other windows.
- [ ] Show the repo briefly (README) at the very end, or in the post text.
- [ ] Stay honest: paper mode throughout; no P&L promises; "not financial advice".
- [ ] Keep it under 2 minutes; the hook (Beat 1) must land in the first 10 seconds.
- [ ] Final cut export H.264 MP4 (Twitter/X friendly).
