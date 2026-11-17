# Williams AI → Williams Bot: Stage 9 integration GO / NO-GO specification

**Status as of 15 November 2026: NO-GO.** The AI is an unapproved research model with zero independent gold chart labels. This file is a future safety contract only. No code is added to the separate Williams Bot repository.

## Non-negotiable repository and authority boundary
- Williams AI (Qwen/LoRA) is a research/advisory component.
- Williams Bot remains the only owner of its deterministic Bill Williams strategy state, risk engine, state machine, Execution Door, Binance USD-M Futures API and order lifecycle.
- AI receives a sanitized, read-only market snapshot only. It does not receive exchange API keys, signed credentials, order methods, account mutation tools or Execution Door privileges.
- AI cannot create, change, cancel or replace any exchange order; it cannot alter stop-loss, position size, leverage, margin mode, campaign state or risk limits.
- An AI answer is never proof of an exchange position, order acknowledgement or fill.
- An opposite AI suggestion cannot close or reverse the Bot's current position.

## Gates before any shadow connection
All are required; one failed gate keeps NO-GO:
1. **Independent book source review:** all 4 editions, all claimed rule coverage, translation discrepancies (including NTD Alligator Lips shift), and source diagrams checked by independent qualified reviewers.
2. **Independent gold chart set:** at least two externally verified reviewers, blinded from model output, agreeing on rationale and source pages. Disagreements remain unscored/untrained.
3. **Leakage proof:** training, dev and sealed holdout cohorts frozen before model evaluation; historical 100 and future H1 exam cohorts never fed into training.
4. **Reasoning evidence:** every model claim traces to exact closed bars, book rule id, edition/page, counter-evidence, timeframe and missing data. No citation to a source card alone without showing the actual market condition.
5. **Error and calibration:** valid-format and technical failure rates reported separately; calibrated abstention/OOD; no catastrophic opposite-side signals or invented position state on holdout. Confidence cannot be self-reported logits alone.
6. **Prospective performance:** locked future H1 cohort evaluated after its cutoff without changing the questions, bars, weights or thresholds.
7. **Safety regression:** all fail-closed, duplicate-intent, restart/reconciliation, stale snapshot, partial-fill, latency and broker state fault-injection tests pass.
8. **Human and operational approval:** explicit independent sign-off for a shadow/advisory-only deployment. A green unit test or successful CI does not constitute approval.

## Allowed deployment stages after all evidence gates
- **S0 — Offline research (current):** local/historical chart reader, no order access.
- **S1 — Read-only shadow:** Bot provides sanitized closed-bar snapshots; AI returns a proposed interpretation to a separate log. Its output is not visible to the Execution Door and does not alter the deterministic Bot decision. Compare model proposal with Bot and later expert labels.
- **S2 — Human advisory display:** only after successful S1 safety review, show evidence and source citations to a human operator; the human remains responsible for every action. No order actions.
- No autonomous AI trading phase is specified or authorized by this document.

## Required response schema
Every AI advisory response must include:
- schema_version, model_id, weights_sha256, prompt_sha256, snapshot_sha256;
- symbol, decision_time, closed_bar_cutoff, timeframe, higher-timeframe context;
- edition, exact rule_ids, page references, retrieved source-card ids and raw source integrity;
- observations (each tied to bar timestamp / OHLCV evidence);
- required_conditions, met_conditions, failed_conditions, unknown_conditions;
- wave degree and alternatives where relevant;
- decision or explicit ABSTAIN, reason, limitations;
- position_state only from broker snapshot; if absent, UNKNOWN;
- execution_authority: NONE, trade_action: null.

Any missing SHA, stale timestamp, unexpected field, unsupported rule edition, unfinished candle, future leak, invalid enum, incomplete reason or unavailable broker fact causes a hard reject/ABSTAIN. Never reinterpret malformed JSON as a model-chosen ABSTAIN.

## Shadow success criteria
- zero calls to order/mutation interfaces;
- zero changes in deterministic Bot output, risk and position state due to AI response;
- every AI event is idempotently logged, reproducible by pinned model/prompt/data bytes, and can be replayed;
- 100% traceability of non-abstain claims to closed evidence and book-rule cards;
- all technical errors distinctly surfaced from reasoned abstentions;
- independent reviewer evaluates a blinded sample and every safety-critical disagreement;
- kill switch can disable AI visibility without restarting or altering the Bot.

## Current decision
**NO-GO for code integration, order integration, or deployment.** Research-only branch research/williams-v6-post-hist100-evidence-20261115 is the only place where this design may evolve. Main and 19wolfalone92/Williams remain untouched until all gates are independently closed.
