# Williams AI v7 — read-only advisory architecture boundary

**Date:** 2026-11-15  
**Status:** Research architecture note only. No implementation, integration, trading, order routing, main merge or Williams Bot changes.

## 1. Purpose and non-goal

Williams AI v7 may be evaluated as a source-grounded research assistant that interprets closed historical chart context and returns a bounded hypothesis with cited book evidence. It is **not** a trading engine, signal publisher, portfolio manager, execution planner or autonomous agent.

This document records a possible future separation boundary only. It does not authorize connecting Williams AI to the Williams Bot.

## 2. Read-only data path

```text
Official public market archive / approved historical source
        |
        v
Immutable market snapshot + source SHA + cutoff timestamp
        |
        v
Causal chart-context builder (closed bars only)
        |
        v
Edition-scoped source retrieval (TC1 / NTD / TC2 / COURSE)
        |
        v
Qwen base or pinned v5 adapter + pinned decoder
        |
        v
Strict response validator
        |
        v
Research-only advisory record + full provenance
        |
        v
Human review / evidence backlog
```

Every result must retain:
- market snapshot and chart-window SHA-256;
- prompt SHA-256 and source/protocol SHA-256;
- model base revision and adapter SHA-256;
- decoder schema/version and generation parameters;
- retrieved source IDs and edition boundaries;
- cutoff timestamp and exact closed-bar count;
- raw technical finish state and normalized validation result.

No result may be represented as a broker order, position, fill or realized PnL.

## 3. Output contract: evidence, hypothesis and technical state are separate

A future read-only record should separate at least:

- `research_status`: `HYPOTHESIS`, `INSUFFICIENT_DATA`, `ABSTAIN` or `UNRESOLVED`;
- `technical_status`: `VALID`, `INVALID_JSON`, `TRUNCATED`, `MISSING_EOS`, `TIMEOUT` or another explicit infrastructure/decoder failure;
- `edition` and cited source passages;
- `observed_closed_market_facts`;
- `interpretation` and its uncertainty;
- `alternative_hypotheses`;
- `provenance` and immutable hashes.

A valid `ABSTAIN` is a model outcome. An invalid JSON response, timeout or truncation is a technical failure. Neither should be converted into the other. Format-completion rate is not book accuracy, trading accuracy or profitability.

The advisory record must not contain executable order instructions, exchange parameters, leverage, position sizing or broker mutation identifiers.

## 4. Hard isolation boundary from Williams Bot

Until a separately approved GO decision:

- no Williams Bot dependency, SDK, shared runtime module or database write;
- no exchange credentials, API keys or broker connector in the v7 research runtime;
- no order endpoint, order intent, cancel/replace, leverage, margin or position mutation capability;
- no Android service, background task or autonomous trading hook;
- no import of v7 research decisions into the bot's state machine;
- no PR targeting `main`, no merge and no changes to the trading repository.

If a future GO is ever approved, the first permissible integration design would still be a one-way, read-only advisory envelope consumed by a separately controlled human-review interface. It would not be an execution interface. A new safety review and explicit authorization would be required before any additional capability.

## 5. Evidence and training gates

The current state remains:

- ten source-pinned review targets have primary-source recheck notes;
- the source-review tests pass in CI;
- independent expert reviewers verified: **0**;
- approved expert labels: **0**;
- valid gold labels: **none**;
- training authorization: **false**;
- new LoRA training: **blocked**;
- independent expert exam: **blocked**.

Before any future training:
1. Obtain original, source-pinned chart cases with causal cutoffs.
2. Obtain at least two independent expert reviews per approved case, including signed rationales and disagreement resolution.
3. Keep reviewer labels outside model inference and checkpoint selection.
4. Freeze train/validation/test partitions by case identity, chart SHA and prompt/source protocol SHA.
5. Prove no overlap or information leakage before training.
6. Run a separate post-training exam on unseen, expert-approved cases.
7. Require an explicit GO/NO-GO review before considering any product integration.

Historical100 model predictions, model-generated rationales, and the prospective H1 technical collection are not expert gold and cannot be promoted into labels by agreement or majority vote.

## 6. Decoder v7 evidence boundary

The eight-case T4 pilot is a format-only experiment on identical frozen prompts. It showed that guided decoding did not improve technical completion on that small sample: Base stop decreased from 3/8 to 2/8; authentic v5 LoRA stop decreased from 5/8 to 4/8. The remaining invalid JSON outputs are technical failures, not model decisions. No book-quality conclusion follows.

The raw artifact also contains a metadata naming caveat: `symbol` is the case-group code (`01`/`02`), while the actual exchange symbol is in `original_symbol`. Pairing was independently verified through the frozen IDs and hashes. Correct that schema in a future artifact version; do not re-run the completed T4 pilot solely for this metadata issue.

## 7. Prospective H1 pilot boundary

The frozen 32-case H1 pilot remains a separate, future public-data collection:
- wait until the predeclared cutoff plus two hours UTC;
- use only official `data.binance.vision` daily H1 ZIP archives with verified `.CHECKSUM` SHA-256;
- 32 cases × 1,024 closed H1 bars;
- no expert gold, no holdout claim, no GPU, no keys, no trade and no main merge;
- distinguish unavailable source archives (HTTP 404) from model or pipeline failure;
- preserve the original case plan and cutoff; never synthesize missing bars.

The pilot is a prospective technical/data-quality collection, not an independent expert exam.

## 8. Current GO/NO-GO

**NO-GO for integration, execution, new training and claims of trading qualification.**

Permitted work before a future decision is limited to:
- primary-source rechecks and regression tests;
- provenance and decoder diagnostics that do not require a new heavy inference run;
- read-only architecture documentation;
- the separately scheduled prospective H1 public-data pilot after its cutoff.

This note is documentation only. It changes no runtime, code path, credentials, model weights, trading system or repository default branch.
