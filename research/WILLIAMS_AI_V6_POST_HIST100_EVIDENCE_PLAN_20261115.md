# Williams AI v6 — next-stage evidence-first plan (2026-11-15)

## Verified baseline

- Real Binance USD-M historical H1 corpus: 25 instruments × 4 fixed snapshots = 100 charts, each with 1024 PRE-cutoff H1. Source ZIP/.CHECKSUM verification must be preserved, and all future bars excluded from model input.
- Kaggle T4 HF/NF4 experiment [43658377350](https://github.com/19wolfalone92/Williams-AI/actions/runs/43658377350) is terminal SUCCESS; [artifact 14138759783](https://github.com/19wolfalone92/Williams-AI/actions/runs/43658377350/artifacts/14138759783) contains verified 100 Base + 100 authentic trained v5 LoRA answers. GPU 98/100 initial 3200 token prompts, 2/100 fallback 4096 token prompts. No independent expert gold. `finish_reason=stop`: Base 43/100, LoRA 66/100; hence improvement in *format completeness*, not methodology accuracy.
- CPU GGUF/llama.cpp independent [43639364206](https://github.com/19wolfalone92/Williams-AI/actions/runs/43639364206) is still in progress. Three 10-chart shards have verified artifacts; remaining jobs must be allowed to complete before publishing a 100×2 CPU outcome. CPU and GPU are distinct inference protocols, not directly comparable accuracy runs.
- Final GPU findings: [verified report](research/WILLIAMS_V6_HIST100_100_BASE_100_V5_FINAL_GPU_RESULTS_20261115_RU.md). Preserve immutable raw model output as baseline. Neither historical inference nor a better-formatted JSON establishes Williams trading correctness.

## P0 — Finish independent CPU (no blind restarts)

1. Track exact statuses, timestamps, errors, artifacts for all ten independent GGUF CPU shards and final aggregate; don't treat queued as success.
2. Require 10/10 successful shards, 100 distinct IDs per role, identical chart/prompt SHA for each Base-v5 pair, pinned base/trained weights, and `complete=true`. Missing results block any complete claim.
3. Produce separate CPU outcome with case-level difference analysis, split by TC1/NTD/TC2/COURSE and generation status; explicitly label backend/context differences versus GPU.
4. Do not change frozen prompts, training weights, historical sample, CPU's model conditions or `main` to improve retrospective metrics.

## P1 — Fix technical decoder and abstention semantics, new experiment version

1. Build an auditable error taxonomy: invalid JSON, disallowed category, non-JSON prose, missing fields, `max_new_tokens` truncation, missing rule evidence, genuinely intentional abstention, insufficient brokerage position data, and infrastructure exception.
2. Record raw output, parsing error, EOS token, token count, model inputs and finish status. **Never** relabel technical `error` / `length` as a valid `ABSTAIN` choice. Keep previous raw outputs immutable.
3. Prototype constrained JSON decoding (e.g. schema/grammar or equivalent), explicit stop rules, `attention_mask`, token budgets, and reason length limits. Avoid leaking future or expert answers into prompts.
4. Verify on unit/integration fixtures: malformed output, longest wave reasoning, missing TC2 position state, invalid stop-order setup, old model regression, exact output schema. Only run a versioned A/B inference on SAME 100-case inputs as a *development* diagnostic; do not call this an unbiased holdout.
5. Report **valid-format rate**, **meaningful abstention rate**, **technical abstention rate**, explanation consistency, case-level regressions and resource/time cost separately.

## P2 — Create independent rule-grounded expert gold

1. Cross-check actual TC1/NTD/TC2/Profitunity pages, diagrams and text citations. Track edition, translation discrepancies and image/diagram interpretation. OCR alone cannot certify wave-degree or chart arrows.
2. Prepare a chart decision rubric for each category: entry preconditions; stop entry/invalidation; existing position state needed for TC2; wave-degree ambiguity and confidence; evidence chain tied to source page/figure and OHLC cutoff. Add legitimate 'NOT DETERMINABLE' labels where essential inputs are absent.
3. Ensure at least two blinded expert assessments on genuinely independent charts. Experts see neither Base/LoRA choices nor future outcomes. Resolve disagreements by adjudication, preserving raw labels, citations and reasons.
4. Audit train/validation/test near-duplicates, timestamp overlap, prompt leakage and source-book contamination; produce explicit leakage declaration. Current 100 retrospective cases may be development data and are not automatically independent gold exam.
5. Freeze benchmark provenance, primary book references and evaluation protocol *before* model iteration.

## P3 — Quantitative scientific comparison

1. Separately report eligible coverage, valid JSON rate, factual source faithfulness, expert decision correctness, conditional correctness among valid outputs, good vs bad abstentions, hallucinated rule/diagram citations, TC2 position-state handling, and wave ambiguity.
2. Compare Base vs authentic v5 on identical inputs and decoding conditions; confidence intervals and confusion matrices by book and signal type. Include a rule-based baseline and a deliberately simple baseline.
3. Define preregistered acceptance gates using actual error severity and risk: no future data, no fabricated brokerage state, zero unguarded live orders, all decision rationales linked to auditable rules, no regression on critical stop/exit conditions. Don't choose a 95% or 99% threshold retrospectively to make results look good.
4. Block any deployment claims unless full expert gold and prospective evidence pass.

## P4 — Targeted training after diagnosis, not '50 retrains' blindly

1. Build an expert-approved high-quality dataset of positive AND negative examples (including ambiguous cases and correct refusal). Keep diagrams/visual interpretation separate from OCR-only labels.
2. Correct erroneous source examples/labels; handle underrepresented wave setups and TC2 exit states with legitimate source-backed examples.
3. Run a reproducible finite GPU experiment matrix (Qwen Base, v5 LoRA, decoder-only, data-augmented LoRA, and ablations), log exact seeds, hardware, adapters, step count, checkpoints, dataset hashes and holdout.
4. Select checkpoints on training/validation data only; do not use the final future snapshots to tune weights or prompts.

## P5 — Prospective frozen H1 exam

- Respect the existing locked 32-case future H1 pilot timeline (2026-11-22) and later prospective assessment (2027-01-04). Collect future samples strictly after training/protocol freeze and before outcome horizon.
- No manual label tinkering after observations. Obtain independent expert adjudication and evaluate decision validity, utility under realistic fees/slippage, downside risk, confidence calibration and drift.
- Do not confuse numerical market direction with correctness under Bill Williams rules.

## P6 — Gated Williams Bot integration, separate repository

- Williams AI and Williams Bot are separate products. Do not merge the research model into `19wolfalone92/Williams` just because JSON validity improved.
- Only after a science+execution safety gate, consider a **read-only advisory/shadow mode** with full decision trace, disagreement alerts and verified signal explanations; bot's deterministic TC2/SL, margin/risk limits, order reconciliation and hard kill-switch remain independently validated.
- No live trading or external account actions in this research plan.

## Required delivery artifacts

1. One immutable CPU/GPU comparison report with raw run/commit/artifact links and verified case-level provenance.
2. Decoder root-cause matrix, regression suite and versioned before/after reproduction.
3. Independent expert rubric with book figure/page citations and label audit.
4. Independent gold evaluation and prospective holdout with uncertainty intervals and documented model limitations.
5. A formal GO/NO-GO recommendation for shadow use in Williams Bot; no implied go-live.

### Stop rules

If any model receipt is incomplete, contains placeholder predictions, has mismatched case/prompt SHA, or misses trained-adapter identity, stop and mark **INCOMPLETE**. If gold labels, leakage audit or prospective evidence are unavailable, report **accuracy unknown** regardless of JSON validity or directional market movements. Do not claim to have finished unfinished CPU jobs.
