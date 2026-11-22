# Williams AI v5 — Stage 1: error audit and independent evaluation protocol

Status: RESEARCH ONLY / BLOCKED FOR DEPLOYMENT. No GPU training, no trading keys, no main merge.

## Evidence and provenance
- CPU recovery/exam: https://github.com/19wolfalone92/Williams-AI/actions/runs/43229923254
- Run SHA: `e3f1b4d34059f10adb1b2791267bb11b79ec67c1`.
- Later quality-gate change: `6fd23442b0e901a7b6c652e92c53ed83229863c0`. The run does not include this change.
- Artifact ID: `13962336774`. JSON reports are inside the workflow artifact, not root repository files.
- Previously reported scores (must be independently re-derived from the artifact before using as a release criterion): closed-book 14/16; source-assisted 15/16; rule-disjoint 24/28; baseline 16/16, 16/16, 23/28 respectively.
- Previously reported rule-disjoint per-book scores: TC1 7/8 vs 6/8 baseline; NTD 6/8 vs 5/8; TC2 7/8 vs 8/8; COURSE 4/4 vs 4/4. TC2 regression blocks promotion.

## Required stage-1 audit (do not mark complete without raw rows)
1. Extract `v5_training_audit.json` and `williams_v5_knowledge_exam.json` from the artifact. Verify checksums, training steps (80), validation-loss trajectory, and LoRA identity against manifest.
2. For each of the 60 exam decisions (16 closed-book, 16 source-assisted, 28 rule-disjoint), record case ID, book, gold answer, raw response, decoded response, fallback path, correct/incorrect, abstention and evidence citation. Count failures by book and evaluation split; never count fallback parsing as semantic correctness automatically.
3. Recompute the four-book gate from raw case-level results, not the stale `research_quality_gate.accepted` flag. Include TC1, NTD, TC2 and COURSE individually, and explicitly identify any regressions vs the matched baseline.
4. Inspect all wrong/abstained answers, classify: source retrieval, source contradiction, rule preconditions, wave/indicator logic, decoding/truncation, or label ambiguity. Link to source page and code path only where evidence is actually present.
5. Identify overlap with training examples and earlier tests; label this repeat evaluation on known tasks, NOT untouched holdout. Do not directly compare checkpoint500 under mismatched case_id/RAG/decoding formats.
6. Record unresolved source figure/OCR uncertainty separately from textual rule accuracy.

## Stage 2 — source-grounded rule registry
Create reviewed entries for each rule with `book, edition, page, figure, prerequisites, exceptions, action, abstain_when, conflicts, reviewer_status`. Keep unreviewed interpretations out of production. Track TC2-specific regression cases first.

## Stage 3 — independent expert exam
Freeze an unseen expert-authored set before inference, with time-separated chart cases, long/short/no-trade, conflicting signals, SL and exits, adversarial missing data, all four books, and double review. Hash and seal cases before running any candidate. Report per-book precision, abstention, critical errors and confidence intervals, not only pooled accuracy. Explicitly require no TC2 critical regression.

## Stage 4 — architecture decision
Compare source-backed RAG + deterministic rules against base and LoRA on the SAME new evaluation protocol. Reject unsupported trades; execution remains separate and disabled. Only consider training after new independent expert data and a documented error analysis. Do not tune to the sealed exam.

## Current decision
BLOCKED: no deployment, no GPU, no main merge. This document is an audit worklist, not a claim that case-level evidence has been inspected or errors repaired.
