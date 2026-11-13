# Williams AI v5 — Stage 1 evidence audit (2026-11-13)

**Decision: RESEARCH BLOCKED.** No deploy, no GPU retraining, no trading credentials, no `main` merge.

## Immutable evidence and scope

- Real GitHub Actions [CPU recovery + exam #43229923254](https://github.com/19wolfalone92/Williams-AI/actions/runs/43229923254), run head `e3f1b4d34059f10adb1b2791267bb11b79ec67c1`, job `131222216812`, concluded success.
- Downloaded [artifact #13962336774](https://github.com/19wolfalone92/Williams-AI/actions/runs/43229923254/artifacts/13962336774) and independently unpacked `williams_v5_encrypted/v5_training_audit.json`, `williams_v5_knowledge_exam.json`, the curriculum manifest and encrypted LoRA archive.
- A newer gate change `6fd23442b0e901a7b6c652e92c53ed83229863c0` was committed **after** this workflow started. The actual result was therefore rechecked across *all four* books, regardless of the workflow's original `accepted` value.
- ZIP upload digest reported by GitHub: `sha256:6ebaa38558fa4edc7423f398fc0574f807e3eb2ce74ae193ea076ae781fe9d7a`.
- Sealed model archive: recomputed SHA-256 `4cc11816cc0b0acea494e9db84189d8203ebbfc79bf38c73e0f69cd14bcfeb66` matches training audit. Raw adapter SHA `8bd26590df627c7452d56e81df6a4572885bfc82515259a7c9de4a3970c850b8` agrees *between reports*, but raw adapter bytes cannot be independently rehashed without decrypting the sealed file.
- Training receipt records **80 optimizer steps**, selected checkpoint 80, train=300, validation=48, test=84, zero test cases passed to trainer. No GPU run was initiated during the recovery workflow.
- Reported overall validation loss **3.7182102203 → 1.4626840353**, relative decrease **60.66%**. All four book-specific validation loss values decreased. Loss improvement is **not** an independent measure of rule interpretation accuracy.

## Recounted comparable exam results

| Cohort | Base Qwen3-4B | V5 LoRA | Interpretation |
|---|---:|---:|---|
| Reused 16 closed book | 16/16 | 14/16 | V5 worse by 2 |
| Reused 16 source-assisted | 16/16 | 15/16 | V5 worse by 1 |
| Developer synthetic, rule-disjoint 28 | 23/28 | 24/28 | V5 better by 1; not independent expert charts |

### Book-level results and independently recomputed gate

| Book | Closed book (base → v5) | With sources (base → v5) | 28-rule test (base → v5) |
|---|---|---|---|
| TC1 | 4/4 → 3/4 | 4/4 → 4/4 | 6/8 → 7/8 |
| NTD | 4/4 → 3/4 | 4/4 → 4/4 | 5/8 → 6/8 |
| TC2 | 4/4 → 4/4 | **4/4 → 3/4** | **8/8 → 7/8** |
| COURSE | 4/4 → 4/4 | 4/4 → 4/4 | 4/4 → 4/4 |

**Recomputed four-book gate: FAIL**, with three specific blockers:
`SOURCE_ASSISTED_WORSE_THAN_EQUAL_CONDITIONS_BASE`,
`SOURCE_ASSISTED_TC2_REGRESSION_VS_BASE`,
`HELDOUT_TC2_REGRESSION_VS_BASE`.

### Actual 28-case failure forensics

Candidate V5 has **4 failed / 28**:
- TC1: `V5-eecd724e72cf0b0b`: chose A; incorrect; also incorrect in base.
- NTD: `V5-36b17184574541af`: chose A; incorrect; also incorrect in base.
- NTD: `V5-bcabcf5d34e9485e`: chose B; incorrect; also incorrect in base.
- TC2: `V5-f1e841d72a5345bf`: **no parsed choice / abstention**, initial `finish_reason=stop`, **no fallback**; base chose B and was graded correct. Do NOT label this as proven incorrect TC2 reasoning: without the raw completion it could be a deliberate abstention or a decoder/format failure.

Matched improvements: TC1 `V5-28206ca5568e4324` and NTD `V5-864e07682572ff8c` were wrong in base and correct in V5. The TC2 abstention erases one net gain.

Recovery occurred on **10/28 V5 cases (10 recorded correct)** and **21/28 base cases (16 correct)**. This is post-processing of model answers, not evidence that the model learned those ten rules. V5's 16 source-assisted cohort had 0 abstentions; its 16 closed-book cohort had 1. Source-assisted TC2 remains the primary release blocker.

## Hard limits of evidence

The artifact includes 28 *case IDs, chosen letters and correct/abstention labels* but **does not export the corresponding gold keys, raw model outputs, question texts or source passages**. The two 16-case cohorts contain only per-book aggregates, **no per-question rows**. Therefore this audit independently verifies tally consistency and provenance, but cannot adjudicate semantic correctness or diagnose every failure's causal path. The four-book source PDFs and their diagrams were **not** fully reviewed in this stage; the manifest explicitly states **0 independently reviewed diagrams** and **0 approved market labels**.

This is a repeat on developer-derived known exercises, **not an untouched benchmark**. Old checkpoint500 differs in case-id and RAG/decoder contract and is **not** a matched control. No accuracy target such as 95% is established from these results.

## Next controlled steps

1. Export **redacted case-level evidence** on the next *permitted* CPU-only diagnostic: question or immutable question hash, separate expected answer (examiner side only), parsed answer, stop reason, fallback reason, book/rule/page citation, and abstention-vs-malformed flag. Never include gold keys in model prompts.
2. Track the **TC2 abstention** separately from source-retrieval and semantics; use raw tracing to determine the root cause before considering a fix.
3. Populate a source-backed registry with explicit original edition/page/figure and reviewer sign-off for all four books; flag OCR/diagram uncertainties as unverified, never invent citations.
4. Freeze a **new independent expert-authored chart exam** before inference. Do not reuse the 28 known prompts for optimization; no new GPU until credible independent data exists.

**Stage 1 outcome:** artifact verified and regression localized to source-assisted TC2 and one rule-disjoint TC2 abstention; deeper semantic root cause **blocked by missing per-case/source/raw completion evidence**. Never auto-promote the adapter.
