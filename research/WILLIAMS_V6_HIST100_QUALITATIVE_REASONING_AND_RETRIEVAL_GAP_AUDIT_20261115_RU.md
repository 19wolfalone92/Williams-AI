# Williams AI v6 — qualitative audit of frozen HIST100 Base/v5 answers, retrieval and four-book gaps

**Evidence only; not accuracy.** Analysed the two immutable 100-case artifacts after their completion. Each backend has 100 Qwen Base + 100 genuine v5 LoRA responses. No human labels exist for these 100 charts; no statements below are expert gold.

## 1. Exact retrieved source cards were identical across all 25 cases per book

| Edition / task | Retrieved Rule IDs, every case | Main missing rule context |
|---|---|---|
| TC1 / ENTRY | TC1.FRACTAL, TC1.PSY.FEAR, TC1.W5 | Rule-ranking brings a psychological card into an entry task; no dedicated entry profile for the five Magic Bullets, MFI windows and wave/entry sequence. Historic prompt itself asks for “AO and Alligator” in TC1, although current TC1 registry has no TC1.Alligator rule id; edition-boundary ambiguity must be resolved explicitly, not by allowing another edition's paraphrase. |
| NTD / ENTRY | NTD.BALANCE, NTD.ZONE.ADDED, NTD.AC | Missing NTD.FRACTAL, NTD.ALLIGATOR and NTD.AO; these define prerequisite context/direction, the first external fractal and the three distinct AO signals. The current retrieval shows why the 3-card ceiling is insufficient for the book's combined entry reasoning. |
| TC2 / EXIT | TC2.WM2, TC2.CAMPAIGN, TC2.WM3 | Missing TC2.REVERSE and WM1 bar/angulation/trigger/stop rules; yet the static question explicitly asks for opposite Wise Man signal and reversing a hypothetical long position. Retrieved cards are mainly entry/addition rules, not the critical reverse/exit conditions. |
| COURSE / WAVE | COURSE.W4.ZERO, COURSE.AO5345, COURSE.TRADING_PARTNER | Mostly on-topic, but Trading Partner is a broad decision worksheet; degree selection, visual nesting and detailed W4/W5 structure are not guaranteed by this three-card selection. |

This pattern is deterministic: all 25 cases of each edition receive the same retrieval set because the question text is fixed by book. This does not adapt evidence retrieval to the actual chart setup or a particular wave/entry/exit subproblem.

## 2. Generation/decision outcome counts by book

These are the raw frozen artifact counts. Error and length are technical generation outcomes, not model-chosen ABSTAIN. Valid ABSTAIN, INSUFFICIENT_DATA, HOLD etc. are model outputs, not confirmed correct decisions.

### CPU GGUF / llama.cpp

| Book | Model | stop | error | length | Decisions |
|---|---|---:|---:|---:|---|
| TC1 | Base | 24 | 1 | 0 | ABSTAIN 25 |
| TC1 | v5 LoRA | 24 | 1 | 0 | ABSTAIN 25 |
| NTD | Base | 24 | 1 | 0 | ABSTAIN 19, NO_SETUP 6 |
| NTD | v5 LoRA | 24 | 1 | 0 | ABSTAIN 25 |
| TC2 | Base | 23 | 1 | 1 | HOLD 22, ABSTAIN 2, INSUFFICIENT_DATA 1 |
| TC2 | v5 LoRA | 23 | 1 | 1 | INSUFFICIENT_DATA 22, ABSTAIN 2, HOLD 1 |
| COURSE | Base | 24 | 1 | 0 | WAVE_HYPOTHESIS 24, ABSTAIN 1 |
| COURSE | v5 LoRA | 24 | 1 | 0 | WAVE_HYPOTHESIS 24, ABSTAIN 1 |

### GPU HuggingFace / NF4

| Book | Model | stop | error | length | Decisions |
|---|---|---:|---:|---:|---|
| TC1 | Base | 25 | 0 | 0 | ABSTAIN 25 |
| TC1 | v5 LoRA | 25 | 0 | 0 | ABSTAIN 25 |
| NTD | Base | 16 | 9 | 0 | ABSTAIN 14, INSUFFICIENT_DATA 11 |
| NTD | v5 LoRA | 3 | 20 | 2 | ABSTAIN 25 |
| TC2 | Base | 2 | 23 | 0 | ABSTAIN 23, INSUFFICIENT_DATA 2 |
| TC2 | v5 LoRA | 25 | 0 | 0 | INSUFFICIENT_DATA 25 |
| COURSE | Base | 0 | 25 | 0 | ABSTAIN 25 |
| COURSE | v5 LoRA | 13 | 4 | 8 | WAVE_HYPOTHESIS 13, ABSTAIN 12 |

The GPU's two final cases were processed using a 4096-token fallback, unlike the first 98 cases capped at 3200; do not treat the full 100 as one homogeneous protocol. CPU and GPU also use different runtime/decoder/backends, so never compare as if only weights differed.

## 3. Qualitative reasoning observations

### TC1
- CPU Base and v5 LoRA abstain on all 25 cases; LoRA explanations frequently mention AO/AC and Alligator. The current TC1 rule registry uses TC1 fractal / wave / Magic Bullets, not a TC1 Alligator card. This is a cross-edition vocabulary/criterion risk and warrants source mapping review.
- The fixed historical question itself asks TC1 to assess “AO and Alligator”. That question may conflate TC1 concepts with other Williams editions. It cannot be repaired in the frozen set; a future exam must split the exact book-specific methodology.
- Most explanations are generic lack-of-confirmed-fractal/oscillator evidence rather than a clear ordered test of the book's specific entry trigger.

### NTD
- CPU LoRA frequently reports insufficient data and sometimes gives exact AO/AC/Balance Line counts. These counts need a rule-by-rule expert audit; the model cannot be judged right merely because it mentions a number.
- GPU LoRA produces only three successful responses out of 25; 22 are technical error/length, not meaningful refusals. Most valid reasons describe broad absence of AO/AC/Balance Line evidence rather than identifying which exact NTD pattern (Saucer, Zero Line Cross, Twin Peaks, AC count, Zone or Balance Line) was tested.
- The current 3-card retrieval omits prerequisite NTD Alligator/Fractal and distinct NTD AO patterns. This is a concrete retrieval defect independent of LoRA weights.

### TC2
- The frozen prompt asks about an opposite Wise Man signal and reversing an assumed long position, but omits a complete position/broker-state record. It is a conceptual hypothetical, not a verified real position.
- CPU Base mostly says HOLD while CPU v5 mostly says INSUFFICIENT_DATA. No gold exists to establish which category is correct.
- GPU LoRA valid outputs all say INSUFFICIENT_DATA and mention WM2/WM3/campaign; retrieval omitted TC2.REVERSE and WM1 angulation/trigger rules, which are central to a possible stop-and-reverse. This is not evidence of correct knowledge; the required source cards were not included.
- A future TC2 exit case must provide a hypothetical educational position explicitly separate from a verified broker position, and ask only about book-level exit logic. Broker execution and fill remain UNKNOWN unless supplied independently.

### COURSE
- Valid Course reasons often correctly distinguish AO zero-line crossing as the minimum W4 requirement from actual W4 completion and call for additional A-B-C/triangle/subwave confirmation. This is consistent with the inspected Course pages 43–45, but not an expert accuracy score.
- Course inference still needs explicit wave degree, nested parent-child count and enough closed bars; if absent, WAVE_HYPOTHESIS/INSUFFICIENT_DATA is a cautious model response, not proof of correct wave count.
- GPU Base had 25 technical errors. GPU v5 had only 13 valid reasons and 12 technical abstentions (4 error + 8 length). This indicates substantial format/context reliability problems regardless of qualitative reasoning.

## 4. Corrective changes prepared for a future version (NOT applied to frozen v6)

1. Replace the TC1 historical question that mixes Alligator/AO vocabulary with edition-specific tests for TC1 Fractal, MFI/Profitunity Windows, Five Magic Bullets, Zero Point and wave degree. Retain old HIST100 exactly as-is.
2. Add book+family retrieval profiles. NTD ENTRY should guarantee Alligator/Fractal, AO, AC, Zones and Balance Line evidence where the scenario requires them. TC2 EXIT must guarantee Reverse/WM1/trigger/stop context, not just WM2/WM3 additions. COURSE WAVE should guarantee degree/nested-wave and W4/W5 context. A lexical top-3-only policy is insufficient.
3. Make case coverage diverse: every book must include entry, exit, wave and no-setup/insufficient-data variants where supported by the text; separate case families instead of one fixed question per edition.
4. Keep source evidence ordered: observation → exact edition rule → prerequisite → trigger → counter-condition → allowed decision/ABSTAIN. Never equate OHLC crossing with broker fill.
5. Preserve two frozen cohorts: supervised source-learning examples after human source review, and genuinely independent market-chart gold (two externally vetted reviewers). No historical100 or prospective H1 holdout may leak into SFT.
6. Only after those conditions are met, run a new, pinned LoRA version and a new matched holdout exam. Do not retrain the current v5 LoRA on model-authored predictions.

## Conclusion

A genuine blocker is not simply weak weights. The historic benchmark question design and sparse lexical retrieval omit source prerequisites and in one case ask a cross-edition hybrid task. Fixing that protocol should precede any new LoRA training. No model correctness/profitability claim is supported by these artifacts.
