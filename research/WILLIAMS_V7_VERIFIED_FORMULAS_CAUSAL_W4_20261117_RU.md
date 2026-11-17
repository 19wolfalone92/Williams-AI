# Williams AI: corrected primary-source formulas + exact causal AO W4 gate

Date: 2026-11-17. Research only. Source PDFs remain private in user's Library and are **not** uploaded to GitHub. No independent expert labels, no weights modified, no order actions.

## Source findings and code audit (verified, not inferred from model)

### AO editorial conflict
- Original English **Trading Chaos 2nd Edition**, SHA-256 `eb734089bb19e72a92e023602bf558e43b7cb20aecf77469fdfe28f8e6cd3523`, physical PDF **155** / printed 126 shows text `(H-L)/2` for a price-bar "midpoint".
- Same original, physical PDF **247** / printed 218, EasyLanguage source code says `PRICE((H+L)/2)`; `AO=AVERAGE(PRICE,5)-AVERAGE(PRICE,34)`.
- Pinned Russian **TC2** (`055b7924f9a4d165294bf3e12069e74b121f6ca510f25ad310a07c94f0a6107f`) physical **136** uses the same problematic text. A label error is an editorial conflict, not a reason to silently switch to (H-L)/2, which computes half range rather than a midpoint.
- **Existing `strategy.calculate_indicators` already uses the appropriate price midpoint `(H+L)/2` and AO SMA(5)-SMA(34).** No indicator algorithm change justified. Added regression tests to detect half-range substitution, not merely a hard-coded text assertion.
- The source disagreement remains `OPEN_NOT_GOLD`, pending an external expert's independent confirmation. Source-backed engineering does not produce market chart gold.

### NTD Lips discrepancy
- Original NTD SHA-256 `855eb052353c852f208436952b12e46d9a18badf7c1dac7ade7580d57b9f4bb4`, physical PDF **36** boxed formula states Lips shift +2, adjacent narrative and repeated formula use +3.
- `strategy.calculate_indicators` already applies 5-period SMMA with **+3 displacement**, and values are time-aligned to the current visible candle using `.shift(3)`. Added actual numerical test disproving +2 and cutoff-prefix invariance. Original-source discrepancy is still explicitly tracked for external English-source adjudication.

### W4 / W5 causal certainty
- TC1 Russian, original SHA `36f6822811088a06254009181480aa1c27af705a76960ecf0342023df803331c`, physical **53–54**: 4th-wave structural context, no automatic fifth-wave prediction; wave degree is provisional.
- Profitunity Course original SHA `fc57f707378087bfc5146db5e242c51ce8135e0b359f0c607305b364e0decfca`, physical **43–44**: **moving past zero** provides minimum W4 evidence and is NOT proof that W4 is over.
- `williams_ai.wave_evidence.alternating_impulse_skeleton` now distinguishes pre-existing **AO near zero** from **strict direction-specific zero crossing** inside the candidate W4 on fully observed bars. Missing observations return `None`. `w4_completion_proven=False` irrespective of crossing. No automatic fifth-wave certification or execution.
- Tests cover touch (does not pass), genuine crossing (observation only), missing bar (unknown); no lookahead.

### Fail-closed source issue integrity
- Previous conflict validator accepted any 64-character hex as a source SHA. Now exact **four pinned original SHA256 hashes plus the separate original English TC2 PDF** are required, and pages must lie within that actual source PDF length. New negative tests reject correctly formed but fabricated hash values.

## SFT and evaluation gate
Human expert gold **0**, model quality accuracy **UNKNOWN**, no additional LoRA launched. This is a mathematical and bibliographic correction, not independent testing of Qwen's answers. The 15 pairs / 30 text variants remain **only source-only unlabelled probes**, not real H1 cases or a blinded holdout. Subsequent paired Base/v5 inference requires actual model outputs and qualified independent book reviewers to grade meaningful correctness. Keep `main`, frozen prospective H1 cohorts, and Williams Bot untouched.
