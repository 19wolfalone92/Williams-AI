# TC2 WM1 angulation — primary-source recheck (research only)

**Date:** 2026-11-15  
**Scope:** RULE_CHECK_005, source review only. No model inference, labels, training, trading or bot integration.

## Source identity and page map

- Original Library PDF: `Торговый Хаос 2 (3книга).pdf`
- SHA-256 of the original PDF bytes, computed after materializing the Library source: `055b7924f9a4d165294bf3e12069e74b121f6ca510f25ad310a07c94f0a6107f`
- This exactly matches the pinned `source_pdf_sha256` in `research/williams_v7_review_targets_20261115.json`.
- Physical PDF pages reviewed: **121–124**; printed book pages: **106–109**.
- Visual diagrams reviewed: Figures 9.1–9.5, including the UOPX invalid-signal example on physical PDF page 124.

## What the source actually specifies

1. **Origin:** begin where price bars cross or pass through the Alligator mouth (Jaw, Teeth and Lips).
2. **Alligator path:** trace or imagine the path of the mouth. The text says Jaw and Teeth receive more attention than Lips; it does not say to use Jaw alone.
3. **Price path:** trace the outer edges of the price bars — lower edges for an upward move and upper edges for a downward move.
4. **Geometric condition:** the two paths must clearly move away from one another. The opening angle must be increasing.
5. **Signal validity:** a bullish/bearish divergent-bar signal is not taken without that increasing angulation.
6. **Negative example:** the UOPX illustration shows a divergent-bar-shaped candidate that is invalid when price and Alligator paths remain at a similar/parallel angle; the text also notes penetration into the Alligator as an invalidating condition.

The book's explanatory analogy is a stretched rubber band between current price and the Blue Line/Jaw. That analogy does not replace the four operational drawing rules above.

## What the source does NOT specify

The reviewed pages provide no numerical slope threshold, fixed lookback length, regression formula, angle-in-degrees threshold, or universal distance-from-Jaw test. Those must not be invented and attributed to the book.

## Research / implementation boundary

A future candidate formalization must preserve the crossing-origin, Alligator-mouth path, directional price-edge path and increasing-divergence conditions. A current price-to-Jaw distance by itself is insufficient. Any numeric estimator would be a new engineering operationalization requiring separate validation, not a verbatim Williams rule. If the geometry cannot be established from available closed bars, the safe research result is unresolved/ABSTAIN rather than a fabricated angulation pass.

This recheck is **not** an expert gold label, does not decide any historical chart, and does not authorize changing the decoder, trading engine or Williams Bot.

## Review status

- Original source SHA verified against the Library PDF: yes.
- Visual source diagrams reviewed: yes.
- Independent expert reviewers: 0.
- Reviewer signatures: 0.
- Approved for SFT/training: no.
- Approved for trading: no.
- Status: `SOURCE_RECHECKED_EXPERT_REVIEW_PENDING`.

The machine-readable record is in `research/williams_v7_review_targets_20261115.json`, RULE_CHECK_005.
