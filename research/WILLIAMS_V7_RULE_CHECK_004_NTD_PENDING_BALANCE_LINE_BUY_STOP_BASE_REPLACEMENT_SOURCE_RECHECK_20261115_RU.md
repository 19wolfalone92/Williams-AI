# NTD RULE_CHECK_004 — pending Balance Line buy stop and base replacement

**Date:** 2026-11-15  
**Scope:** Primary-source recheck of RULE_CHECK_004 only. No model labels, expert gold, training, trading or Williams Bot integration.

## Source identity

- Original Library PDF: `Билл_Вильямс_Новые_измерения_в_биржевой_торговле_2_книга.pdf`
- SHA-256 recomputed from the original PDF bytes: `855eb052353c852f208436952b12e46d9a18badf7c1dac7ade7580d57b9f4bb4`
- The recomputed SHA-256 matches the pinned source hash in the v7 review backlog.
- PDF pages reviewed: **103–105**. Figure 8.9 and the base/signal illustrations in Figures 8.10–8.11 were inspected visually.

## Source-backed pending-stop behavior

1. After the Balance Line base bar has been established and the buy stop is placed, the source says that subsequent bars with higher highs do not, by themselves, move the pending buy stop.
2. The buy stop remains at its existing level unless either:
   - the stop is triggered; or
   - another bar with a higher high is formed and becomes a new base under the Balance Line base-selection procedure.
3. The source explicitly points out that the last three bars in Figure 8.10 and the last two bars in Figure 8.11 do not move the Balance Line buy signal. This is a visual guard against moving the stop on every new higher high.
4. The new-base condition must be evaluated using the Balance Line rules: scan right-to-left, identify the valid base, then count the required one/two new higher highs according to whether price is moving away from or toward the Balance Line. A raw higher high is not automatically a new base.
5. The paragraph and diagrams reviewed here describe the buy-side pending stop. Do not claim a fully specified sell-side stop-replacement algorithm from these pages alone.

## Required pending-state separation

Keep these distinct:

- `BASE_CONFIRMED`;
- `BUY_STOP_PENDING` at the currently established signal level;
- `HIGHER_HIGH_OBSERVED` that does not qualify as a new base (pending stop unchanged);
- `NEW_BASE_CONFIRMED` under the Balance Line base-selection rule (re-evaluate/re-place the pending signal as specified);
- `BUY_STOP_TRIGGERED`;
- `ORDER_ACKNOWLEDGED` and `FILL_CONFIRMED` as separate broker facts.

## Interpretation guards

- Do not slide a pending buy stop upward for every higher high after base confirmation.
- Do not treat any higher-high candle as a new base without reapplying the base-selection rule.
- Do not move the stop before a new base is actually established.
- Do not keep the old pending stop after it has triggered as though it were still pending.
- Do not infer a fill from the chart trigger.
- Do not extrapolate the buy-side diagrams into a complete sell-side implementation without further source review.

## What these pages do not establish

- A complete sell-side pending-stop replacement sequence.
- Broker order modification/acknowledgement, gap-through, partial-fill or slippage behavior.
- A universal signed tick-offset rule for every instrument.
- A numerical tolerance for deciding whether a new bar qualifies as a new base.

## Review status

- Original PDF SHA recomputed and matched to pinned target: **yes**.
- Pages 103–105 and Figures 8.9–8.11 visually inspected: **yes**.
- Independent expert reviewers/signatures: **0**.
- Verified chart gold: **none**.
- Approved for SFT/training: **no**.
- Approved for trading: **no**.
- Status: `SOURCE_RECHECKED_EXPERT_REVIEW_PENDING`.

This is a primary-source clarification, not an expert gold label or trading authorization.
