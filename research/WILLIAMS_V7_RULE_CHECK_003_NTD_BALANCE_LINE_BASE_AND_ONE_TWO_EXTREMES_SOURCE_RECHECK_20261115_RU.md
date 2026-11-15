# NTD RULE_CHECK_003 — Balance Line base and one/two new extremes

**Date:** 2026-11-15  
**Scope:** Primary-source recheck of RULE_CHECK_003 only. No model labels, expert gold, training, trading or Williams Bot integration.

## Source identity

- Original Library PDF: `Билл_Вильямс_Новые_измерения_в_биржевой_торговле_2_книга.pdf`
- SHA-256 recomputed from the original PDF bytes: `855eb052353c852f208436952b12e46d9a18badf7c1dac7ade7580d57b9f4bb4`
- The recomputed SHA-256 matches the pinned source hash in the v7 review backlog.
- PDF length: 156 pages; file size: 4,383,031 bytes.
- PDF pages reviewed: **99, 103 and 104**; the buy-side diagrams on pages 103–104 were inspected visually.

## Source-backed Balance Line procedure

1. **Read from right to left.** The source instructs the reader to begin with the most recent bars and work back toward older bars when finding the base.
2. **Use the relevant price extreme only.** For a buy search, examine highs; for a sell search, examine lows.
3. **Identify the base before counting the signal.** For a buy, the base is the current bar or the bar with the lowest high. For a sell, it is the current bar or the bar with the highest low.
4. **Count new extremes relative to the base and Balance Line direction.**
   - Buy: one new higher high when price is moving away from the Balance Line; two new higher highs when moving toward it.
   - Sell: one new lower low when moving away from the Balance Line; two new lower lows when moving toward it.
5. **The one-versus-two rule is directional, not a generic bar count.** It counts new highs for buys or new lows for sells, not elapsed bars or any two candles that happen to close beyond a line.
6. **Buy-side special case above the Balance Line.** The text says that when the signal bar is above the Balance Line, the formation is treated as if the entire formation were above it; the example uses the base plus one higher high.
7. **A pullback is needed before a new buy setup can form after fresh highs.** The source advises that while the market is making new highs there is no Balance Line or Fractal buy signal; price must pull back and create lower highs before those signals can form.

## Context from the chapter opening

The Balance Line is described as the hypothetical price location in the absence of new incoming information. The distance between current price and that line represents the effect of incoming information. The source's directional analogy is that moving away from the Balance Line is easier than moving back toward it; this explains why the required count differs.

## Interpretation guards

- Do not identify the base after counting the new highs/lows; base selection comes first.
- Do not scan left-to-right as though the oldest bar were the active starting point.
- Do not count highs for a sell setup or lows for a buy setup.
- Do not use one new high/low when price is moving toward the Balance Line; the source calls for two.
- Do not treat “one or two” as the number of bars in the formation; it is the number of new directional extremes.
- Do not form a new buy signal while price is simply continuing to make new highs without the required pullback/lower-high structure.
- The specific special-case illustration reviewed here is the buy side above the Balance Line; do not invent an unreviewed mirrored exception.

## What these pages do not establish

- A complete numerical formula for the Balance Line itself on these pages.
- A universal tolerance for equal highs/lows, gaps or tick rounding in the base comparison.
- Broker order acknowledgement, fill, slippage or stop replacement mechanics.
- A separate special-case exception for every possible sell-side geometry beyond the general one/two lower-low rule.

## Review status

- Original PDF SHA recomputed and matched to pinned target: **yes**.
- Pages 99, 103 and 104 reviewed; Figures 8.7–8.9 visually inspected: **yes**.
- Independent expert reviewers/signatures: **0**.
- Verified chart gold: **none**.
- Approved for SFT/training: **no**.
- Approved for trading: **no**.
- Status: `SOURCE_RECHECKED_EXPERT_REVIEW_PENDING`.

This is a primary-source clarification, not an expert gold label or trading authorization.
