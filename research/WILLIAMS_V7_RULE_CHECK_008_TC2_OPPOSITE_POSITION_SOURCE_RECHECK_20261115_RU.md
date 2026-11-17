# Williams AI v7 — RULE_CHECK_008: source recheck of opposite signal with an open position

**Date:** 2026-11-15  
**Scope:** source-only review of Trading Chaos, 2nd Edition (2004). No model labels, no SFT approval, no trading implementation.

## Source identity

- Primary source: `Bill Williams - Trading Chaos 2nd Ed.pdf`
- SHA-256 pinned in the review backlog: `055b7924f9a4d165294bf3e12069e74b121f6ca510f25ad310a07c94f0a6107f`
- Pages checked: printed pp. 131, 149–151; p. 150 figure inspected visually.
- The previous target also cited p. 164. That page is in the creative-process / morning-pages discussion and does **not** establish an opposite trading signal or position-management rule. It is excluded from the evidence for RULE_CHECK_008.

## Source findings

1. **Printed p. 131 — protecting an existing long.** In the Second Wise Man discussion, the authors distinguish a protective/trailing stop from a stop-and-reverse strategy. They describe a qualified Fractal sell signal as a reason to reverse a long position when the relevant sell trigger is hit; the accompanying example also describes taking the short side after a bearish divergent-bar signal. This is conditional position management, not an unconditional close on signal appearance.

2. **Printed p. 149 — short-side setup and reversal condition.** The short-side procedure is presented as the mirror of the long-side procedure. It identifies the bearish divergent-bar setup, its Alligator angulation condition, the sell-stop trigger, and a protective stop. The following buy-Fractal condition is tied to the protective-stop level and a price trigger.

3. **Printed pp. 150–151 — chart and campaign sequence.** Figure 12.2 was inspected visually. The text describes adding to a short campaign only on subsequent qualifying sell signals, protecting the position with a trailing stop, and then being reversed to the long side by a bullish divergent bar. The figure and accompanying text therefore support a **triggered stop/reverse state transition**, not immediate liquidation merely because an opposite signal is detected.

## Safe interpretation boundary

For a source-grounded research representation, keep these events distinct:

- opposite setup detected;
- setup qualifies under its own book-specific filters;
- stop/reverse trigger is placed or becomes pending;
- price reaches the trigger;
- position is flattened and the reverse exposure is established according to the specified size;
- broker execution is independently confirmed.

Do not collapse these into one Boolean signal or infer a fill from a chart signal. The source examples describe a reverse order sized to close the existing exposure and leave a net position in the opposite direction; actual sizing and execution still require separate operational rules. This note does not authorize code integration, live trading, or a change to Williams Bot.

## Review gate

- Source spot-check: complete for the pages listed above.
- Page 164: rejected as irrelevant evidence for this target.
- Independent expert reviewers required: 2.
- Reviewer signatures received: 0.
- Verified chart gold: none.
- `approved_for_SFT`: false.
- `training_allowed`: false.

This is a primary-source spot-check, not a full-book audit, not an independent expert consensus, and not a trading qualification.
