# TC2 WM3 — Fractal trigger and Teeth filter at activation

**Date:** 2026-11-15  
**Scope:** RULE_CHECK_007, primary-source recheck only. No model inference, chart labels, expert gold, training, trading or Williams Bot integration.

## Source identity and integrity

- Original Library PDF: `Торговый Хаос 2 (3книга).pdf`
- SHA-256 computed from the materialized original PDF bytes: `055b7924f9a4d165294bf3e12069e74b121f6ca510f25ad310a07c94f0a6107f`
- The computed SHA-256 exactly matches the source hash pinned in `research/williams_v7_review_targets_20261115.json`.
- PDF length: 224 pages; file size: 6,362,767 bytes.
- Physical PDF pages reviewed: **147–149**; printed book pages: **132–134**.
- Visual diagrams inspected: Figures 11.3, 11.4, 11.5 and 11.6, including the valid/invalid activation examples.

## Source-backed sequence

1. **Formation is not activation.** A fractal pattern can be identified before its breakout level is crossed. The completed pattern creates a candidate fractal signal; it does not, by itself, mean the price trigger has fired.
2. **Price-stop level.** For a buy fractal, the book places a buy stop one tick above the high of the upward fractal. For a sell fractal, it places a sell stop one tick below the low of the downward fractal.
3. **Teeth filter is evaluated at activation.** A buy fractal is used only if price is above the red balance line (Alligator Teeth) when the signal fires. A sell fractal is used only if price is below that line when it fires.
4. **Formation location is not the deciding test.** The text explicitly distinguishes where a fractal signal forms from where it fires: the latter determines whether the signal is valid relative to the red line.
5. **Pending-signal lifetime.** Once a fractal signal has formed and is recognized as valid relative to the Alligator, it remains a signal until it fires or a new fractal signal forms. The source therefore does not support treating every newly formed fractal as an immediate trade.
6. **Direction of breakout.** The source's examples use the breakout in the direction of the fractal: above the upward-fractal high for a buy, below the downward-fractal low for a sell.

## Required state separation for any future research representation

Keep these as separate events/states:

- `FRACTAL_FORMED`: the required price pattern is confirmed using only bars available at that time;
- `FRACTAL_PENDING`: the corresponding one-tick breakout level is waiting;
- `PRICE_TRIGGER_TOUCHED`: price reaches/crosses that level;
- `TEETH_FILTER_PASS_AT_ACTIVATION` or `TEETH_FILTER_FAIL_AT_ACTIVATION`: compare price with the red line at the trigger time, not merely at formation;
- `SIGNAL_QUALIFIED`: the breakout and Teeth condition both pass;
- `BROKER_ORDER/FILL`: not established by these book pages and must never be inferred from the chart signal.

A buy that forms above Teeth but fires below Teeth is not a valid buy under the described filter. A sell that forms below Teeth but fires above Teeth is not a valid sell. The converse applies when the trigger occurs on the correct side of Teeth.

## Diagram and negative-example interpretation

- Figure 11.4 distinguishes a fractal's location at formation from the red-line condition at firing.
- Figure 11.5 shows an eligible fractal breakout and a later fractal that forms inside the Alligator and is not to be used if it fires there.
- Figure 11.6 contrasts valid and invalid fractal signals and reinforces that the trigger location, not the formation location alone, determines validity.

## Engineering limits not specified by the source

The reviewed pages do **not** specify:

- a numeric tolerance or buffer around Teeth;
- a universal tie-breaking policy for multiple overlapping pending fractals beyond the statement that a new fractal signal ends the prior signal's waiting period;
- how to reconstruct the exact intrabar order of a stop touch and a moving Teeth line from OHLC candles alone;
- broker gap-through, partial-fill, slippage, tick-rounding or order-acknowledgement behavior.

Consequently, if only a closed H1 OHLC candle is available and the intrabar sequence cannot establish which side of Teeth price occupied at the instant the stop was crossed, the research result must remain unresolved/ABSTAIN. Do not substitute the candle close or the Teeth value at fractal formation for the source's activation-time test. Lower-timeframe or event-level data may be required for a faithful reconstruction. This is an engineering limitation inferred from the source's timing requirement, not a new rule attributed to Williams.

## Negative interpretations guarded against

- Fractal formation above/below Teeth is not enough to qualify the later breakout.
- Do not evaluate the Teeth filter only when the five-bar fractal becomes confirmed.
- Do not treat fractal confirmation as a filled entry.
- Do not discard a still-pending signal merely because price has not triggered it; the source gives its lifetime as until trigger or a new fractal signal.
- Do not invent an intrabar ordering, numeric buffer, or exchange fill from a single OHLC candle.

## Review status

- Original source SHA recomputed from Library PDF bytes: **verified**.
- Pinned SHA match: **yes**.
- Physical pages 147–149 / printed pages 132–134: **reviewed**.
- Visual diagrams 11.3–11.6: **inspected**.
- Independent expert reviewers: **0**; reviewer signatures: **0**.
- Verified chart gold: **none**.
- Approved for SFT/training: **no**.
- Approved for trading: **no**.
- Status: `SOURCE_RECHECKED_EXPERT_REVIEW_PENDING`.

This is a primary-source research note, not a complete book audit, expert consensus, training label, decoder qualification or trading authorization.
