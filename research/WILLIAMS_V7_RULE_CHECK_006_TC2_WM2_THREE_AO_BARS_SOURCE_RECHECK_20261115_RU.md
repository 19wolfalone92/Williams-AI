# TC2 WM2 — three-bar AO momentum add-on, primary-source recheck

**Date:** 2026-11-15  
**Scope:** RULE_CHECK_006. Source review only; not a chart label, expert gold, training example or trading instruction.

## Source identity

- Original Library PDF: `Торговый Хаос 2 (3книга).pdf`
- Verified SHA-256: `055b7924f9a4d165294bf3e12069e74b121f6ca510f25ad310a07c94f0a6107f`
- Physical PDF pages: **135–139**.
- Printed book pages: **120–124**.
- Visual diagrams reviewed: Figures 10.1–10.5, including the three-green-bar add-on illustration and the Canadian-dollar bearish example.

## Source-backed rule sequence

1. WM2 is described as momentum follow-through after the initial WM1 reversal-bar entry. Its purpose in this chapter is to add to an existing position as the market confirms the direction, not to create an unrelated standalone entry.
2. The AO histogram is colored green when its current bar is higher than the preceding AO bar and red when it is lower. These colors describe bar-to-bar momentum direction; they do not mean AO is above/below zero.
3. **Long example:** after a bullish WM1 reversal-bar entry, AO is initially expected to be red. When momentum turns and AO bars become green, the book says this commonly happens within 2–4 bars after entry. It says the position is usually increased after three consecutive green AO bars. The buy stop is placed one tick above the high of the price bar corresponding to the third green AO bar.
4. **Short example:** after a bearish WM1 reversal-bar entry, AO can continue rising for two bars. The sell signal is the third declining/red AO bar; the example says this occurred six bars after the bearish reversal bar and warns not to expect it immediately. Figure 10.5 depicts the add-on sell order one tick below the price bar corresponding to that third declining AO bar.
5. The order trigger is a price level beyond the corresponding price bar. The AO bar identifies the candidate price bar; AO color change alone is not a filled order.

## Negative interpretations explicitly guarded against

- Not three alternating color changes.
- Not three AO bars merely above or below zero.
- No AO zero-line crossing is required by this WM2 three-bar rule.
- Not a fixed three-bar delay after the WM1 entry; the bearish example demonstrates a longer wait.
- Count consecutive rising or falling AO histogram bars, not the number of elapsed candles since the first entry.

## Boundaries and unresolved items

The reviewed pages do not establish a universal maximum waiting period after WM1, a separate position-sizing formula for each add-on, or exchange fill confirmation. Do not infer actual position state, order acknowledgement or fill from an AO signal. Keep WM2 linked to a prior qualified WM1 setup and preserve the distinction between signal, pending stop, price trigger and fill.

This is a source-backed review note, not approval to implement or trade. No numerical model labels were created.

## Review status

- Original PDF SHA verified against the pinned review target: yes.
- Visual diagrams reviewed: yes.
- Independent experts: 0; reviewer signatures: 0.
- Approved for SFT/training: no.
- Approved for trading: no.
- Status: `SOURCE_RECHECKED_EXPERT_REVIEW_PENDING`.

Machine-readable record: `research/williams_v7_review_targets_20261115.json`, RULE_CHECK_006.
