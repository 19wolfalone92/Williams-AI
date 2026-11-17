# TC1 RULE_CHECK_001 — fractal start/signal, candidate confirmation and leverage cancellation

**Date:** 2026-11-15  
**Scope:** Primary-source recheck of RULE_CHECK_001 only. No model labels, expert gold, training, trading or Williams Bot integration.

## Source identity

- Original Library PDF: `1. Торговый Хаос - 1(1).pdf`
- SHA-256 recomputed from the original PDF bytes: `36f6822811088a06254009181480aa1c27af705a76960ecf0342023df803331c`
- The recomputed SHA-256 matches the pinned source hash in the v7 review backlog.
- PDF length: 140 pages; file size: 1,667,993 bytes.
- Physical PDF pages reviewed: **72–74**; printed pages: **72–74**.
- Figures 8.4–8.10 were inspected visually, including the paired fractal start/signal, leverage and three-bar setup illustrations.

## Source-backed structure

1. **A paired structure is required.** The text describes a fractal formation consisting of two adjacent, oppositely directed fractals. This pair establishes the **Fractal Start** and the **Fractal Signal**. The book's own terms on these pages are “start” and “signal”; an implementation-specific label such as `origin_fractal` must not be attributed to the source without a separate mapping.
2. **The signal direction matters.** When the fractal signal is initiated, the trader acts in the direction indicated by that signal. The source distinguishes the signal from the start fractal; they are not interchangeable.
3. **A three-bar shape is only a possible-fractal setup.** In the “inside the fractal” explanation, the hand/thumb arrangement is described as a setup for a possible fractal. A full fractal requires two subsequent bars with the required lower highs (for an upward fractal) or higher lows (for a downward fractal). Therefore, the central/three-bar candidate is not confirmed at the time it first appears.
4. **Leverage decays as price returns toward the start.** The book explains that as price moves back toward the Fractal Start, the available leverage is reduced.
5. **Price invalidation cancels the signal.** If price returns all the way to the relevant extreme of the start bar and exceeds it by one minimum price movement (one tick), the leverage is fully lost and the buy/sell signals are immediately cancelled.
6. **The fractal stop is a separate concept.** The text describes it as the most distant of the last two oppositely directed fractals; it is often, but not invariably, the second-to-last opposite fractal.
7. **Stop versus stop-and-reverse.** A stop based on the opposite fractal is distinguished from a new opposite-direction buy/sell signal, which the text describes as a stop-and-reverse case. The two must not be collapsed into one generic “opposite fractal appeared” event.

## Causal state separation

A faithful research representation should keep these states distinct:

- `FRACTAL_CANDIDATE`: the initial possible-fractal setup is visible;
- `FRACTAL_CONFIRMED`: the two required subsequent bars have completed and the formation is confirmed;
- `FRACTAL_START`: the start element of the paired structure is identified;
- `FRACTAL_SIGNAL_PENDING`: the opposite-direction signal fractal is identified and awaiting initiation;
- `FRACTAL_SIGNAL_INITIATED`: price has initiated the signal in its indicated direction;
- `LEVERAGE_REDUCING`: price has returned toward the start but has not crossed the cancellation boundary;
- `LEVERAGE_CANCELLED`: price has crossed the start-bar extreme by the source's one-tick invalidation amount;
- `FRACTAL_STOP` versus `STOP_AND_REVERSE_SIGNAL`: distinct position-management meanings;
- broker order acknowledgement and fill remain external execution facts, not chart-signal states.

Do not backdate a confirmed fractal to its central bar: the two subsequent bars are required evidence. A candidate that has not yet received those bars must remain unconfirmed.

## Directional arithmetic and unresolved implementation detail

The prose describes the relevant top/bottom extreme of the start bar plus one minimum price movement, but the reviewed passage does not provide a complete signed arithmetic table for both long and short cases. Any implementation must define the adverse-side boundary consistently with the paired direction and instrument tick size, and validate that mapping separately. Do not hard-code the same signed “+1 tick” operation for both directions merely because the prose uses a generic parenthetical.

The source pages also do not define exchange order acknowledgement, gap-through fills, slippage, partial fills, or a broker-side stop replacement protocol.

## Interpretation guards

- Do not treat the initial three-bar candidate as a confirmed fractal.
- Do not use future bars before they have closed to confirm the candidate.
- Do not merge Fractal Start and Fractal Signal into one pivot.
- Do not treat a return toward the start as immediate cancellation; the source specifies loss of all leverage only after the start extreme is exceeded by one tick.
- Do not conflate a fractal stop with an opposite-direction stop-and-reverse signal.
- Do not infer an executed order from a fractal signal.

## Review status

- Original PDF SHA recomputed and matched to pinned target: **yes**.
- Physical pages 72–74 and Figures 8.4–8.10: **reviewed**.
- Independent expert reviewers/signatures: **0**.
- Verified chart gold: **none**.
- Approved for SFT/training: **no**.
- Approved for trading: **no**.
- Status: `SOURCE_RECHECKED_EXPERT_REVIEW_PENDING`.

This is a primary-source clarification, not an expert gold label, a full TC1 audit or trading authorization.
