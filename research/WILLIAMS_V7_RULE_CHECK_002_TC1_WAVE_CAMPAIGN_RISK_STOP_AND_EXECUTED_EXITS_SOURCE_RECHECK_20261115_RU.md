# TC1 RULE_CHECK_002 — wave hypothesis, campaign risk stops and realized exits

**Date:** 2026-11-15  
**Scope:** Primary-source recheck of RULE_CHECK_002 only. No model labels, expert gold, training, trading or Williams Bot integration.

## Source identity

- Original Library PDF: `1. Торговый Хаос - 1(1).pdf`
- SHA-256 recomputed from the original PDF bytes: `36f6822811088a06254009181480aa1c27af705a76960ecf0342023df803331c`
- The recomputed SHA-256 matches the pinned source hash in the v7 review backlog.
- PDF length: 140 pages; file size: 1,667,993 bytes.
- Physical PDF pages reviewed: **89–93**; printed pages: **89–93**.
- Figures 9.8–9.11 were inspected visually.

## Source-backed findings

1. **The Zero Point is a hypothesis, not certainty.** In the Swiss-franc example, the author describes identifying a candidate Zero Point using a five-wave count on a smaller timeframe, together with a downward fractal and a Squat day among the three lowest bars. The author explicitly says that at the initial stage it is only an assumption and that one cannot know with certainty whether the point was correctly identified.
2. **A contingency plan is part of the method.** The example describes a plan for what to do if the initial Zero Point calculation proves wrong. The wave count is therefore a working market hypothesis that must remain revisable, not an immutable label.
3. **A risk stop is distinct from the wave estimate.** In the Swiss-franc example, an initial retracement entry is paired with a stop-and-reverse level at the proposed Zero Point. The text gives specific prices and contract quantities for that campaign; these are historical example values, not universal risk percentages or position-sizing rules.
4. **Fractals and Squat participate in partial campaign decisions.** In the example, a downward fractal together with a Squat is used to reduce two of three long contracts while retaining one in case the first wave extends. Later, a larger campaign is partially reduced before the apparent top; the remaining contracts are kept under a separate campaign/risk plan.
5. **The source distinguishes planned stops from reported executions.** Some levels are described as contingent stop-and-reverse or protective stops. Other actions are explicitly reported as executed purchases, sales, partial covers or stop-outs. A planned level is not itself proof that an order filled.
6. **Campaign exposure changes over time.** The Swiss-franc and soybean examples show partial exits, retained residual positions, subsequent additions on retracements, protective stops and final exits at a projected wave-5 area. Position state is not represented by a single wave label.
7. **The examples are not a universal template.** The chapter's conclusion explicitly cautions that the two campaign examples are illustrative and should not be treated as a pattern that generates every trade.

## Required state separation for research representation

Keep these distinct:

- `WAVE_COUNT_HYPOTHESIS` and its alternative counts;
- `ZERO_POINT_CANDIDATE` and its supporting observations;
- `RISK_STOP_PLANNED` / `STOP_AND_REVERSE_PLANNED`;
- `ENTRY_OR_ADD_TRIGGERED`;
- `ORDER_SUBMITTED`, `ORDER_ACKNOWLEDGED` and `FILL_CONFIRMED`;
- `PARTIAL_EXIT_PLANNED` versus `PARTIAL_EXIT_FILLED`;
- `RESIDUAL_POSITION` and its current protective stop;
- `CAMPAIGN_EXIT_HYPOTHESIS` versus `CAMPAIGN_EXIT_FILLED`.

A fractal/Squat observation may support a specific partial campaign action in the example, but it must not be generalized into “any opposite fractal closes the whole position.” The example itself retains residual exposure and manages it separately.

## Interpretation guards

- Do not promote the first Zero Point estimate to a certain wave origin.
- Preserve an explicit alternative/contingency plan when the initial count is wrong.
- Do not conflate a stop level being planned with the stop being triggered or filled.
- Do not infer a complete exit from a partial reduction.
- Do not infer a position size or risk percentage from the historical contract counts.
- Do not treat the historical campaign sequence as a universal mechanical template.
- Do not collapse wave hypothesis, stop, order, fill and position state into one signal.

## What these pages do not establish

- A deterministic, universally correct Elliott-wave counting method.
- A universal risk percentage, leverage, contract-sizing or maximum-campaign-risk formula.
- A generic rule that every opposite fractal immediately closes the entire position.
- Broker acknowledgement, partial-fill, slippage, gap-through or order-replacement mechanics.
- A guarantee that the projected wave-5 target or Zero Point will be reached.

## Review status

- Original PDF SHA recomputed and matched to pinned target: **yes**.
- Physical pages 89–93 and Figures 9.8–9.11: **reviewed**.
- Independent expert reviewers/signatures: **0**.
- Verified chart gold: **none**.
- Approved for SFT/training: **no**.
- Approved for trading: **no**.
- Status: `SOURCE_RECHECKED_EXPERT_REVIEW_PENDING`.

This is a primary-source clarification, not an expert gold label, a full TC1 audit or trading authorization.
