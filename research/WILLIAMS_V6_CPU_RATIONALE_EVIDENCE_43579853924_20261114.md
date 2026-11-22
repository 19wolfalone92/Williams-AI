# WILLIAMS AI v6 — REAL CPU rationale evidence and SOL contradiction

**Evidence acquired and checked: 2026-11-14.**

## Actual successful run, NOT simulated

- [GitHub Actions run 43579853924](https://github.com/19wolfalone92/Williams-AI/actions/runs/43579853924) **SUCCESS**. Its job `real-public-data-no-model` and the `actual-four-book-retrospective-comparison` job completed. Artifact ID **14110795509**, `Williams-v6-ACTUAL-Qwen-4book-RETROSPECTIVE-CPU-NO-EXPERT-GOLD`.
- Actual downloaded artifact ZIP size: **3,152 bytes**, SHA256 `06b3b8df5814bbba25ead0110098f2064644c226b56289c2a2268c71f07c00ce`. Original JSON **11,203 bytes**, SHA256 `2b4e3cfbc80149f612d907de53baa2e6703b5cfe93b292daef2e48499e0f8c60`.
- Protocol SHA256 `4bc3b68dcda50a8207de8852367385582c6a9cce4db46663ad936eb36963421f`; Qwen3-4B GGUF SHA256 `f6f851777709861056efcdad3af01da38b31223a3ba26e61a4f8bf3a2195813a`; original v5 adapter `8bd26590df627c7452d56e81df6a4572885bfc82515259a7c9de4a3970c850b8`; converted LoRA GGUF `649fb33358e2db9cca8b6c210c6234d34bd56e99f1ce3d3a4ab7e8e3a2d0f46e`.
- **4 real historical cases × 2 actual models**. For every pair, `chart_sha256`, `prompt_sha256`, edition and RAG IDs match; all **8/8 `finish_reason=stop`**. **No expert labels, no score, no forward trading qualification.**
- This report reflects generated model output and deterministic consistency checks; **the model's explanation is not evidence of its own correctness**.

## Observed decisions and evidence quality

| Book/symbol | Base decision | LoRA decision | Base rationale (verbatim or abbreviated) | LoRA rationale (characteristics) | Factually established |
|---|---|---|---|---|---|
| TC1 BTC ENTRY | ABSTAIN | ABSTAIN | 'Insufficient data to confirm a stop entry...' (86 chars) | **190 chars**, invokes `TC1.W5`, unverified lower-degree wave structure | Neither version cites the actual 5-bar fractal extreme, activation tick, or timestamp. Neither justified entry. |
| NTD ETH ENTRY | ABSTAIN | ABSTAIN | Exactly `INSUFFICIENT_DATA` (**17 chars**) | **122 chars**, says AO/AC and zone confirmations unresolved | No verifiable AC histogram sequence, zero sign, colour, candle/price level in either explanation. |
| TC2 SOL EXIT | **HOLD** | INSUFFICIENT_DATA | **Exactly `INSUFFICIENT_DATA` (17 chars)** | **103 chars**, requests 'additional price and volume data' | **Confirmed contradiction**: Base chooses HOLD but simultaneously explains that data are insufficient. LoRA is more cautious but gives a generic reason despite closed OHLCV already being in prompt; neither demonstrates the trigger or position state. |
| COURSE XRP WAVE | WAVE_HYPOTHESIS | WAVE_HYPOTHESIS | **240 chars**, ends mid-thought '...the last lower wave is' | **240 chars**, ends mid-thought '...может относиться к W5 внутри' | Both are at the JSON reason length ceiling with incomplete text; they paraphrase source-card wave-four theory but do not identify XRP's actual bar sequence or independently verified wave degree. |

## Root-cause forensic conclusions

1. **BASE SOL logical contradiction, confirmed at the output layer:** the `reason` and decision category disagree. No expert assessment is required to establish this *internal inconsistency*. It does not prove which external market action would be correct. In the non-scoring diagnostic, this now emits `FAIL_DECISION_REASON_CONTRADICTION`; the frozen inference protocol is kept unchanged.
2. **Potential decoder/explanation length ceiling:** both XRP explanations are exactly 240 chars, end as incomplete sentences, although `finish_reason=stop`. This is strongly consistent with the `reason.maxLength=240` grammar constraint. Preserve evidence as **POTENTIAL_LENGTH_CEILING_CLIP** rather than claiming an established backend bug without token-level traces. The diagnostic now raises `at_reason_length_ceiling_240` warning.
3. **Generic/extracted rather than market-specific reasoning:** BTC LoRA mainly cites TC1.W5 despite asking about entry/fractional price level. COURSE outputs closely mirror the RAG cards without bar-level evidence. NTD Base only repeats an uncertainty label. These are **observed explanation weaknesses**, not proof LoRA is better/worse on independent cases.
4. **Why SOL changed:** identical prompt SHA `87296dd5f6a20e5da36dd090fc0a9b928d9e39b89169c49139bebd3413b7fed8`, matched `TC2.WM2 / TC2.CAMPAIGN / TC2.WM3` and chart SHA `0517472962801b360420833dd5f70fa47ad966234d205527cf2c86dfc0c67ce7`. Since real position entry/fills/stops are explicitly missing, **neither answer is a verified trade instruction**. The observed difference comes from model outputs, but whether training caused improved book understanding remains **UNDETERMINED**.
5. **What was actually corrected:** a deterministic post-hoc contradiction/length detector and three additional regression tests. The earlier frozen inference experiment and model checkpoints remain untouched. Source-card corrections remain pending independent review.

## Decision

Keep trained v5 solely as a research candidate. **NO automatic GPU retrain; NO live trading; NO expert-accuracy estimate.** Require independent page-and-market review (two vetted human experts), closed-candle OHLCV evidence, and post-hoc causal signal validation. 4 historical cases do not establish statistical superiority. The separate prospective H1 plans remain unchanged and cannot be completed before their future cutoffs.

