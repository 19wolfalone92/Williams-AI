# WILLIAMS AI v6 — independent original-PDF SHA and VISUAL page recheck (2026-11-14)

**Status: REAL four-file byte verification + direct figure inspection COMPLETE for the listed pages, not a full page-by-page audit of all 642 pages.** Original books were located in the user's private connected file library and materialized locally for a *read-only* SHA256/PDF-page check. **No PDF, scan, copyrighted page image or OCR corpus is committed to GitHub.** The SHA and count computations were performed on actual original file bytes; image review was performed on rendered PDF pages, including the scanned COURSE. This audit corrects the earlier statement that source PDFs were unavailable.

## Four books verified against knowledge/SOURCE_RULE_CARDS.json

| Edition | Local source variant actually verified | SHA-256 computed from original bytes | PDF pages | Matches fixed source registry |
|---|---|---|---:|---|
| TC1 | `1. Торговый Хаос - 1.pdf` | `36f6822811088a06254009181480aa1c27af705a76960ecf0342023df803331c` | 140 | YES |
| NTD | `Билл_Вильямс_Новые_измерения_в_биржевой_торговле_2_книга(1).pdf` | `855eb052353c852f208436952b12e46d9a18badf7c1dac7ade7580d57b9f4bb4` | 156 | YES |
| TC2 | `Торговый Хаос 2 (3книга)(1).pdf` | `055b7924f9a4d165294bf3e12069e74b121f6ca510f25ad310a07c94f0a6107f` | 224 | YES |
| COURSE | `Bill Williams - Trading Chaos Course(1).pdf` | `fc57f707378087bfc5146db5e242c51ce8135e0b359f0c607305b364e0decfca` | 122 | YES |

**Important alternate-edition warning:** a different file named `Bill Williams - Trading Chaos 2nd Ed.pdf` in the library has 257 PDF pages and SHA-256 `eb734089bb19e72a92e023602bf558e43b7cb20aecf77469fdfe28f8e6cd3523`; it is **NOT** the 224-page TC2 pinned in this repository. Mixing page numbers or rule extraction from these editions silently corrupts provenance. That file was excluded from source-rule confirmation.

## Verified graphical/book evidence (PDF page numbers, not printed chapter numbers)

| Rule/case | PDF pages **actually** visually reviewed | What the original page and figures support | What must not be inferred |
|---|---|---|---|
| `TC1.FRACTAL` / BTC entry | TC1 pp. **68–70**, especially figure **8-1** and **8-2** | Five-bar *initiating* fractal needs central extreme plus two eligible preceding and two following bars; ties at the same price level must not be counted as a strictly lower high / higher low; the page explicitly allows an extended 6-bar example. A subsequent move through the prior fractal extreme is distinguished from initial formation. | A 5-bar graphic or a generic fractal flag alone does not prove a live stop-entry or Elliott degree. |
| `NTD.AC` / ETH entry | NTD pp. **72–75**, figures **6-1**, **6-2**, **6-5** | AC's zero-line crossing is not its own signal. Buying when **AC above zero** requires two increasing green bars; when **AC below zero** requires three. For each signal, the corresponding price bar's high plus one minimum price movement is the intended buy stop level; the reverse-color bar is needed before a new signal. | Do not use AO sign as the AC sign, or call the mere zero crossing an entry. The specific instrument tick size/fill is not provided. |
| `TC2.CAMPAIGN` / SOL exit | TC2 pp. **153–154**, figure **12.1** | The illustrated gold campaign progresses WM1→WM2→fractal entries, then has a bearish reversal bar at point 10; the **next** price bar moving below that bar triggers the example's short reversal. Merely seeing the bearish reversal bar is not proof that the level was crossed. | Figure 12.1 is a *gold* book example, not SOLUSDT broker/position evidence. Cannot certify HOLD/EXIT/REVERSE for SOL without its closed bars, position assumptions and activation test. |
| `COURSE.W4.ZERO` / XRP wave | COURSE pp. **43–45**, image-only original scanned pages; figures headed wave-4 end and Fibonacci time projections | The text states that 5/34 crossing the **zero line** satisfies **minimum requirements** for wave four; it explicitly does **not** prove wave four has finished. It recommends examining a-b-c or triangle corrective structure and smaller subwaves, using an approximately **100–140 bar** scope for scale, and contextual Fibonacci time intervals. | A screenshot with the caption wave-four-looking-complete is not an independently certified Elliott count; no guarantee of wave-five projection or current XRP completion. |

**Source-index issue:** `TC2.CAMPAIGN` currently lists pages `[153,161,162,164,165]` but omits **154**, which actually contains the explicit condition "following bar below the reversal bar" in chapter 12.1. This is a *confirmed missing page in the existing source-card citation list*, not evidence that SOL exit itself is right/wrong. Keep the frozen v5 registry unchanged until a versioned, independently approved source-card revision; for now reference both PDF 153 and 154 in developer audit, not in the fixed exam prompt.

## Interpretation of the four model categories

- **TC1 BTC**: ABSTAIN / ABSTAIN — cannot establish correctness from category alone; would need finished fractal candidates, triggering level and movement after formation.
- **NTD ETH**: ABSTAIN / ABSTAIN — cannot establish whether the AO/AC signals were correctly recognized without actual archived bar-level recalculation and AC-sign/color trace.
- **TC2 SOL**: HOLD / INSUFFICIENT_DATA — different categories, but absent actual position/stops and no captured prior model reasons; no winner or LoRA superiority can be inferred.
- **COURSE XRP**: WAVE_HYPOTHESIS / WAVE_HYPOTHESIS — provisional label consistent with uncertain degree, but graphical criteria were not verified for that market snapshot.

## Mandatory chain for further claims

`edition source SHA -> PDF page number + verified figure -> versioned rule card -> historical chart SHA + cutoff -> explicit closed-bar / calculated-indicator evidence -> short model reason -> independent reviewer judgment`.

Stage level: `SHA_VERIFIED` for all four original files; `VISUAL_PAGE_SPOT_CHECKED` for the above selected pages; `COMPLETE_BOOK_VISUAL_AUDIT` **NOT DONE**; `INDEPENDENT_EXPERT_MARKET_GOLD` **BLOCKED**; `TRADE_QUALIFIED` **FALSE**.

No external expert identity/signature or gold was fabricated.

## All-page PDF technical integrity pass (non-semantic)

All **642 pages** of the four SHA-verified originals were individually decoded and rendered through PyMuPDF (low-resolution display integrity pass), without replacing the actual files. This checks that an image can be rendered, **not** that all 642 pages were read and validated semantically. Outcomes:

| Book | PDF pages rendered | Pages with little/no extractable text | Pages containing image objects | Rendering errors |
|---|---:|---:|---:|---:|
| TC1 | 140/140 | 0 | 140 | 0 |
| NTD | 156/156 | 2 | 96 | 0 |
| TC2 | 224/224 | 0 | 56 | 0 |
| COURSE | 122/122 | **122** | **122** | 0 |
| **Total** | **642/642** | **124** | — | **0** |

**COURSE is fully image-only in this particular PDF.** A text-only extraction cannot support independent verification of COURSE figures; its high-risk pages 43–45 were visually inspected separately. Automated OCR on all pages is not claimed, and no false recognition/independent page review is asserted. Do not upload these source PDFs or page images to public GitHub.

The all-page integrity pass local JSON SHA256 is `9ab42dd2b4130df53e7a0df7b0d9205a4e483914b8dbf55b475142ff6ae0ce84`. Original PDFs remain private. This pass is valid solely as technical reading/rendering, not as independent two-expert signoff.

## Prospectively frozen H1 cohorts: concrete non-independence warning

Read-only review of the already-frozen pilot `williams_v6_prospective_chart_plan_20261115_22.json` and final `williams_v6_full_forward_expert_plan_20261228_20270104.json` shows that each consists of **32 cases**, **8 per edition**, **4 symbols (BTC/ETH/SOL/XRP)** across **8 daily UTC cutoffs**. Each case requests a **1024-H1-bar** historical lookback.

For **each cohort**, the same instrument's eight windows overlap for **all 28 possible pairs**, or **112 overlapping same-symbol pairs across four symbols**. Adjacent dates share **1000/1024 H1 bars = 97.65625%** of the lookback. This is not a violation of the frozen prospective cutoff, but it means those 32 cases cannot be assumed to be **32 independent statistical observations**. Daily market dates and cross-symbol conditions may also be correlated.

Thus the naive binomial Wilson95 interval must not be treated as independent skill qualification; use block/cluster methods and a newly precommitted genuinely diverse cohort for inferential certification. **Do not modify or reselect the 32 frozen cases.**

Collection remains future-gated: pilot 15–22 Nov 2026, final 28 Dec 2026–4 Jan 2027. No future market data fetched, no expert labels fabricated. Existing collection and human review modules remain intact. 
