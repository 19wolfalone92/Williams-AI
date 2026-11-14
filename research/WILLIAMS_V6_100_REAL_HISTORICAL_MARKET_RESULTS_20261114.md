# Williams AI v6 — 100 REAL Binance H1 historical charts: verified dataset and ongoing MODEL inference

**Observed: 2026-11-14 UTC** — [GitHub Actions real data run 43588356736](https://github.com/19wolfalone92/Williams-AI/actions/runs/43588356736), **SUCCESS**.

## Actual audited dataset result

- **100 / 100 historical charts passed** strict data checks; **25 distinct Binance USD-M perpetual pairs** × **four widely separated 2025 UTC cutoffs**: 2025-02-15, 2025-05-15, 2025-08-15, 2025-11-15.
- **100 × 1024 = 102,400 historical closed H1 bars** (valid aligned continuous OHLCV, H4/D1 AO/AC derived causally).
- **100 × 24 = 2,400 next-H1 bars**, **stored in a separate `future24/` directory**. These are evaluation-only and **never** used as input to the model.
- **200 real Binance official monthly ZIP archives, 200 SHA256 .CHECKSUM verifications**. Downloaded Action artifact **14111146341** has **201 files**: 100 `charts/`, 100 `future24/`, one `market_evidence.json`. The local download ZIP SHA256: `9de3dbfa3869a279c2b2c5f02376e207c722659390ad4619ab36975cffea8bbe`. Individual chart SHA256 recorded in `market_evidence.json`.
- All four book families are equally represented: **TC1 25 / NTD 25 / TC2 25 / COURSE 25**. Cases are prospective **from the moment of model inference** but the **charts themselves are historic**, not a prospective future qualification.
- Initial source selection erroneously used non-existent `PEPEUSDT`; it was documented and corrected to the genuine historical `1000PEPEUSDT` USD-M contract **before obtaining any completed dataset**. Current exchange metadata independently confirmed 25 valid tradable historical instruments. Previous failed run 43588031518 must NOT be cited as successful.

## Real market outcome, NOT model accuracy or AI PnL

| Historical cutoff (UTC) | Cases | Market higher over next 24h | Market lower over next 24h | Median 24h market move |
|---|---:|---:|---:|---:|
| 2025-02-15 | 25 | 5 | 20 | −2.4376% |
| 2025-05-15 | 25 | 1 | 24 | −4.3649% |
| 2025-08-15 | 25 | 5 | 20 | −1.5224% |
| 2025-11-15 | 25 | 24 | 1 | +1.8537% |
| **Total** | **100** | **35** | **65** | — |

These are **actual raw asset returns** from last pre-cutoff H1 close to next-24h H1 close, **not model performance** or Binance testnet profit. Cross-asset date correlation is strong: 24/25 move together on multiple dates, making naive 100-independent-binomial inference invalid. 100 charts are **not 100 independent proof observations**.

H1 AO sign: 25 positive / 75 negative. H4 AO: 59 positive / 41 negative. D1 AO: 56 positive / 44 negative. All three timeframes AO-positive simultaneously on just 1 chart; all three negative on 19 charts. **AO signs alone are not Williams signals**, and Alligator, confirmation prices, expert wave labels or exit position evidence are not established by this source-only pass.

By book assignment, each 25-chart group experienced a different distribution of market movement but **none is an actual model prediction yet**.

## Actual Qwen/LoRA comparison launched separately

New dedicated [GitHub Actions run 43588651963](https://github.com/19wolfalone92/Williams-AI/actions/runs/43588651963) performs two explicit stages:
1. Recreate **exactly the same 100** checksum-verified official market charts with separate future.
2. Recover real previously trained Qwen3-4B v5 LoRA (no retraining), verify pinned GGUF base/adapter hashes, run **matched Base + trained LoRA CPU inference on all 100 historical chart inputs**, store `decision`, `model_reason`, per-case chart/prompt hashes and both-model consistency; no orders.

**Do not mark the second stage as PASSED until its separate `ACTUAL-100-BASE-V5` artifact has a `complete=true` flag and contains 100+100 real model rows.** At the time this initial data report was authored, no such completed model result had been retrieved. No gold, accuracy or claimed superiority is fabricated. Even successful 100-model-output generation is not a qualified expert exam without independently graded source/chart evidence.

### Deliverables
- [Fixed 100-case selection](research/williams_v6_historical100_selection_20261114.json).
- [Real 100-case SHA-verified Binance collector](scripts/ai_v6_historical100_real_binance.py).
- [Actual model inference runner](scripts/ai_v6_historical100_models.py).
- [100-case research workflow](.github/workflows/ai-v6-historical100-real-data.yml).
- [Successful 100 historical chart artifact](https://github.com/19wolfalone92/Williams-AI/actions/runs/43588356736): file `Williams-V6-REAL-100-H1-HISTORICAL-CHARTS-NOT-MODEL-EXAM`, artifact ID 14111146341.
- `main` unchanged; original Williams Bot unchanged; no live or demo order sent; no GPU retraining.

**Required next scientific step**: read the actual completed Base/LoRA decision records, check internal contradictions and false-book-rule assertions against the real market evidence, then have independently qualified experts review selected difficult cases. Holdout/train overlap remains unverified; no model accuracy percentage can be awarded now.
