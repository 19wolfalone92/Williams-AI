# Historian100 cohort selection correction, 14 Nov 2026

Initial run [43588031518](https://github.com/19wolfalone92/Williams-AI/actions/runs/43588031518) failed **before any valid 100-case corpus or score**: historical ZIP `PEPEUSDT-1h-2025-02.zip.CHECKSUM` returned 404 and read-only Binance market call confirmed **PEPEUSDT = invalid symbol (-1121)**.

The real USD-M perp is **1000PEPEUSDT**; current Binance exchange metadata confirms `TRADING/PERPETUAL` and onboarded in 2023, before all chosen 2025 cutoffs. This is a documented fixed contract-name correction for the four preselected PEPE slots, **not** replacement due to outcomes. The set remains **25 pairs × 4 historical dates = 100 cases**, 25/book. IDs retained for reproducibility, corrected `symbol` in each case, never a pseudo market.

Never present run 43588031518 as pass or validated data. No trained-model output, holdout exam, expert label, or trading score resulted from that failed job.
