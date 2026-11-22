# Williams AI v6 — factual live preflight on 2026-11-13 (research only)

The requested **independent four-book expert market exam cannot finish on 2026-11-13**: the prospective date-scoped market cases have not closed, and two independently vetted domain experts have not signed unseen gold decisions. No training, order, GPU or main merge was attempted.

## What was actually executed now

**1. Real public Binance USDⓈ-M data feed:** through the connected read-only market-data API, requested four instruments `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `XRPUSDT`, **1024 hourly bars per instrument** ending on **2026-11-13 00:00 UTC**. All four calls returned 1024 ordered H1 bars. First bar: `1790841600000` ms; last close: `1794527999999` ms. Checked all hourly start offsets, expected close-time offsets, and OHLC high/low inequalities. **Zero detected structural failures.** This was a live API read, not mock tests. The endpoint was read-only: no account key or order. [Machine-readable connectivity smoke record](WILLIAMS_V6_PUBLIC_MARKET_SMOKE_20261113.json).

This **past** history is neither training nor untouched evaluation; it cannot be used to assert expert-model accuracy.

**2. Real prior CPU recovery run #43229923254:** downloaded its original GitHub Actions ZIP, artifact `13962336774`, produced at run head `e3f1b4d34059f10adb1b2791267bb11b79ec67c1`. The ZIP archive itself is **23,647,418 bytes** with SHA-256:

```
6ebaa38558fa4edc7423f398fc0574f807e3eb2ce74ae193ea076ae781fe9d7a
```

Its nested **encrypted** LoRA bundle's SHA-256 is **`4cc11816cc0b0acea494e9db84189d8203ebbfc79bf38c73e0f69cd14bcfeb66`**, independently recalculated and matched against `v5_training_audit.json`. The v5 declared *raw plaintext* adapter SHA-256 is `8bd26590df627c7452d56e81df6a4572885bfc82515259a7c9de4a3970c850b8` and was consistent across exam + training receipts, **but cannot be recalculated from sealed bytes without the separate pre-existing private decrypt secret**. No decryption was attempted.

The existing GitHub Actions artifact expires on **2026-12-03 16:22 UTC**, **before** the fully prospective study period closes on 2027-01-04. A bit-identical encrypted ZIP was therefore saved in **the user's private ChatGPT Library**, and independently downloaded back and byte-compared. Its outer ZIP SHA-256 after private Library round-trip was exactly `6ebaa38558fa4edc7423f398fc0574f807e3eb2ce74ae193ea076ae781fe9d7a`. No private encryption key, decrypted weights or private source PDFs were posted to GitHub.

**3. Independently recomputed all four edition aggregates from the real original exam artifact, instead of trusting `research_quality_gate`:**

| Evaluation (prior, developer-authored — not untouched expert market gold) | V5 Qwen+LoRA | Matched base | Book-specific regression |
|---|---:|---:|---|
| Closed-book 16 | 14/16 | 16/16 | TC1 3/4 vs 4/4; NTD 3/4 vs 4/4 |
| Source-assisted 16 | 15/16 | 16/16 | TC2 3/4 vs 4/4 |
| Rule-disjoint 28 | 24/28 | 23/28 | TC2 7/8 vs 8/8, including one abstention |

The validated training receipt reports **80 optimizer steps**, best checkpoint **80**, **zero test cases supplied to trainer**, and a lower per-book validation loss in all TC1, NTD, TC2 and COURSE. This proves source-card optimization occurred, **not** independent book comprehension or a safe entry/exit model. Historical `research_quality_gate.accepted_for_further_research` is correctly **false** under independent four-book recounting. An older checkpoint500 had different RAG/decoder conditions and is not directly comparable.

## Real blockers and scheduled next actions

1. The 32-case **pilot** H1 cutoffs end on 2026-11-22 00:00 UTC. The collection is only technically safe no earlier than **2026-11-22 02:00 UTC**. It is not a clean post-training history set; no expert gold is implied.
2. The **fully prospective** 32-case H1 windows close 2027-01-04 00:00 UTC. Collection is gated until **2027-01-04 02:00 UTC**; the H1 series must remain after the v5 training cutoff, per precommitted immutable plan.
3. Two **independent qualified experts**, separate signed labels with book-page/figure rationales, and the original recovered/converted authentic Qwen3-4B LoRA weights are still not present for actual evaluation. This is not fixable by synthetic labels or more repeats of old 28 tasks.
4. The future plans have frozen source hashes; selective editing to improve scores invalidates a prospective evaluation. The same RAG and decoding protocol must apply to the base and real v5 adapter.

Two one-time **ChatGPT automations** are active for the safe collection gates: pilot 2026-11-22 03:20 Europe/Amsterdam and fully prospective 2027-01-04 03:20 Europe/Amsterdam. They are instructed to read the exact research branch and update only the read-only collector request when complete data are available; after a terminal GitHub Actions run they must verify artifact/logs before reporting. These automations cannot manufacture expert decisions.

All work remains in `research/williams-ai-curriculum-v5-20261111` and **does not merge into `main`**. The user's personal encrypted artifact copy is not part of GitHub.
