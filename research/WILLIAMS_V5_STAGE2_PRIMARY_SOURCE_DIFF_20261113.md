# Williams AI — Stage 2: targeted primary-source review (2026-11-13)

**Scope:** original user-provided books, **not** secondary internet retellings. This is a *partial targeted review* of error-related pages, NOT a claim to have read all 642 PDF pages or visually verified every figure. No changes to frozen source card registry or training split hashes.

## Confirmed source passages and provisional audit findings

| Rule under audit | Source PDF and physical PDF pages | Observations / action |
|---|---|---|
| `TC1.RHYTHM` (V5 error n=6; same error in base) | `1. Торговый Хаос - 1.pdf`, PDF pp. 53–54 | The 100–140 bars discussion on p.54 is specifically about examining the internal wave **c** within corrective W4, in the process of seeking an entry into W5. The existing registry paraphrase merely says "appropriate scale of the observed sequence"; this loses the critical W4→c→W5 context. **Proposed editorial correction, not yet in frozen registry:** state the wave-c/W4 context explicitly; changing timeframe to inspect a comparable nested segment is not permission to invent bars or act without other confirmations. |
| `NTD.ZONE.ADDED` (V5 error n=6; same error in base) | `Билл_Вильямс_Новые_измерения_в_биржевой_торговле_2_книга.pdf`, pp. 92–93 | p.92 lists conditions for **zone additions**: rising AO, rising AC, and favorable price close. p.93 says to stop *zone-based additions* after five consecutive red/green zone bars; signals **from other dimensions in the same direction may still be used**. Candidate and baseline both failed the known synthetic n=6 exercise; no basis to claim a newly learned misconception caused this. |
| `NTD.ALLIGATOR` (V5 error n=6; same error in base) | same NTD PDF, pp. 34–37 | Jaw/Teeth/Lips are displaced smoothed averages and must not read future OHLC when backtesting. On p.36 a **text-internal apparent contradiction** exists: an isolated block states green Lips shifted by **2** bars, while the later full specification says **5-period smoothed / 3-bar shift** (standard Alligator 13/8, 8/5, 5/3). Do **not** silently reconcile via OCR; inspect the printed figure / original edition before adjudicating the contradictory passage or writing a numerical gold label. |
| `TC2.REVERSE` (V5-specific n=9 abstention) | `Торговый Хаос 2 (3книга).pdf`, p. 131 | The chapter illustrates closing an open long and entering net short when the valid opposite signal triggers. It also describes placing a protective stop immediately beyond the opposite end of the initial signal bar. The model's recorded n=9 choice was **empty**, with `finish_reason=stop` and no fallback; this **does not prove semantic rejection** of the reversal rule because raw completion is not archived. |
| `TC2.WM1.ANG` (no test error, context-critical) | same TC2 PDF, pp. 122–124, figures 9-3, 9-4 and 9-5 | Angulation requires *diverging slopes* of price vs Alligator from a passage through its mouth, not merely a large fixed jaw distance or an AO sign. The source explicitly treats a reversal bar without the angulation as **invalid**. Graph figures 9-3/9-4 were visually inspected; source text/figures support the existing cautious paraphrase, not a universal numeric threshold. |

### Fourth book cross-check: Course

Also checked `Bill Williams - Trading Chaos Course.pdf` scanned physical PDF **p.44** against `Course_122_SEARCHABLE_PRIVATE_OCR_UNVERIFIED.pdf` pp.43–45. The scan confirms that moving across the 5/34/5 oscillator's zero line establishes **minimum requirements for W4** and explicitly does **not** prove that W4 is over. The nearby instruction uses a **100–140 bar** observation scale for the wave sequence, then asks for structural correction patterns and other confirmations. OCR contains obvious stray handwritten/garbled fragments; those are **not** promoted to verified rule text. `COURSE.W4.ZERO` is correctly cautious in the frozen registry, and no model regression on this synthetic rule is observed. This is a *targeted* Course source check, not a claim that all 122 Course pages/figures were validated.

## Source-card changes proposed for expert approval, NOT yet committed into frozen v5 source

```json
{
  "rule_id": "TC1.RHYTHM",
  "edition": "TC1",
  "source_pdf_pages": [53, 54],
  "proposed_paraphrase": "В контексте поиска входа в волну 5 после коррекции волны 4 ориентир 100–140 баров в TC1 относится к внутренней волне c; временной масштаб выбирают для наблюдения структуры, не растягивают сами данные. Ориентир не заменяет волновые подтверждения и не является самостоятельным торговым входом.",
  "reviewer_status": "SOURCE_TEXT_CHECKED_PENDING_SECOND_REVIEW",
  "override_frozen_training_dataset": false
}
```

**Important:** the existing `SOURCE_RULE_CARDS.json` and `knowledge_curriculum_v5.py` are byte-identical to run `e3f1b4d34059f10adb1b2791267bb11b79ec67c1`; modifying either in place would invalidate the historical source-registry/manifest fingerprints and mix a **new protocol** into an old comparison. Any approved changes must go to a new versioned curriculum and a new independently authored evaluation set, not be used to repair the already-seen test answers.

## Immediate blocker list

1. The 16-with-source TC2 regression cannot be tied to a specific prompt or source card from the old ZIP; per-case projection has now been added to `v5_postfit_exam.py` for **future** CPU-only exam receipts without exposing the gold or raw reasoning.
2. The 28-case `TC2.REVERSE` abstention cannot be attributed to knowledge or decoder from a stop reason alone.
3. The existing 28 MCQs are source-card paraphrases with two blatantly non-book distractors; accuracy on them is not proof of applying Williams rules to unseen real price charts.
4. No expert-reviewed visual wave examples / original Course diagrams are present in the v5 train/validation manifest (despite a manual check of Course p.44 in this investigation); **GPU retraining and promotion remain blocked**.

## Safeguards

Do not change exam keys, do not patch model responses, do not claim percentage of book coverage from number of cards, do not mix the new source-note revisions into historical experiment `43229923254`, do not deploy or touch `main`, no exchange keys or GPU.
