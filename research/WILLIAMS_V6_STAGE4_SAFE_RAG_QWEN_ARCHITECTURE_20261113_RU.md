# Williams AI v6 — Этап 4: RAG + детерминированная проверка книги + Qwen3-4B (13.11.2026)

**Реализация:** `williams_ai/v6_research_pipeline.py` в исследовательской ветке. Предназначена только для изучения источников и разъяснения уже **закрытых** рыночных данных. **Никаких торговых ключей, GPU, торговых действий или merge в main.**

## Реальная архитектура

```
read-only closed OHLCV observation
         |
  strict causal admission (no future as_of/candidates/H4/D1/native frames;
  gaps=0, execution_allowed=false, exact symbol/edition)
         |
  frozen source registry: exact Git blob b09553108... checked on every request
         |
  weighted_v2 RAG: question-only, same edition, max 3 already page-reviewed
  public paraphrase cards, SHA-256 of original PDF, citations
         |
  (optional) offline LocalWilliamsModel Qwen3-4B GGUF CPU
    propose_chart_reasoning(..., prior_lessons=[RAG cards])
         |
  EXISTING algorithm_guard.audit_reasoning(chart, model_proposal)
         +
  independent citation boundary (model can cite only the retrieved rules)
  pending-source corrections quarantine
         |
  immutable safe projection: NO order / NO exchange SL or fill claims /
  NO training / NO main merge; only BLOCKED or PROVISIONAL research text
```

Unlike a purely diagrammatic architecture note, **the wrapper calls the existing model and independent algorithm guard** when verified local model weights are explicitly provided. Without a local model it performs real CPU-only source lookup, but **does not invent** any neural inference.

The retrieved reference is not book text, chart labels, evidence of an expert-reviewed wave degree, or permission for stop/position changes. Importantly, source cards with unapproved revisions (six targeted cards in `research/williams_v6_source_rule_proposals_20261113.json`) are explicitly marked as *not yet independently certified*. Historical v5 `knowledge/SOURCE_RULE_CARDS.json` and SFT splits are unchanged; any future approved source edit requires a **new versioned experiment**.

### Safety properties

- No candidate/gold `case_id` or answer options used in RAG. No answer key goes into inference.
- A legitimate source from the wrong edition or a rule not returned by question-only RAG cannot be declared evidence of the model's selected setup.
- Missing RAG evidence: **ABSTAIN** without calling Qwen. A model exception, malformed response, future chart observation, unconfirmed signal or execution request fails closed.
- Existing `algorithm_guard` independently checks book-specific setup/side/entry-state, TC2 WM3 pending vs fresh touch, causal timestamps, unacknowledged STOP/fills, unsupported wave degree and contradictory model explanations.
- The pipeline returns `trade_action:null`, `execution_allowed:false`, `gpu_training_allowed:false`, `main_merge_allowed:false` **even when the AI answer passes all structural checks**.
- A guard PASS means only `PROVISIONAL_BOOK_REASONING_NOT_TRADE`; it is NOT model reliability / book mastery / safety of an order.

### How to use on a read-only stored observation

**Without model weights** (only verified question-specific source retrieval; no model call):

```bash
python -m williams_ai.v6_research_pipeline \
  --chart /private/closed_chart_analysis.json \
  --edition TC2 \
  --question "Какие доказательства нужны для подтверждения третьего Мудреца?" \
  --out /private/tc2_source_only.json
```

**With an authenticated local Qwen3-4B GGUF** (CPU-only, optional, actual inference only if weights exist):

```bash
python -m williams_ai.v6_research_pipeline \
  --chart /private/closed_chart_analysis.json \
  --edition TC2 \
  --question "Подтверждён ли по закрытым свечам третий Мудрец?" \
  --model-gguf /private/Qwen3-4B-Q4_K_M.gguf \
  --base-sha256 f6f851777709861056efcdad3af01da38b31223a3ba26e61a4f8bf3a2195813a \
  --out /private/tc2_cpu_report.json
```

Optional v5 LoRA requires **all three flags** `--lora-gguf`, `--lora-sha256`, `--lora-manifest`; adapter must first be legitimately unsealed and converted to actual GGUF. No decryption, downloading, GPU fallback or key access is performed by this module. `LocalWilliamsModel` provides SHA-/manifest-bound runtime and no network API. Do not supply raw `safetensors` as GGUF.

## Версионированная база знаний v6 (только КАНДИДАТ)

Добавлен `williams_ai/v6_rule_registry.py`: он детерминированно собирает все карточки четырёх книг из **исходного неизменного** V5 реестра, привязывает PDF SHA-256 и страницы и отдельно помечает **шесть уточнений, ожидающих второй независимой проверки**. Сохраняет первоначальный текст и предложенное уточнение раздельно; отдельно отмечает спорные рисунки Course и смещение губ Alligator. Вся база содержит `training_allowed:false`, `execution_allowed:false`, ноль экспертно проверенных рыночных меток. Отдельного статуса «изучено 100% книг» нет.

```bash
python -m williams_ai.v6_rule_registry --out /private/williams_v6_candidate_source_index.json
python -m unittest tests.test_williams_ai_v6_rule_registry -v
```

Это **не тренировочная база**: она служит реестром источников и независимых замечаний для будущего двойного рецензирования. Ни одному из шести неподтверждённых предложений не присвоен экспертный статус автоматически. `SOURCE_RULE_CARDS.json` не редактировался; обновление весов LoRA запрещено до создания новой независимой выборки.

## Testing and limitations

- `tests/test_williams_ai_v6_research_pipeline.py` covers all books, edition-pinned SHA, source-only CLI, model input separation, source ID hallucinations, pending source corrections, execution attempts, future H4/native/candidate timestamps, gapped observations, schema/runtime failures, exact GGUF byte hash and no-overwrite report behavior.
- `.github/workflows/ai-v6-readonly-reasoning-cpu.yml` runs on Python 3.11 without credentials, model downloads or GPU. Full read-only tests run separately on PR.
- **Not run:** a real 4B+v5 LoRA market chart exam. The historical recovered adapter is sealed and the necessary independent expert-labelled chart set does not exist. Faking performance with synthetic mocks would be a false report.
- **Still blocked:** v5 source-assisted TC2 regression, `TC2.REVERSE` unanswered raw-output root cause, missing independent visual wave labels, Course diagram uncertainty, and no two qualified experts signed off on a genuinely unseen cohort.

## Required next step (not GPU)

Prepare a new truly independent, source-reviewed chart cohort using `williams_ai/independent_exam_v6.py`. Fix any identified **software** defects against separate constructed tests, **without** changing exam keys or tuning model prompts to revealed outcomes. Keep Qwen3-4B+LoRA frozen until the double expert-reviewed evaluation exists. Do not claim 95% book mastery on the historic 28 synthetic questions.
