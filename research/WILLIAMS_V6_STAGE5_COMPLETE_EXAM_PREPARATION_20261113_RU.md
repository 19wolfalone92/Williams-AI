# Williams AI v6 — комплект готовности независимого экзамена (13.11.2026)

**Состояние:** код, планы отбора, шаблоны рецензирования, одинаковый CPU-протокол Qwen3-4B/LoRA и локальные автоматические проверки подготовлены. **Фактической экспертной разметки нет; реальная точность модели не измерена.** Торговля, GPU и слияние в `main` запрещены.


## Подтверждённые реальные данные и исправление GitHub HTTP 451 (13.11.2026)

**Выполнено:** публичный Binance USD-M H1 тест на реальных OHLCV через официальный источник с проверкой SHA-256. GitHub Actions [run 43398296898](https://github.com/19wolfalone92/Williams-AI/actions/runs/43398296898) — **success**, оффлайн unit-тесты **5/5**; в артефакте `Williams-v6-REAL-H1-H4-D1-SMOKE-NOT-EXAM` сохранены только цифровые хеши и сведения о проверке, **не** ответы экспертов.

4 пары BTC/ETH/SOL/XRP: **по 1024 настоящих замкнутых H1-свечи**, период 19.09.2026 08:00 — 01.11.2026 00:00 UTC, на каждую пару H4=256 закрытых, D1=42 закрытых, AO/AC доступны. Для каждой из двух месячных ZIP проверен официальный `.CHECKSUM` SHA-256 с `data.binance.vision`. Это **исторический smoke-test**, не незнакомая послеобучающая выборка, не показатель точности Qwen/LoRA.

**Root cause устранён для будущего сбора:** GitHub hosted runner получает HTTP **451** на прямом `fapi.binance.com/fapi/v1/klines`. Новая реализация `williams_ai/v6_official_archive_collector.py` получает строго те же заранее назначенные UTC H1-окна **из официальных Binance daily ZIP**, проверяет SHA-256 каждого архива против опубликованного `.CHECKSUM`, 24 UTC-бара в день, идентичность 1024 последовательных H1. Не использует VPN, прокси, закрытые API, биржевые ключи, не подменяет недостающие дни. Результаты привязываются к заранее зафиксированным файлам планов; изменение даже одного байта плана отклоняется.

**Ограничение:** дневные архивы иногда публикуются с задержкой. Если архив за последний UTC-день отсутствует, сбор завершается блокером; не дополняйте его непроверенными котировками. Контрольные задачи на пилот и финал запланированы соответственно **24.11.2026 05:10 CET** и **06.01.2027 05:10 CET** с часовыми повторными проверками, но результаты пока не существуют. Исходный GitHub artifact v5 run 43229923254 истекает **03.12.2026 16:22 UTC**; в чате сохранена скачанная зашифрованная копия ZIP, SHA-256 `6ebaa38558fa4edc7423f398fc0574f807e3eb2ce74ae193ea076ae781fe9d7a`. Отдельно хранить приватный секрет расшифрования, **не публиковать** в GitHub.

## Что подготовлено в GitHub

| Компонент | Назначение |
|---|---|
| `research/williams_v6_prospective_chart_plan_20261115_22.json` | ПИЛОТ: 32 заранее назначенных H1-момента, 8 UTC-дней, 4 книги × 8, ротация BTC/ETH/SOL/XRP. Последние свечи после обучения, но начало 1024-часового окна **может пересекаться** со старой историей. Нельзя выдавать за полностью независимый holdout. |
| `research/williams_v6_full_forward_expert_plan_20261228_20270104.json` | ФИНАЛЬНЫЙ КАНДИДАТ: 32 других заранее назначенных момента с 28.12.2026 по 04.01.2027; **все 1024 H1-бара каждого окна начинаются после 13.11.2026 UTC**. Модель/протокол должны остаться неизменными. Требуется внешний аудит источника и возможных пересечений с другими наборами. |
| `williams_ai/v6_public_chart_collector.py` | Только публичный Binance USDⓈ-M GET `/fapi/v1/klines`, 1024 строго замкнутых H1-свечи, 32 кейса, SHA-256, сверка UTC и запрещение частичных файлов. Никаких API-ключей, ордеров или GPU. |
| `williams_ai/v6_history_context.py` | Детерминированная агрегация H1/H4/D1 только из закрытых баров, AO=(SMA5-SMA34) медианной цены, AC=AO-SMA5(AO). Эти численные признаки **не** подтверждают Elliott degree, ангуляцию и вход. |
| `williams_ai/v6_expert_review_packets.py` | Безответные частные формы для двух рецензентов, **локальные** Ed25519-ключи (0600), ручное подписание после проверки первоисточников, trust-шаблон с **false**-разрешениями по умолчанию. Код не подтверждает квалификацию реального человека. |
| `williams_ai/v6_v5_lora_binding.py` | Отдельный SHA-проверяемый паспорт **исторического** Qwen v5 Kaggle LoRA на 80 шагах; не подменяет несовместимый контракт локального `model_bundle` и не декодирует закрытый архив. |
| `williams_ai/v6_expert_model_runner.py` | Реальный оффлайн-интерфейс Qwen3-4B GGUF / настоящий GGUF LoRA, SHA-проверка, одинаковые системные сообщения, RAG/контекст/seed/декодирование для базовой модели и кандидата, явный ABSTAIN при ошибках, **не знает эталонных ответов**. Переоценивает источниковые хеши и заново строит каждый промпт при объединении отчётов. |
| `williams_ai/independent_exam_v6.py` | Строгая фиксация корпуса, проверка двух внешне зарегистрированных подписей, подсчёт по TC1/NTD/TC2/COURSE, отказов, критических ошибок и Wilson95. Отсутствие независимой разметки блокирует выставление оценки. |
| `.github/workflows/ai-v6-expert-intake-cpu.yml` | Только статические/мок-тесты без биржевых запросов, секретов, GPU и весов. |
| `.github/workflows/ai-v6-collect-forward-h1.yml` | **Только явный запуск**: изменение отдельного заблокированного по умолчанию файла `research/v6_public_chart_collection_request.json` в исследовательской ветке (push trigger). Выбор `pilot` или `fully_forward` загружает публичные свечи в временный artifact `UNREVIEWED`. `workflow_dispatch` из UI может быть недоступен, пока workflow отсутствует в default branch; менять `main` для этого не нужно. Gold не хранится. |

## Календарь и различие пилота и финала

Все даты заданы в UTC, чтобы исключить неоднозначное время. Пилот можно собирать **не раньше 22 ноября 2026, 02:00 UTC**, финальный полностью перспективный набор — **не раньше 4 января 2027, 02:00 UTC**. Попытка раньше завершится явным `PROSPECTIVE_CANDLE_NOT_YET_CLOSED` **до вызова биржи**.

Финальный набор имеет 32 события/32 снимка; он полезен для проверки независимости и обнаружения грубых регрессий, **но даже 32/32 правильных недостаточно для нижней 95%-ной доверительной границы >=95%**. Соседние окна на 1024 часа и одновременные рыночные даты сильно коррелируют: обычный интервал Wilson — лишь описательная метрика. Для формального подтверждения нужен внешний аудит независимости случаев и оценка неопределённости с учётом кластеров по символам/датам. После проверки механики нужно заранее зафиксировать расширенный набор (200+; нужный объём зависит от фактической точности и доли ошибок), не выбирая кейсы по ответам модели. Не переносить пилот в финал, не дообучаться на раскрытых кейсах.

## Точный порядок выполнения после получения закрытых баров

**1.** Исследовательская ветка остаётся отдельно от `main`. После фактического закрытия всех будущих окон в GitHub изменить только `research/v6_public_chart_collection_request.json`: `request` → `COLLECT_NOW`, `cohort` → `fully_forward`, `request_id` → новую уникальную строку длиной от восьми символов. **Push только этого файла** запускает `ai-v6-collect-forward-h1.yml` в исследовательской ветке. До этого файл имеет `request=DISABLED`; сейчас запуск не произведён. Прямой `Run workflow` через UI может отсутствовать, поскольку GitHub требует наличия workflow_dispatch-файла в default branch. При HTTP 451 на прямом FAPI скрипт использует ТОЛЬКО официальные архивы `data.binance.vision` с проверкой SHA-256; если они ещё не опубликованы или недоступны, сбор честно блокируется, без подставных данных и без proxy. Биржевые ключи запрещены.

```bash
python -m williams_ai.v6_public_chart_collector \
  --plan research/williams_v6_full_forward_expert_plan_20261228_20270104.json \
  --out /private/WILLIAMS_V6_UNREVIEWED_32_FORWARD_H1.zip
```

**2.** Распаковать ZIP **в отдельный приватный каталог**, где `cases.json` лежит рядом с `charts/`. Проверить SHA реальных свечей и манифест независимо:

```bash
python -m williams_ai.independent_exam_v6 freeze \
  --cases /private/cohort/cases.json --chart-root /private/cohort \
  --out /private/cohort/frozen_manifest_RECHECK.json
```

Результат должен быть байтово эквивалентен сохранённому `frozen_manifest.json`; не подменять ошибки. Обязательно отдельно проверить биржевой источник по второй независимой выгрузке, корректность часовых границ, даты/символы и отсутствие пересечений c тренировочными и раскрытыми наборами.

**3.** Два **реальных квалифицированных независимых эксперта** отдельно получают только книги, свечи и описание случая, **без** ответов моделей и друг друга. Создать две формы:

```bash
python -m williams_ai.v6_expert_review_packets template \
  --cases /private/cohort/cases.json --manifest /private/cohort/frozen_manifest.json \
  --chart-root /private/cohort --reviewer-id expert_A \
  --out /private/expert_A_draft.json
```

Для второго использовать `expert_B`. Дефолтные `decision=INSUFFICIENT_DATA`, `reviewable=false`, `NOT_REVIEWED` — **заглушки, не метки**. Квалификация, независимость, оригинальная страница/рисунок и обоснование каждой метки проверяются людьми.

**4.** Каждый эксперт **на своём устройстве** генерирует отдельные ключи, не загружает закрытый seed в GitHub/чат и подписывает только законченную рецензию:

```bash
python -m williams_ai.v6_expert_review_packets new-local-signing-key \
  --private-key /private/expert_A.ed25519_seed \
  --public-key /private/expert_A.ed25519_pub
python -m williams_ai.v6_expert_review_packets sign \
  --draft /private/expert_A_reviewed_draft.json \
  --private-key /private/expert_A.ed25519_seed \
  --out /private/expert_A_signed.json --confirm-full-independent-review
```

Сборщик создаёт `trust.json` с обоими флагами **false**:

```bash
python -m williams_ai.v6_expert_review_packets trust-scaffold \
  --reviewer-a expert_A --public-a /private/expert_A.ed25519_pub \
  --reviewer-b expert_B --public-b /private/expert_B.ed25519_pub \
  --out /private/trust_UNVERIFIED.json
```

Лишь независимый координатор после реального подтверждения квалификации/независимости может **локально** выставить обоим `externally_vetted_expert=true` и `independent_of_training=true`. Две цифровые подписи сами по себе не доказывают компетентность.

**5.** Фактический CPU-экзамен требует `Qwen3-4B GGUF` с SHA-256 `f6f851777709861056efcdad3af01da38b31223a3ba26e61a4f8bf3a2195813a` и **реального проверенного GGUF LoRA** из v5, плюс манифест загрузчика. Исторический SHA `8bd26590df627c7452d56e81df6a4572885bfc82515259a7c9de4a3970c850b8` относится к исходному `adapter_model.safetensors`, **не** к конвертированному GGUF: не подставлять его вместо проверенной новой суммы. 

**Важное исправление совместимости:** исторический `training_receipt.json` Qwen3-4B v5 имеет схему `WILLIAMS_V5_GENUINE_QWEN3_4B_SOURCE_ONLY_LORA`, а модуль `model_bundle.runtime_identity` ожидает другую локальную тренировочную схему; подделывать её нельзя. Для v5 подготовлен отдельный `v6_v5_lora_binding.py`, проверяющий настоящие 80 optimizer steps, четыре книги, SHA исходного адаптера, результат GGUF конвертации и неизменность receipt. Исходный LoRA нужно получить **только законным приватным способом**, уже предусмотренным старым CPU-recovery, затем конвертировать локально с `llama.cpp` commit `970b51ce7c9ba5fff0726c5e9994f1c2f0e4eedc`. Никаких повторных GPU jobs.

```bash
# После локального восстановления исходных safetensors и pinned converter:
python /private/llama-cpp/convert_lora_to_gguf.py \
  --base /private/Qwen3-4B-pinned-tokenizer-config \
  --outtype f16 --outfile /private/verified_v5_lora.gguf \
  /private/v5_recovered_adapter/

python -m williams_ai.v6_v5_lora_binding \
  --base /private/qwen3-4b.gguf \
  --lora-gguf /private/verified_v5_lora.gguf \
  --raw-safetensors /private/v5_recovered_adapter/adapter_model.safetensors \
  --training-receipt /private/v5_recovered_adapter/training_receipt.json \
  --out /private/v5_recovered_adapter/v5_historic_lora_binding.json
```

Затем в `infer-one` передать `--lora-manifest /private/v5_recovered_adapter/v5_historic_lora_binding.json` и SHA-256 фактического GGUF-файла. Все исходные веса, секреты и манифесты остаются приватными. Изолировать модель; не запускать GPU, не использовать реальные торговые ключи.

```bash
python -m williams_ai.v6_expert_model_runner infer-one \
 --cases /private/cohort/cases.json --chart-root /private/cohort \
 --role matched_base --model-gguf /private/qwen3-4b.gguf \
 --base-sha256 f6f851777709861056efcdad3af01da38b31223a3ba26e61a4f8bf3a2195813a \
 --out /private/base.json

python -m williams_ai.v6_expert_model_runner infer-one \
 --cases /private/cohort/cases.json --chart-root /private/cohort \
 --role candidate_v5 --model-gguf /private/qwen3-4b.gguf \
 --base-sha256 f6f851777709861056efcdad3af01da38b31223a3ba26e61a4f8bf3a2195813a \
 --lora-gguf /private/verified_v5_lora.gguf \
 --lora-sha256 <SHA256_ACTUAL_GGUF_LORA_FILE> \
 --lora-manifest /private/v5_recovered_adapter/v5_historic_lora_binding.json \
 --out /private/candidate.json

python -m williams_ai.v6_expert_model_runner combine \
 --cases /private/cohort/cases.json --chart-root /private/cohort \
 --candidate /private/candidate.json --base /private/base.json \
 --out /private/matched_predictions.json
```

**6.** Только ПОСЛЕ подписания обоих секретных экспертных отчётов и независимого исполнения модели:

```bash
python -m williams_ai.independent_exam_v6 grade \
 --cases /private/cohort/cases.json --chart-root /private/cohort \
 --manifest /private/cohort/frozen_manifest.json \
 --review-a /private/expert_A_signed.json --review-b /private/expert_B_signed.json \
 --trust /private/trust_VETTED.json --predictions /private/matched_predictions.json \
 --out /private/WILLIAMS_V6_4BOOK_EXPERT_RESEARCH_SCORE.json
```

Эталонные решения и исходные страницы **никогда не передаются в промпт Qwen**. Сравнивать checkpoint500 напрямую нельзя — его RAG/формат/декодирование отличаются. Даже успешно проведённый новый тест **не активирует торговлю** и не разрешает GPU/merge без отдельного решения.

## Реальная готовность 13 ноября

✅ Подготовлены: код H1/H4/D1, сборщик без ключей, неизменяемый план, SHA-контроль, формы и подписывающий инструмент для независимых экспертов, два сопоставимых протокола GGUF, раздельный приватный gold, CI с искусственными свечами.

⛔ Отсутствуют: будущие реальные 32 графика, два фактически проверенных эксперта, их подписанная разметка, доступный на CPU оригинальный base GGUF и конвертированный проверенный v5 LoRA с манифестом, реальные ответы модели и точность на неизвестной истории. **Не говорить, что независимый экзамен выполнен.**

Контрольные средства `source_ranker_v2` и `v6_research_pipeline` остаются research-only, `main` не меняется.
