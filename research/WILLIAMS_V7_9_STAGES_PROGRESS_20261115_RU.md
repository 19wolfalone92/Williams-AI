# Williams AI — 9 этапов, фактическое выполнение на 15 ноября 2026

## Неизменяемые результаты 100 исторических H1 (этап 1: ЗАВЕРШЁН)

- CPU GGUF/llama.cpp: [GitHub Actions SUCCESS 43639364206](https://github.com/19wolfalone92/Williams-AI/actions/runs/43639364206), [JSON 14150632357](https://github.com/19wolfalone92/Williams-AI/actions/runs/43639364206/artifacts/14150632357): 100 Base + 100 authentic v5 LoRA, 10/10 успешных шардов, 27 различий классов. Обе модели `stop=95, error=4, length=1`, 5 технических отказов каждая. Хеши одинаковых prompt/chart пары проверены агрегатором.
- GPU HuggingFace NF4/T4: [SUCCESS 43658377350](https://github.com/19wolfalone92/Williams-AI/actions/runs/43658377350), [JSON 14138759783](https://github.com/19wolfalone92/Williams-AI/actions/runs/43658377350/artifacts/14138759783): 100 Base +100 genuine v5, 47 различий; Base stop43/error57; LoRA stop66/error24/length10. Исторические 98 случаев вход до 3200 токенов, последних 2 fallback до 4096; CPU и GPU **не одинаковый протокол**.
- Формальная перекрёстная проверка: [postflight CI 43718298629 SUCCESS](https://github.com/19wolfalone92/Williams-AI/actions/runs/43718298629), [artifact 14163615724](https://github.com/19wolfalone92/Williams-AI/actions/runs/43718298629/artifacts/14163615724). Проверены 100 SHA-пиннинг case/chart исходного корпуса, исходный CPU full prompt SHA записан в GPU output и совпадает; отдельные счётчики `technical ABSTAIN` vs `model ABSTAIN`, нет gold.
- **Неизвестно:** правильность торговых сигналов, 95–99% знание книг, PnL. Никакой независимой экспертной проверки 100 рыночных решений не было.

## 2. Декодер: РЕАЛЬНЫЙ T4 PILOT ЗАВЕРШЁН, ФОРМАТНЫЙ РЕЗУЛЬТАТ НЕУДОВЛЕТВОРИТЕЛЕН

- Новый, **не меняющий старую модель/веса**, модуль `williams_ai/v7_structured_decoder.py`: schema-guided JSON с `lm-format-enforcer==0.11.3`, enum решения по ENTRY/EXIT/WAVE, bounded `reason`, явный `decision=null` при формате/length/инфраструктурной ошибке (не подделывать модельный `ABSTAIN`).
- [Контрактные unit-тесты и загрузка JSON grammar SUCCESS](https://github.com/19wolfalone92/Williams-AI/actions/runs/43718621379).
- Первоначальный [T4 run 43719042676](https://github.com/19wolfalone92/Williams-AI/actions/runs/43719042676): submit прошёл, но audit monitor остановился после 225 проверок со статусом `KERNELWORKERSTATUS.QUEUED`. Это не доказывало сбой kernel.
- Исправленный [read-only reuse/check run 43739479560](https://github.com/19wolfalone92/Williams-AI/actions/runs/43739479560): оба jobs SUCCESS; уже существующий kernel переиспользован, `new_kernel_submitted=false`, Kaggle вернул `KERNELWORKERSTATUS.COMPLETE`.
- Реальный [artifact 14172295276](https://github.com/19wolfalone92/Williams-AI/actions/runs/43739479560/artifacts/14172295276), SHA-256 ZIP `5592a2790e0c22a08677a7c7dda5baa54fde5797d5a472a29b7d1a48feee2ba6). Проверено 8 одинаковых запросов Base/v5, по 2 на TC1/NTD/TC2/COURSE, одинаковые chart/prompt SHA, pinned Qwen3-4B revision и authentic v5 adapter SHA.
- Технические завершения: Base v6 `stop=3,error=5,length=0`; Base v7 `stop=2,error=6,length=0`; v5 LoRA v6 `stop=5,error=2,length=1`; v5 LoRA v7 `stop=4,error=4,length=0`. Все ошибки v7 — `INVALID_JSON`: 6/8 Base и 4/8 LoRA. Это **не ABSTAIN и не книжная точность**. В v7 валидно получены 2 `WAVE_HYPOTHESIS` у Base и 2 `ABSTAIN` + 2 `WAVE_HYPOTHESIS` у LoRA.
- Raw artifact сохраняет output SHA, но не невалидный текст; точную причину каждого `INVALID_JSON` нельзя восстановить. Повторный GPU-прогон и переобучение не запускались. Подробности: [отчёт Stage 2](WILLIAMS_V7_GUIDED_JSON_8CASE_PILOT_RESULTS_20261115_RU.md).

## 3. Сверка первоисточников: ВЫБОРОЧНО ПРОВЕРЕНА, НЕ ВСЯ

[Первичный spot-check четырёх книг](WILLIAMS_AI_V6_PRIMARY_BOOK_SPOTCHECK_AND_GAPS_20261115_RU.md): TC1 PDF72–74/89–93; NTD99/103–104; TC2 121–124,135–139,147–149; Course графические страницы 43–45. Проверены фрактальная отмена, Balance Line 1 vs 2 новых уровней, WM1 angulation, WM2 3 AO-изменения, WM3 Teeth именно на *activation*, W4 переход через AO=0 как минимальное условие, а не завершение W4. **Это не проход всех 642 страниц** и не gold-разметка.

- Дополнительно завершена точная сверка **RULE_CHECK_005 / TC2 WM1 angulation** по оригинальному Library PDF с совпавшим SHA-256 `055b7924f9a4d165294bf3e12069e74b121f6ca510f25ad310a07c94f0a6107f`.
- Физические PDF pages 121–124 соответствуют печатным страницам 106–109. Визуально проверены Figure 9.1–9.5, включая отрицательный UOPX пример. Четыре правила геометрии записаны; источник не даёт численного порога, фиксированного lookback или регрессионной формулы. Расстояние от Jaw само по себе недостаточно.
- Затем завершена **RULE_CHECK_006 / TC2 WM2** по тому же SHA-проверенному оригиналу: physical PDF pages 135–139 = printed pages 120–124. Визуально проверены Figures 10.1–10.5. Уточнено, что WM2 — add-on после исходного WM1 входа; long: три последовательных растущих (green) AO histogram bars и buy stop на один тик выше high соответствующего price bar; short example: третий падающий (red) AO bar, который может наступить через шесть баров после WM1, и add-on sell stop на один тик ниже соответствующего price bar. Цвет — изменение AO относительно предыдущего бара, не положение относительно нуля. Это не три чередующихся изменения и не обязательный AO zero-cross.
- Для WM2 также не выдумываются максимальный срок ожидания, размер add-on или подтверждение биржевого fill. Сигнал, pending stop, price trigger и fill остаются разными состояниями.
- [Отчёт RULE_CHECK_005](WILLIAMS_V7_RULE_CHECK_005_TC2_ANGULATION_PRIMARY_SOURCE_RECHECK_20261115_RU.md) и [отчёт RULE_CHECK_006](WILLIAMS_V7_RULE_CHECK_006_TC2_WM2_THREE_AO_BARS_SOURCE_RECHECK_20261115_RU.md). Машинные записи и регрессионные проверки сохранены; [CI #43743422164 SUCCESS](https://github.com/19wolfalone92/Williams-AI/actions/runs/43743422164) и [CI #43743619083 SUCCESS](https://github.com/19wolfalone92/Williams-AI/actions/runs/43743619083). RULE_CHECK_008 также source-rechecked. Независимых рецензентов по-прежнему 0.

## 4. Пробелы: ВЫЯВЛЕНЫ

- В исторических 100 случаях всего по **одному фиксированному вопросу на книгу**, поэтому 25 case по книге не покрывают WM1–WM3, решения по всем волнам, позиции и сопровождение.
- TC2 гипотетический выход спрашивается без подтверждённой биржевой позиции/заявок; высокий `HOLD`/INSUFFICIENT_DATA не значит умение сопровождать кампанию.
- Source RAG даёт максимум 3 paraphrase cards; BOOK PDF visual/diagrams и скрытые контексты вне prompt нельзя считать усвоенными.
- GPU-выводы 34/100 LoRA `error/length`, форматная проблема не может быть названа экспертным воздержанием.

## 5. Учебная база: ПОДГОТОВЛЕН КАНДИДАТНЫЙ REVIEW-БЭКЛОГ, НЕ TRAIN-READY

[10 source-pinned задач для независимой проверки](research/williams_v7_review_targets_20261115.json) с оригинальными PDF SHA и страницами, отрицательными/двусмысленными сценариями. `training_approved:false`, `valid_gold_labels:null`, `reviewer_signatures_received:0`. Автоматический запрет выдавать предварительные примеры за gold: [CI SUCCESS 43719443582](https://github.com/19wolfalone92/Williams-AI/actions/runs/43719443582). Качественные целевые торговые SFT пары можно составить только после проверенных разметок и независимого контроля source/case leakage.

## 6. Дополнительное LoRA: ОЖИДАЕТ КАЧЕСТВЕННЫХ МЕТОК

Запускать обучение сейчас на ответах той же модели или на 100 случаях, от которых уже измерялись результаты, будет data leakage/pseudo-gold. Без проверенных первичных примеров и второго независимого эксперта фактическое улучшение знаний неподтверждаемо. Поэтому никаких ложных «50 успешных глубоких обучений» или необъяснимого GPU расхода.

## 7. Независимый экзамен: ПРОТОКОЛ ГОТОВ, ЭКСПЕРТЫ/МЕТКИ ОТСУТСТВУЮТ

Уже есть `williams_ai/independent_exam_v6.py`, `v6_expert_review_packets.py`: приём собственных H1, заморозка SHA, два независимых подписанных Ed25519 expert review и grade только на реальных экспертных решениях. Два ключа сами по себе не доказывают компетентность рецензентов. **Независимый gold-экзамен не сдавался.**

## 8. Будущие H1: ЗАРАНЕЕ ЗАФИКСИРОВАНЫ

Есть исходный prospective план 32 кейса 15–22 ноября 2026, технический сбор запланирован после cutoff **22 ноября 2026**, более поздний полный перспективный экзамен — январь 2027. Сбор и экспертная оценка не могут честно быть «досрочно завершены». Уже есть отдельный автоматический контроль; повторный сбор не создавать, не допускать future-peek.

## 9. Williams Bot: ТОЛЬКО НА СТАДИИ GO/NO-GO ОБСУЖДЕНИЯ

Отдельный репозиторий `19wolfalone92/Williams` **не изменялся**. До экспертной оценки нельзя автоматически подключать эту LoRA к торговле. Возможный первый шаг после успешных safety gates — строго **read-only/shadow-advisory** с журналированием причин, без ключей/ордеров/контроля Stop Loss, который остаётся у детерминированного бота.

### Stop Conditions

При отсутствии экспертных меток сохраняй статус `book_accuracy=UNKNOWN`, `trading_qualification=false`, `training_approved=false`; никакие изменения синтаксиса JSON, скорость GPU или процент совпадающих Base/LoRA классов не заменяют экспертное знание.

---

# Дополнение после первичного отчёта — 15 ноября 2026, 22:00 +03

## Stage 2.1 — исправление guided JSON: v7.1 запущен, результат ещё не получен

Установлена вероятная причина ошибок v7: upstream LM Format Enforcer имел дефект incremental tokenizer decoding, где нормализация пробелов могла терять запятые/кавычки и преждевременно разрешать EOS. Официальное исправление upstream — commit `d4d9b7cc20937bcd3a643be7a8dafb7b1ca20dfb` от 4 апреля 2026 года, в котором декодирование выполняется с `clean_up_tokenization_spaces=False`. Это совпадает с короткими invalid JSON в v7, однако не доказывает причину каждого старого ответа: raw generation text в v7 не был сохранён. Старая v7 выборка остаётся неизменной.

Ветка теперь содержит отдельный v7.1 decoder `williams_ai/v71_guided_json_diagnostics.py`, который закрепляет точный upstream commit, записывает raw decode (с/без special tokens), IDs/токены, EOS и prefix-token trace. Новые веса не обучаются и не меняются.

- Настоящий приватный Kaggle T4 kernel создан с коротким slug `wolf19loner92/williams-v71-lmfe8-20261115`: [submit CI 43752606966 SUCCESS](https://github.com/19wolfalone92/Williams-AI/actions/runs/43752606966), artifact [14177175095](https://github.com/19wolfalone92/Williams-AI/actions/runs/43752606966/artifacts/14177175095).
- Read-only проверка существующего kernel: [CI 43752856794 SUCCESS](https://github.com/19wolfalone92/Williams-AI/actions/runs/43752856794), artifact [14176916708](https://github.com/19wolfalone92/Williams-AI/actions/runs/43752856794/artifacts/14176916708).
- Последний фактический Kaggle status на момент отчёта: **QUEUED**. Новая инференция ещё не подтверждена как выполненная; paired result artifact не появился. Следующий контроль может только прочитать статус/результаты того же kernel — повторная отправка не нужна.

## Stage 3 — десять проверок source rules по оригинальным книгам

Подготовлены отдельные SHA-проверенные recheck документы (выборочные страницы, не полное чтение книг):

1. [RULE_CHECK_001 — TC1 Fractal Start/Signal и отмена](WILLIAMS_V7_RULE_CHECK_001_TC1_FRACTAL_START_SIGNAL_CANCEL_PRICE_INVALIDATION_SOURCE_RECHECK_20261115_RU.md).
2. [RULE_CHECK_002 — TC1 Zero Point и волновая кампания](WILLIAMS_V7_RULE_CHECK_002_TC1_WAVE_CAMPAIGN_RISK_STOP_AND_EXECUTED_EXITS_SOURCE_RECHECK_20261115_RU.md).
3. [RULE_CHECK_003 — NTD Balance Line base и 1/2 экстремума](WILLIAMS_V7_RULE_CHECK_003_NTD_BALANCE_LINE_BASE_AND_ONE_TWO_EXTREMES_SOURCE_RECHECK_20261115_RU.md).
4. [RULE_CHECK_004 — NTD pending buy stop и новая база](WILLIAMS_V7_RULE_CHECK_004_NTD_PENDING_BALANCE_LINE_BUY_STOP_BASE_REPLACEMENT_SOURCE_RECHECK_20261115_RU.md).
5. [RULE_CHECK_005 — TC2 WM1 angulation](WILLIAMS_V7_RULE_CHECK_005_TC2_ANGULATION_PRIMARY_SOURCE_RECHECK_20261115_RU.md).
6. [RULE_CHECK_006 — TC2 WM2 три AO bars](WILLIAMS_V7_RULE_CHECK_006_TC2_WM2_THREE_AO_BARS_SOURCE_RECHECK_20261115_RU.md).
7. [RULE_CHECK_007 — TC2 WM3 Teeth при активации](WILLIAMS_V7_RULE_CHECK_007_TC2_WM3_TEETH_AT_ACTIVATION_SOURCE_RECHECK_20261115_RU.md).
8. [RULE_CHECK_008 — TC2 противоположный сигнал при открытой позиции](WILLIAMS_V7_RULE_CHECK_008_TC2_OPPOSITE_POSITION_SOURCE_RECHECK_20261115_RU.md).
9. [RULE_CHECK_009 — Course W4: zero-cross, не касание](WILLIAMS_V7_RULE_CHECK_009_COURSE_W4_AO_ZERO_CROSS_NOT_TOUCH_SOURCE_RECHECK_20261115_RU.md).
10. [RULE_CHECK_010 — Course степень волны и W5 target zone](WILLIAMS_V7_RULE_CHECK_010_COURSE_WAVE_DEGREE_W5_TARGET_DIVERGENCE_SOURCE_RECHECK_20261115_RU.md).

Результат — проверенные источниковые формулировки, негативные условия и состояния. Это не экспертная разметка реальных графиков и не подтверждение правильности модели.

## Stage 5 — source-grounded curriculum подготовлен, но не одобрен для SFT

Создан `knowledge/WILLIAMS_V7_SOURCE_CURRICULUM_CANDIDATES_20261115.json`: 10 тематических уроков, привязанных к редакции, SHA PDF, PDF pages и rule IDs. В каждом уроке выделены книжные принципы, последовательность состояний, недопустимые выводы и открытые вопросы. Есть cross-book confusion guards (например, WM2 three AO bars не смешивается с Course W4 AO zero-cross).

- CI [43752708130 SUCCESS](https://github.com/19wolfalone92/Williams-AI/actions/runs/43752708130) проверил все десять карточек, точные PDF SHA, привязку Rule IDs к своей книге и отсутствие псевдо-gold/решений.
- В `williams_ai/v7_training_data_contract.py` добавлен отдельный приватный конструктор train/dev кандидатов и sealed holdout. Он требует две разные подписи Ed25519 по одной frozen выборке, согласованные метки и общие страницы источника, проверенный rule map, предзарегистрированный split, отсекает пересекающиеся same-symbol окна, запрещает model-prediction labels и пишет файлы локально create-only 0600. CI [43752095880 SUCCESS](https://github.com/19wolfalone92/Williams-AI/actions/runs/43752095880) использовал исключительно синтетические fixture/key для проверки кода — **это не реальные экспертные метки**.
- Статус базы: содержимое подготовлено для экспертного рассмотрения, но `expert_approved=false`, `independent_expert_reviews=0`, `sft_ready=false`. Исторические 100 кейсов не превращаются в train data; будущий H1 остаётся holdout.

## Остальные gates без изменений

- Stage 6: новая LoRA не обучалась; без действительных независимых expert labels она заблокирована.
- Stage 7: независимый gold-exam не проводился; протокол freeze/grade готов, личности/компетентность подписантов должны проверяться вне кода.
- Stage 8: 32 prospective H1 остаются под зафиксированным планом; сбор после cutoff 22 ноября 2026, дальнейшая holdout выборка — январь 2027.
- Stage 9: Williams Bot repo не менялся; ни API торговли, ни ордеров, ни интеграции нет.

---

# Дополнение по дальнейшей первичной сверке и Stage 5 — обновление 15 ноября 2026

## Stage 3: NTD расширен с 4 до 6 отдельных проверок (RULE_CHECK_011–016)

Углублённо изучены оригинальные страницы NTD, включая сканы Figures 3-1…3-8, 5-1…5-35, 6-1…6-14, 7-1…7-4, 8-7…8-23 и 9-1…9-9. Исходный PDF закреплён SHA-256 `855eb052353c852f208436952b12e46d9a18badf7c1dac7ade7580d57b9f4bb4`.

11. RULE_CHECK_011 — Alligator: три временных уровня Balance Lines, фаза сна/пробуждения, направление по стороне пасти, first external fractal, close-only stop. **Найдено противоречие перевода:** Lips shift на PDF36 описан как +2 в одном предложении, но Figure 3-4/резюме и следующий текст указывают +3. Также фраза PDF38 о фрактале «внутри челюсти» требует сверки с внешним условием на PDF35/39.
12. RULE_CHECK_012 — NTD AO: три самостоятельных buy и sell паттерна Saucer, Zero Line Cross, Twin Peaks. Цвет столбца — направление изменения histogram, не знак относительно нуля. AO Zero Cross — сигнал; Twin Peaks действителен только при сохранении стороны нуля между пиками. Не смешивать с TC2 WM2.
13. RULE_CHECK_013 — NTD AC: сторона нуля меняет счёт последовательных растущих/падающих histogram bars. Сам AC zero-cross самостоятельным сигналом не является; он корректирует необходимое число столбцов. Не путать с AO Zero Cross.
14. RULE_CHECK_014 — NTD Zone: согласованность AO+AC создаёт Green/Red/Gray; добавление требует согласованной цены на close и текущей позиции; после пятого подряд одноцветного price bar зональные добавления прекращаются, но другие допустимые сигналы в том же направлении остаются.
15. RULE_CHECK_015 — Balance Line + Alligator + Zone filters: правая-к-левой поиск базы, 1/2 экстремума от/к линии, Special Blue Light/rebase, запрет покупать под Alligator mouth и продавать над пастью, удвоение числа требуемых экстремумов для BUY in Red Zone и SELL in Green Zone.
16. RULE_CHECK_016 — четыре рыночных выхода NTD: Teeth close-only; trailing stop после пяти последовательных баров одного цвета; Lips close-only на running market; противоположный qualifying signal как менее предпочтительное exit decision, а не автоматическое reverse.

[CI NTD source rechecks 43753636409](https://github.com/19wolfalone92/Williams-AI/actions/runs/43753636409) проверяет PDF SHA, страницы, допустимые NTD rule IDs, и отсутствие model labels. [Curriculum CI 43753636502](https://github.com/19wolfalone92/Williams-AI/actions/runs/43753636502) проверяет расширенный source lesson backlog (на момент записи запущены).

## Stage 5: учебный curriculum расширен с 10 до 16 кандидатов

`knowledge/WILLIAMS_V7_SOURCE_CURRICULUM_CANDIDATES_20261115.json` содержит 16 отдельных lessons, из них 6 NTD: Alligator, AO, AC, Zone additions, Balance Line with Zone filters, exit hierarchy. В карточках хранятся первоисточник SHA, страницы, разрешённые Rule IDs, требуемая последовательность состояний, недопустимые выводы и нерешённые вопросы.

Это source-grounded curriculum candidate, а не самостоятельные экспертные метки графиков: `independent_expert_reviews=0`, `chart_labels=0`, `sft_ready=false`. Stage 6 по-прежнему заблокирован до появления независимых качественных labels; никакой новой LoRA не обучалось.

## Stage 2: v7.1 GPU pilot без нового запуска

Последний подтверждённый one-shot read-only статус прежнего kernel `wolf19loner92/williams-v71-lmfe8-20261115` — **QUEUED**: [CI 43752856794 SUCCESS](https://github.com/19wolfalone92/Williams-AI/actions/runs/43752856794), [status artifact 14176916708](https://github.com/19wolfalone92/Williams-AI/actions/runs/43752856794/artifacts/14176916708). В этом workflow явно записано: новый kernel не submitted, новая инференция не начата, обучение не начато. Поэтому результат эффективности upstream fix пока **неизвестен**; не подавать дубль. Предполагаемая причина v7 invalid JSON — tokenizer punctuation-cleanup bug в старом LM Format Enforcer; это правдоподобная техническая гипотеза, но отсутствие raw текста из первоначального прогона не позволяет восстановить конкретный каждый ответ.

## Рамки доказательности остаются неизменными

- 100 historical H1 решений на CPU/GPU не имеют экспертного gold и не измеряют accuracy.
- 16 curriculum lessons охватывают конкретные проверенные страницы четырёх книг, но не все 37 записей rule registry и не все 642 PDF-страницы.
- Психологические и визуальные элементы книг требуют отдельной review: их нельзя свести только к механическим карточкам.
- Пока нет внешне верифицированных экспертов и согласованных подписанных решений, Stage 6 LoRA retraining и Stage 7 independent gold exam — BLOCKED.
- Перспективный Stage 8 остаётся замороженным до cutoff по существующим планам; Williams Bot не менялся.

---

# Обновление по Stage 3 и Stage 5 — расширена первичная база четырёх книг

## Stage 3: добавлены RULE_CHECK_017–029

### TC1 — страницы 43–47, 49–65, 82–84, 129–140
- **017:** MFI и Profitunity Windows Green/Fade/Fake/Squat; Squat может быть measuring squat внутри W3; Air Bag — защитный, а не входной сигнал.
- **018:** выбор волновой степени; 100–140 баров относится к выбранной степени; MFI W3–W5 divergence; 5/34/5 signal line отдельно от zero line; crossing нуля — минимальные требования W4, не завершение W4; W5 target 62–100% только после независимого completion W4.
- **019:** пять Magic Bullets; Zero Point — гипотеза потенциальной смены поведения, не гарантия разворота.
- **027:** четыре психологические программы автора (fear/self-preservation, greed, confidence, pleasure). Авторские проценты и нейропсихологические объяснения не выдаются за установленную науку; AI не может диагностировать состояние человека по OHLCV.
- **028:** последовательность Inside Risk → Structure → I Can → Pleasure/Capital; worst-case loss планируется до рынка; структура включает wave degree, higher timeframe, Air Bag, rhythm, fractal entries/exits. «Risk-free» у автора — сохранение возможности продолжать торговлю, не нулевая вероятность денежного убытка.

### TC2 — страницы 150–165
- **020:** стандартная последовательность WM1→WM2→WM3 типична, но не обязательна; допустимый WM2 или WM3 может стать первым сигналом, если раньше сработал и прошёл фильтр.
- **021:** campaign additions, reverse pyramid 1–5–4–3–2 как исторический пример, а не универсальный risk model; trailing stop на 3 или 5 завершённых барах.
- **022:** обычный trailing exit и WM1 stop-and-reverse — разные механизмы; один только противоположный setup не закрывает позицию; частичная фиксация опциональна.

### Course — сканы pp 24–29, 56, 69–72, 88–89
- **023:** MFI и четыре Profitunity окна; тик-объём не равен Binance base volume, поэтому точная имитация MFI требует корректной volume source.
- **024:** Thumb setup; inside bars не считаются пальцами; выявлено различие между текстовым критерием p69 (2 из 3 SQUAT/GREEN в любой комбинации) и идеальной сводкой p71 (минимум один SQUAT и один GREEN). Сохраняем разницу, не подгоняем.
- **025:** Hump / Five-Fingered Boogie — пара противоположных фракталов с неперекрывающимися экстремальными формационными барами; первый — Hump Start, второй — Hump Signal.
- **026:** Trading Partner — таблица мульти-ТФ контекста, rhythm, Hump/Thumb, позиции и планируемого действия. Это структурированная процедура, а не набор булевых индикаторов и не команда на биржу.
- **029:** визуальная вложенность родительских 1–5 и коррекционной A–B–C структур со вложенными волнами меньшей степени; рисунок иллюстрирует degree nesting, не алгоритм распознавания силуэта.

## Stage 5: 29 source-grounded curriculum lessons закрывают все rule IDs текущего реестра

Текущий файл: `knowledge/WILLIAMS_V7_SOURCE_CURRICULUM_CANDIDATES_20261115.json`. В нём **29 lessons** с точным PDF SHA, страницами, правильным Rule ID из соответствующего издания, последовательностью состояний, негативными выводами и нерешёнными ограничениями.

| Издание | Source lesson cards | Уникальные rule IDs покрыты | Всего rule cards в реестре |
|---|---:|---:|---:|
| TC1 | 7 | 14/14 | 14 |
| NTD | 8 | 8/8 | 8 |
| TC2 | 7 | 8/8 | 8 |
| Course | 7 | 7/7 | 7 |
| **Всего** | **29** | **37/37** | **37** |

Покрытие реестра означает, что на каждый текущий rule ID есть хотя бы один source lesson. Оно **не означает полного исчерпывающего покрытия всех 642 PDF страниц**, всех визуальных примеров, всех рыночных режимов или независимую правильность трактовки. Нужен независимый источник/экспертный аудит, особенно для переводческих разногласий и диаграмм.

**Curriculum CI:** [43755098705 SUCCESS](https://github.com/19wolfalone92/Williams-AI/actions/runs/43755098705) — 29 lessons валидированы, все страницы/хэши в допустимых диапазонах, rule IDs существуют и принадлежат правильной книге, псевдо-gold не найден. Обучение не запускалось.

## Что по-прежнему заблокировано
- Независимых экспертов и подписанных реальных review решений: **0**.
- Экспертных меток реальных торговых графиков: **0**.
- SFT-ready: **false**; новая LoRA не обучалась.
- Независимый gold-exam: не проводился.
- Historical 100 model answers остаются без expert gold и не используются для обучения.

---

# Финальное дополнение по доступной сейчас работе — 15 ноября 2026

## Дополнительные первичные проверки: RULE_CHECK_027–029
- **027 TC1:** страницы 129–136 — четыре авторские психологические программы Fear/Self-Preservation, Greed, Confidence, Pleasure. Доли пользователей и объяснения «мозговых программ» рассматриваются как модель автора, не доказанная нейронаука. AI нельзя автоматически диагностировать по OHLCV.
- **028 TC1:** страницы 136–138 — Inside Risk → Structure → I Can → Pleasure/Capital. Worst-case потери всех защитных стопов оцениваются до рынка; если совокупный риск выше пользовательского болевого порога, объём сокращается заранее. «Risk-free» не означает отсутствие денежного риска.
- **029 Course:** визуальная страница 56 — родительские 1–5 и A–B–C коррекции со вложенными подволнами разных степеней. Диаграмма не является алгоритмом определения Elliott count по силуэту.

## Итог Stage 5: 29 уроков, 37/37 текущих rule IDs покрыты
Текущая source-based curriculum в `knowledge/WILLIAMS_V7_SOURCE_CURRICULUM_CANDIDATES_20261115.json`:
- TC1 — 7 уроков, 14/14 rule IDs;
- NTD — 8 уроков, 8/8;
- TC2 — 7 уроков, 8/8;
- Trading Chaos Course — 7 уроков, 7/7.
Итого 29 уроков покрывают все 37 rule cards текущего Source Rule Registry. [Curriculum quality CI 43755098705 SUCCESS](https://github.com/19wolfalone92/Williams-AI/actions/runs/43755098705).

**Ограничение:** это покрытие текущего реестра rule cards, а не исчерпывающее прочтение всех 642 страниц книг, всех схем, примеров или психологических разделов. Несогласованности переводов/иллюстраций остаются предметом независимого review.

## Stage 9: интеграция с ботом — документирован NO-GO
Добавлен [Stage 9 GO/NO-GO gate](WILLIAMS_AI_STAGE9_BOT_INTEGRATION_NO_GO_GATE_20261115_RU.md). Он фиксирует, что AI не должен владеть API-ключами, ордерами, Execution Door, risk state или position state. Будущая допустимая ступень после независимой проверки — отдельный read-only shadow, а не автономная торговля. Репозиторий Williams Bot не менялся.

## Важная терминологическая поправка по Stage 1 / Stage 7
Сравнительный CPU inference на 100 исторических H1 charts для Base и v5 LoRA завершён технически. Но поскольку историческим 100 графикам не присвоен независимый экспертный gold, **это не завершённый независимый экспертный экзамен модели** и не доказательство, что она усвоила книги. Stage 7 remains BLOCKED until an externally vetted independent gold cohort is prepared and graded. GPU/CPU model agreement is not correctness.

## Статусы, которые не изменились
- Stage 2: decoder v7.1 private T4 job `wolf19loner92/williams-v71-lmfe8-20261115` last reported QUEUED by read-only status workflow 43753869176; no new inference or training has been launched. The original 0.11.3 constrained-decoder pilot on eight cases showed worse JSON validity than v6; the upstream tokenizer whitespace fix awaits an actual run.
- Stage 6: no additional LoRA trained; training is BLOCKED by lack of real independent expert-approved supervision.
- Stage 7: no genuine independent gold exam yet.
- Stage 8: 32 prospective H1 cases remain frozen under original plan; they cannot be collected/graded ahead of the locked future cutoff. Existing prospective workflow/automation should not be duplicated.
- Stage 9: NO-GO for any code or order integration; only the future read-only shadow design has been specified.
