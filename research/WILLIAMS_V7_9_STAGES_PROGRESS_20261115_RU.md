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
