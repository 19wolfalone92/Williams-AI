# Williams AI v7 — реальный guided-JSON пилот, 8 парных H1-кейсов

**Дата проверки:** 15 ноября 2026  
**Статус:** Kaggle kernel завершён; raw artifact скачан и проверен.  
**Назначение:** сравнить техническое завершение/структурированный вывод v7 с исходным v6 на восьми неизменных запросах. Это не проверка торгового качества.

## Проверенные запуски и артефакты

- Первичная отправка и неудачный долгий monitor: [GitHub Actions #43719042676](https://github.com/19wolfalone92/Williams-AI/actions/runs/43719042676). Первый job отправил один private T4 kernel. Второй monitor остановился после 225 проверок со статусом `KERNELWORKERSTATUS.QUEUED`; это был timeout мониторинга, а не доказанный сбой kernel.
- Исправленный read-only check: [GitHub Actions #43739479560](https://github.com/19wolfalone92/Williams-AI/actions/runs/43739479560). Оба jobs — SUCCESS. Submit job повторно использовал уже существующий kernel и явно записал `new_kernel_submitted=false`; audit job увидел `KERNELWORKERSTATUS.COMPLETE`, скачал результат и проверил парность.
- Результат: [Williams-V7-T4-REAL-GUIDED-8-vs-V6-RAW-NO-GOLD, artifact 14172295276](https://github.com/19wolfalone92/Williams-AI/actions/runs/43739479560/artifacts/14172295276). ZIP SHA-256: `5592a2790e0c22a08677a7c7dda5baa54fde5797d5a472a29b7d1a48feee2ba6`.
- Статус kernel: [artifact 14172280232](https://github.com/19wolfalone92/Williams-AI/actions/runs/43739479560/artifacts/14172280232). Повторная отправка kernel не выполнялась: [artifact 14171607171](https://github.com/19wolfalone92/Williams-AI/actions/runs/43739479560/artifacts/14171607171).

## Integrity gates

- В raw v7 JSON: `complete=true`, ровно 8 frozen IDs, по 8 ответов в `matched_base` и `candidate_v5`.
- Все четыре книги представлены по два случая: TC1, NTD, TC2, COURSE.
- Для каждой пары Base/v5 совпали `chart_sha256`, `prompt_sha256`, исходный полный prompt SHA, число видимых баров и input token count.
- Те же восемь IDs, chart SHA и prompt SHA совпали с соответствующими строками исходного v6 GPU artifact.
- Base: pinned Qwen3-4B revision `3fb356690bbdae7353c94efb3eecdce276e6ed1c`. Authentic v5 adapter SHA-256: `8bd26590df627c7452d56e81df6a4572885bfc82515259a7c9de4a3970c850b8`.
- Все prompts уложились в предел 3200 tokens (фактически 3092–3155).
- `expert_gold=null`, `book_gold=null`, `training_started=false`, `weights_changed=false`, `trading_allowed=false`.

## Сводка по finish reason

| Роль | Протокол | stop | error | length |
|---|---|---:|---:|---:|
| Qwen Base | v6 исходный | 3/8 | 5/8 | 0/8 |
| Qwen Base | v7 guided JSON | 2/8 | 6/8 | 0/8 |
| authentic v5 LoRA | v6 исходный | 5/8 | 2/8 | 1/8 |
| authentic v5 LoRA | v7 guided JSON | 4/8 | 4/8 | 0/8 |

В v7 все технические ошибки имеют код `INVALID_JSON`: 6/8 у Base и 4/8 у v5 LoRA. Это **не ABSTAIN** и не модельное решение. В v7 не было `length`-завершений.

## Валидные решения на тех же восьми запросах

Учитываются только строки с `finish_reason=stop` и валидным решением; технические error/length не засчитываются как решения.

| Книга | Base v6 | Base v7 | v5 LoRA v6 | v5 LoRA v7 |
|---|---|---|---|---|
| TC1 | 2 ABSTAIN | 0; 2 INVALID_JSON | 2 ABSTAIN | 0; 2 INVALID_JSON |
| NTD | 1 INSUFFICIENT_DATA; 1 error | 0; 2 INVALID_JSON | 0 valid; 2 error | 2 ABSTAIN |
| TC2 | 0 valid; 2 error | 0; 2 INVALID_JSON | 2 INSUFFICIENT_DATA | 0; 2 INVALID_JSON |
| COURSE | 0 valid; 2 error | 2 WAVE_HYPOTHESIS | 1 WAVE_HYPOTHESIS; 1 length | 2 WAVE_HYPOTHESIS |

Итого среди валидно завершённых строк:
- Base v6: 2 ABSTAIN, 1 INSUFFICIENT_DATA; 5 технических error.
- Base v7: 2 WAVE_HYPOTHESIS; 6 технических INVALID_JSON.
- v5 LoRA v6: 2 ABSTAIN, 2 INSUFFICIENT_DATA, 1 WAVE_HYPOTHESIS; 2 error и 1 length.
- v5 LoRA v7: 2 ABSTAIN, 2 WAVE_HYPOTHESIS; 4 технических INVALID_JSON.

## Интерпретация и ограничение диагностики

На этом маленьком frozen subset guided decoder v7 **не улучшил техническое завершение** относительно v6: у Base stop уменьшился с 3 до 2, у v5 LoRA — с 5 до 4; ошибки INVALID_JSON составили соответственно 6 и 4 из 8. У LoRA исчезло одно `length`-завершение, но часть прежних stop-ответов превратилась в INVALID_JSON. Это только сравнение декодирования/формата на восьми парных запросах.

Raw artifact сохраняет `output_sha256`, но не сам невалидный raw generation text. Поэтому точную причину каждого INVALID_JSON (обрыв, лишний текст, нарушение грамматики или декодирование) из текущего артефакта восстановить нельзя. Не следует приписывать ошибку конкретной причине без этих данных. Следующий безопасный шаг — отдельный unit/property test для JSON decoder и bounded diagnostic capture в исследовательском режиме; повторный GPU-прогон не требуется и не запускается автоматически.

**Никакая точность по книгам не вычислялась.** Нет expert gold, независимой разметки или оценки правильности торговых сигналов. Эти данные не разрешают новое обучение, SFT, торговлю или интеграцию с Williams Bot.
