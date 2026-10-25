# Единый Williams AI: Qwen, LoRA, книги и опыт

`WilliamsAI` — один локальный экземпляр модели для изучения, практики и
разбора рынка. LoRA подключается к той же Qwen, а не запускается как второй
советчик. Все новые записи обучения, ошибки, прошлые paper-исходы и
дополнительные материалы хранятся в одной приватной SQLite памяти.
Оригинальные книги сохраняют отдельный неизменяемый индекс первоисточников.

Смена базы, адаптера или коэффициента образует другую идентичность модели.
Старые записи остаются в общей памяти, но не становятся доказанным знанием
новой версии. Общие исходные материалы доступны для повторного чтения.
Базовый fingerprint сохранён совместимым с прежними checkpoint-файлами.
Перед разбором память также выбирает уже изученный фрагмент этой же модели:
перечитывается его нынешний оригинал с проверкой хеша. Исправленный OCR
отменяет старый указатель; собственный пересказ не передаётся как правило.

## Запуск

```bash
python -m williams_ai.unified_ai \
  --local-model /private/qwen.gguf \
  --book-index /private/Williams_AI_private_books.sqlite3 \
  --memory /private/Williams_AI_Unified_Memory.sqlite3 \
  import-memory /private/old-study.sqlite3

python -m williams_ai.unified_ai \
  --local-model /private/qwen.gguf \
  --book-index /private/Williams_AI_private_books.sqlite3 \
  --memory /private/Williams_AI_Unified_Memory.sqlite3 \
  study --budget-per-book 2

python -m williams_ai.unified_ai \
  --local-model /private/qwen.gguf \
  --book-index /private/Williams_AI_private_books.sqlite3 \
  --memory /private/Williams_AI_Unified_Memory.sqlite3 \
  analyze --symbol ETHUSDT --bar-index 900
```

`status` показывает изучение каждого из четырёх источников текущей моделью,
счётчики сохранённых материалов и фактическую загрузку адаптера. `practice`
использует существующий источник-ориентированный учебный курс. Это повторная
практика, не независимый экзамен. Материалы добавляются через
`retain-material FILE --source ID --title TITLE --edition TC2 --available-at ISO`.
Полный текст сохраняется, в текущий запрос выбираются ограниченные фрагменты.
Материалы с более поздней датой и другого издания исключаются. Они остаются
непроверенными исследовательскими данными и не заменяют книгу.

## Подключение настоящего обученного адаптера

Текущий `private_lora` записывает хеши HF safetensors и конфигурации вместе
с фактическим числом шагов и loss. `model_bundle` конвертирует существующий
PEFT-адаптер в GGUF локально, без сетевого скачивания и pickle-fallback:

```bash
python -m williams_ai.model_bundle \
  --trained-adapter /private/trained-peft \
  --hf-base /private/qwen-hf \
  --base-gguf /private/qwen.gguf \
  --llama-cpp /private/pinned-llama-cpp \
  --output /private/runtime-adapter
```

Нужен чистый checkout llama.cpp на `970b51ce7c9ba5fff0726c5e9994f1c2f0e4eedc`.
Затем в тот же запуск `unified_ai`, `study_cli` или `reasoning_trial` передаются
`--lora-adapter /private/runtime-adapter/adapter.gguf` и
`--lora-manifest /private/runtime-adapter/runtime_manifest.json`.
Они доходят до настоящих `llama_cpp.Llama(lora_path=..., lora_scale=1.0)`.
Ошибка привязки/загрузки завершает запуск; скрытого возврата к чистой базе нет.

Хеши фиксируют заявленную пару файлов и изменения после подготовки. Они
не удостоверяют личность тренера или эквивалентность HF/GGUF исходников:
происхождение этой пары остаётся декларацией оператора, а совместимость
тензоров проверяет native runtime. Успешная загрузка и меньший validation loss
не дают допуска к торговле.

## Проверка и ограничения

Импорт старой памяти идемпотентен и атомарен. Конфликтующие записи не
перезаписываются. Старые модельные идентификаторы, статусы и исходные даты
сохраняются; подтверждение происхождения цитаты не становится экспертизой.
На каждый анализ подаются только закрытые свечи и уже известные результаты.
Собственные пересказы остаются изолированы от проверенных книжных правил.

Объединение кода не означает, что LoRA уже обучена, книги поняты полностью,
все изображения распознаны или рыночное преимущество установлено. Фактическая
память показывает оставшиеся OCR/учебные пробелы. Публичный CI проверяет
реальную Qwen без частных книг и без обученного LoRA; применение настоящих
адаптерных весов и экспорт нуждаются в отдельно предоставленном адаптере.
Android-APK пока не получает новый Python-адаптер автоматически.
Ордеров и автоматического допуска Demo/live этот интерфейс не содержит.

Первичные API: [llama-cpp-python](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/)
и [конвертер llama.cpp](https://github.com/ggml-org/llama.cpp/blob/970b51ce7c9ba5fff0726c5e9994f1c2f0e4eedc/convert_lora_to_gguf.py).
