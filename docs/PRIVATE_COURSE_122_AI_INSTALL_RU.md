# Подключение личного индекса Course (122 страницы) к Williams AI

Это инструкция к личному архиву пользователя, **не содержит саму книгу**.

## Что подготовлено

В приватном ZIP: `williams_all_four_private.sqlite3` (TC1/NTD/TC2/COURSE), `private_course_ai.sqlite3` (только Course), OCR-JSONL/карточки для поиска, оригинальные JPEG-изображения **всех 122 страниц**, ссылки на 68 страниц с диаграммами, исходные координаты абзацев, контрольные SHA, поисковый PDF и QA.

## Локальная установка (без GitHub-публикации)

1. Распаковать ZIP полностью в личную закрытую папку **вне клона GitHub** — например `/private/Williams_Course_122/`. Не разносить SQLite отдельно от `page_images/`: изображения привязаны по относительным путям и SHA-256.
2. Указать существующему Python Williams AI путь к `williams_all_four_private.sqlite3`. Все четыре книги доступны через `book_index.search_pages`. OCR скана Course возвращает статус `PRIVATE_OCR_UNVERIFIED_NEEDS_PAGE_IMAGE_REVIEW`, а описание страницы без текста — `PRIVATE_VISUAL_DESCRIPTION_UNVERIFIED_NEEDS_PAGE_IMAGE_REVIEW`.
3. При наличии собственного старого многокнижного индекса его можно **не менять**: `scripts/ai_merge_private_course.py --existing /private/my_old.db --course /private/Williams_Course_122/private_course_ai.sqlite3 --output /private/Williams_Course_122/my_four_books_with_course.sqlite3`. Команда создаёт новую копию и оставляет старый DB нетронутым.
4. Локальный исследовательский вызов с установленной **собственной** GGUF-моделью:

```bash
python -m williams_ai.reasoning_trial \
  --symbol ETHUSDT --bar-index 300 --edition COURSE \
  --local-model /private/models/your_local_qwen.gguf \
  --book-index /private/Williams_Course_122/williams_all_four_private.sqlite3
```

Это историческая научная проверка, не запуск торговли. Аргумент `--book-index` **должен указывать на действительный файл**, иначе новый индекс не попадёт в inference. Android-пакет не синхронизирует SQLite автоматически.

## Что может и не может нейросеть

Text-only Qwen3-4B получит OCR-текст и **предварительные описания** рисунков с номером PDF-страницы; она не получит пиксели сканов как вход для компьютерного зрения. Для утверждений о геометрии, степенях волн, стрелках и сложных рисунках нужен vision-вход с явной валидацией либо независимая экспертная разметка.

Вся OCR-разметка и изображения имеют статус **UNVERIFIED**. Их можно цитировать для исследования/поиска с проверкой по оригиналу, но **нельзя автоматически использовать как expert gold или допускать к реальной торговле**. Для QLoRA требуется отдельная проверенная человеческая обучающая выборка.

## Проверка

`course_visual_audit(db_path)` сверяет зарегистрированный Course SHA, наличие всех 122 страниц и SHA всех 122 изображений. `verified_private_book_pages(db_path, "COURSE", question="...", limit=2)` возвращает локальные OCR выдержки с отметкой и, где имеются, ссылки на рисунки. OCR-only PDF page 56 доступна по запросам к рисунку даже без распознанного непрерывного текстового блока.

Исходные 122 страницы и четыре книги **не коммитить** в GitHub, даже если источник их загрузки был пользовательским.
