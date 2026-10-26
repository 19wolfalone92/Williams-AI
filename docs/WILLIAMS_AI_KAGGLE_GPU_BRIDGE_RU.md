# Бесплатное GPU-обучение Williams AI: GitHub → приватный Kaggle

## Что реализовано
`.github/workflows/ai-private-kaggle-gpu.yml` — **только ручной workflow_dispatch**, не автоматическое ежедневное расходование квоты. GitHub предоставляет CPU для подготовки **одного приватного** Kaggle kernel и вызывает официальный `kaggle kernels push`. GPU выбирается `NvidiaTeslaT4` (актуальный пул T4×2, квота и наличие не гарантированы). `launch_gpu=false` делает `--no-run`, то есть только сохраняет приватную версию без запуска.

**GitHub не загружает никакие книги, размеченные графики, модели или обучающие данные.** Они должны быть отдельно и заранее размещены пользователем в приватных входах Kaggle с правом облачной обработки. Публичный репозиторий содержит только код и проверки.

## Подключение
1. Подключить свой аккаунт Kaggle, получить API token через настройки Kaggle и сохранить его как GitHub Actions secret `KAGGLE_API_TOKEN` в `19wolfalone92/Williams-AI`. Не пересылать токен в чат и не класть в код.
2. Отдельно подготовить законно разрешённые приватные Kaggle dataset + Hugging Face Qwen safetensors model source; для JSONL действуют существующие пороги и строгая независимая экспертная проверка `checked_private_training_data`. Телефонный GGUF не подходит.
3. GitHub → Actions → **Williams AI private Kaggle GPU QLoRA** → Run workflow. Заполнить свои `kernel_id`, `dataset_source`, `model_source`, пути к JSONL и HF weights внутри `/kaggle/input/...`. Отметить разрешение на облачную обработку.
4. Начать с `launch_gpu=false` (сохранить приватный тестовый kernel без расходования GPU), проверить метаданные. Для реального теста вручную `launch_gpu=true` запускает максимум 10 шагов оптимизатора, затем сохраняет контрольную точку **в Kaggle output**, которую пользователь должен сохранить как приватную следующую входную версию.
5. Проверить отчёты и сохранить полный собственный V2 output как новый приватный Kaggle dataset. Для следующего ручного сеанса указать **тот же `training_revision`**, JSONL и HF base; в `resume_config` передать JSON ниже. Генератор добавляет приватный input и worker автоматически выбирает последнюю запечатанную точку, проверяет контракт и SHA, затем продолжает следующие 10 шагов. Отсутствующий/изменённый/чужой checkpoint не заменяется новым обучением молча. Автоматический запуск следующих сеансов и сохранение Kaggle Output не реализованы: расходование квоты запускается только вручную.

```json
{"dataset_source":"your-user/own-qlora-checkpoints","directory":"/kaggle/input/own-qlora-checkpoints/williams-qlora-session","trusted_own":true}
```

`trusted_own=true` означает доверие к своему optimizer pickle, а не независимое удостоверение владельца. Предыдущие V1 receipts не подходят. Пустой `resume_config={}` начинает новый эксперимент. Пустой `training_revision` использует SHA текущего запуска; при продолжении это поле нужно заполнить SHA исходной сессии.

В worker используется один видимый GPU (`CUDA_VISIBLE_DEVICES=0`), чтобы T4×2 не менял размер batch и не включал DataParallel для квантованной модели. Заголовок Kaggle совпадает с запрошенным slug: постоянный заголовок больше не меняет идентичность kernel.

## Почему не включено без разрешения
Kaggle квоты ограничены; нет доступа к аккаунту, API token, проверенным размеченным данным и совместимым HF весам. Прямой CI не имеет GPU; штатный GitHub runner ничего не обучает. Даже после успешной GPU-сессии LoRA **не публикуется** на Android без независимого слепого экзамена, реального holdout, тестов no-lookahead и проверенного Demo исполнения.

## Безопасность
Workflow не выполняется на pull request; GitHub token имеет только contents:read, Kaggle token используется исключительно как Actions secret. Генератор `scripts/ai_private_kaggle_gpu_bridge.py` допускает только канонические Kaggle IDs и входные пути. Сгенерированный worker скачивает публичный исходный код на зафиксированной ревизии, устанавливает ML-зависимости и запускает research-only обучение QLoRA. Возможны несовместимости библиотек/версий Kaggle: настоящая CUDA приёмка ещё требуется. Никаких ключей Binance.

Официальные источники:
- Kaggle CLI kernels: https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md
- Kaggle kernel metadata: https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md
- Kaggle CLI token auth: https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/auth.md
