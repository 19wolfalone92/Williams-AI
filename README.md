# Williams AI 🧠 — локальный ИИ с добровольным интернет-поиском и Binance DEMO

Отдельная исследовательская копия Williams Bot. Исходный торговый проект НЕ изменяется.

## Создано
- Python: локальный GGUF runtime (llama-cpp-python), анализ JSON снимков, no-lookahead.
- Знания: локальный поиск по заметкам Markdown/text, без облачной БД.
- Android: отдельное приложение com.williamsai.offline, выбор GGUF через Android file picker, атомарное копирование модели по SHA-256.
- Нативный Android backend из фиксированного коммита официального llama.cpp. Сборка проходит в GitHub Actions.
- В локальном AI нет вызовов OpenAI API, а Android manifest не запрашивает INTERNET.

## Как использовать
[Инструкция по Android](mobile/README_RU.md).
[Сборки APK](https://github.com/19wolfalone92/Williams-AI/actions/workflows/ai-android-apk.yml).
[Полная архитектура](docs/WILLIAMS_AI_ARCHITECTURE.md).
[Черновой PR #1](https://github.com/19wolfalone92/Williams-AI/pull/1).

Весов нейросети внутри APK НЕТ. Получите лицензионный совместимый GGUF отдельно, проверьте контрольную сумму и выберите файл в приложении. До аппаратной приёмки APK является экспериментальным.

На компьютере установить llama-cpp-python и выполнить:

    python -m williams_ai.cli examples/ai_market_snapshot.json
    python -m williams_ai.cli examples/ai_market_snapshot.json --local-model models/williams-local.gguf --question "Что происходит на H1?"

Пока ИИ не получает живые свечи, не доказана точность книжного анализа и не разрешены торговые команды. GitHub хранит код и собирает APK, а нейросеть выполняется на устройстве.

## Новая исследовательская ветка

- **Интернет-исследования в Android:** добровольный переключатель, поиск по Wikipedia и Google News, ссылки; без API-ключа облачного ИИ. Подробнее: [web research + demo](docs/WILLIAMS_AI_RESEARCH_DEMO_RU.md).
- **Python DEMO Futures:** отдельный исполнитель только для Binance Futures Demo, обязательный стоп и два явных подтверждения. **Он не подключён напрямую к Android-чату.** По умолчанию ордера запрещены. Примеры: [документация](docs/WILLIAMS_AI_RESEARCH_DEMO_RU.md).
- **Production-торговля не поддерживается**. Демо сделки и сеть требуют проверки на устройстве/демо-счёте.
