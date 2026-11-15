# RULE_CHECK_025 — Profitunity Course: Hump / Five-Fingered Boogie

**Источник:** Trading Chaos Course, PDF SHA-256 fc57f707378087bfc5146db5e242c51ce8135e0b359f0c607305b364e0decfca, PDF page 72, Level II. Схема Hump визуально проверена.

## Формирование Hump
- Course называет Hump “Five Fingered Boogie” и связывает его с фрактальным поведением.
- Hump возникает, когда противоположные по направлению фракталы формируются так, что их экстремальные формационные бары не перекрываются.
- Первый фрактал называется Hump Start, второй — Hump Signal. Движение между ними рассматривается как волна Эллиота некоторой степени.
- Для LONG стоповый триггер расположен выше HIGH Hump Signal на X тиков; для SHORT — ниже LOW на X тиков. Значение X зависит от платформенного/торгового контекста; не придумывать фиксированный тик там, где его не задаёт этот фрагмент Course.

## Машинная последовательность
HUMP_START_FRACTAL → OPPOSITE_HUMP_SIGNAL_FRACTAL → VERIFY_EXTREME_BARS_DO_NOT_OVERLAP → STOP_TRIGGER_PENDING → PRICE_TRIGGERED → BROKER_FILL_CONFIRMED.
- Hump требует пары противоположных фракталов, один фрактал сам по себе Hump не создаёт.
- Подтверждающие правые бары должны закрыться до фиксации фрактала; никакого look-ahead.
- Не путать Hump с TC2 WM3 (фильтр Teeth в момент активации) и с TC1 Fractal Start/Signal.

**Статус:** source diagram spot-check; независимые метки отсутствуют.
