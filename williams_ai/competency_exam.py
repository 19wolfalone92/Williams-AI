"""A small, source-scoped reasoning EXAM for a local GGUF Williams AI.

Curated questions are not a complete or blind evaluation of four books. The
answer key never enters the model prompt. All outputs are non-executable.
"""
from __future__ import annotations

from typing import Any, Callable

CASES: tuple[dict[str, Any], ...] = (
    {"id":"TC1-WAVE-UNCERTAINTY", "edition":"TC1", "pdf_pages":[53,54],
     "question":"После снижения AO вернулся почти к нулю, но волновые экстремумы и масштаб H4 ещё не проверены. Что можно заключить?",
     "choices":{"A":"Четвёртая волна доказана; немедленно открыть LONG.",
                "B":"Есть гипотеза о коррекции, требуется проверка структуры и масштаба.",
                "C":"Нужно безусловно закрыть любую позицию."},
     "correct":"B"},
    {"id":"TC1-NESTED-DEGREE", "edition":"TC1", "pdf_pages":[53,54],
     "question":"На H1 похожа волна 5, а на H4 возможен незавершённый импульс. Какой вывод корректен?",
     "choices":{"A":"H1 всегда старше H4.",
                "B":"AO однозначно доказал старший волновой уровень.",
                "C":"Сохранить альтернативные гипотезы, проверить иерархию и предыдущую историю."},
     "correct":"C"},
    {"id":"NTD-FIRST-ENTRY", "edition":"NTD", "pdf_pages":[41,42,125,126],
     "question":"Аллигатор просыпается, AO положителен, но первый допустимый фрактал за пастью ещё не пробит. Что делать согласно профилю NTD?",
     "choices":{"A":"Открыть позицию по одному AO.", "B":"Ждать фактического преодоления первого подходящего фрактала.", "C":"Открыть встречный SHORT."},
     "correct":"B"},
    {"id":"NTD-ZONE-COLOR", "edition":"NTD", "pdf_pages":[91,92],
     "question":"На последнем закрытом баре AO вырос относительно прошлого бара и AC тоже вырос. Цвет зоны NTD?",
     "choices":{"A":"Зелёная.", "B":"Красная.", "C":"Серая."},
     "correct":"A"},
    {"id":"TC2-WM1-TRIGGER", "edition":"TC2", "pdf_pages":[121,122,124,125],
     "question":"Появился подходящий разворотный бар WM1 и предполагаемая ангуляция, но на следующих закрытых барах экстремум входа не пересекался. Что известно?",
     "choices":{"A":"Уже исполнена сделка.", "B":"Есть гипотеза WM1, ждём ценовой триггер и фактическое исполнение.", "C":"Разворот требует продажи немедленно."},
     "correct":"B"},
    {"id":"TC2-WM3-TEETH", "edition":"TC2", "pdf_pages":[147,148,149,150,151],
     "question":"Фрактал продажи подтверждён ниже Teeth при формировании. Позднее цена пересекла уровень продажи, но на момент касания цена была ВЫШЕ Teeth. Можно ли считать этот пробой допустимым?",
     "choices":{"A":"Да, потому что при формировании фрактал был ниже Teeth.",
                "B":"Нет, фильтр по Teeth должен пройти в момент срабатывания.",
                "C":"Всегда да — Teeth не влияет на WM3."},
     "correct":"B"},
    {"id":"TC2-OPPOSITE-NOT-EXIT", "edition":"TC2", "pdf_pages":[131,153,154],
     "question":"Длинная кампания с подтверждённым SL. Появился противоположный Мудрец, но ни уровень разворота, ни SL не исполнены биржей. Правильная оценка?",
     "choices":{"A":"Само наблюдение автоматически означает закрытие LONG.",
                "B":"Нельзя утверждать выход без исполненного триггера/стопа и проверенной причины.",
                "C":"Нужно убрать защитный стоп."},
     "correct":"B"},
    {"id":"TC2-SL-MISSING", "edition":"TC2", "pdf_pages":[131,156,157],
     "question":"Биржа подтвердила вход, но отвергла защитный SL. ИИ обнаружил новый WM2. Что главное?",
     "choices":{"A":"Сосредоточиться на добавлении по WM2.", "B":"Увеличить плечо.",
                "C":"Зафиксировать критическое отсутствие биржевой защиты; никакой сигнал не отменяет риск."},
     "correct":"C"},
    {"id":"COURSE-SCAN-AMBIGUOUS", "edition":"COURSE", "pdf_pages":[77,89,91],
     "question":"В сканированном Course OCR распознал фрагмент графика неоднозначно. Можно ли объявить точный визуальный сигнал доказанным только по OCR?",
     "choices":{"A":"Да, текст автоматически подтверждает рисунок.",
                "B":"Нет, нужно проверить саму схему/страницу и контекст.",
                "C":"Подменить её правилом из TC2 без пометки."},
     "correct":"B"},
    {"id":"COURSE-EDITION-BOUNDARIES", "edition":"COURSE", "pdf_pages":[89,90,91,92],
     "question":"Course обсуждает этапы кампании и Trading Partner, а TC2 — Трёх Мудрецов. Как строить проверяемую стратегию?",
     "choices":{"A":"Перемешать правила без указания издания.",
                "B":"Сохранить разные источники/профили и документировать переходы.",
                "C":"Не использовать исторические графики."},
     "correct":"B"},
)


def public_question(case: dict[str, Any]) -> dict[str, Any]:
    """Only fields safe to send to the model; never send the answer key."""
    return {name: case[name] for name in ("id","edition","question","choices")}


def safe_exam_answer(request: Callable[[], Any]) -> dict[str, Any]:
    """Separate a model runtime/format failure from a conceptual book mistake."""
    try:
        response=request()
    except (ValueError,RuntimeError,TypeError,KeyError) as exc:
        return {"choice":None,"reason":"","answer_error":type(exc).__name__}
    if (not isinstance(response,dict) or set(response)!={"choice","reason"}
            or not isinstance(response["choice"],str) or response["choice"] not in {"A","B","C"}
            or not isinstance(response["reason"],str) or not response["reason"].strip()):
        return {"choice":None,"reason":"","answer_error":"MODEL_MALFORMED_ANSWER"}
    return {"choice":response["choice"],"reason":" ".join(response["reason"].split())[:500],
            "answer_error":None}


def grade_exam(
    answer: Callable[[dict[str, Any]], dict[str, str]],
    cases: tuple[dict[str, Any], ...] = CASES,
) -> dict[str, Any]:
    if not callable(answer):
        raise TypeError("Answer adapter must be callable")
    graded: list[dict[str, Any]] = []
    for case in cases:
        response = safe_exam_answer(lambda:answer(public_question(case)))
        choice = response.get("choice")
        if choice not in case["choices"]:
            choice = None
        why = response.get("reason", "")
        reason = (" ".join(why.split())[:500] if isinstance(why, str) else "")
        graded.append({
            "id":case["id"],"edition":case["edition"],"pdf_pages":case["pdf_pages"],
            "chosen":choice,"correct":choice == case["correct"],
            "answer_error":response["answer_error"],
            "reason_for_human_review":reason,
        })
    n = len(graded)
    correct = sum(item["correct"] for item in graded)
    return {
        "mode":"READ_ONLY","trade_action":None,"execution_allowed":False,
        "total_cases":n,"correct_cases":correct,
        "failed_model_calls":sum(x["answer_error"] is not None for x in graded),
        "score_on_curated_cases_percent":round(correct*100/n,1) if n else None,
        "book_comprehension_percent":None,"books_retrained":False,
        "results":graded,
        "limitations":[
            "This is a non-blind, small, hand-curated multiple-choice benchmark.",
            "A correct choice does not prove correct reasoning, book mastery or trading profitability.",
            "Review reasons, illustrations and hold-out scenarios independently.",
        ]
    }
