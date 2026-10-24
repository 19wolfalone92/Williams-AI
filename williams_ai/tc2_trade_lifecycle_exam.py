"""Independent blind TC2 trade-lifecycle exam for actual local Qwen.

No answers or future OHLC are provided to the model; source citations remain
outside the test prompt. Synthetic rule-comprehension exercises, NOT empirical
trading proof. The distribution intentionally includes opportunities to WAIT,
HOLD, and refuse risky scaling, not merely 'buy/sell' pattern matching.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from typing import Any

# id, learning target, chapter, scenario, correct answer, two tempting errors.
_EXAM = [
 ("WM1_WAIT","ENTRY","TC2.WM1.TRIGGER",
  "Закрытый H1-бар дал бычий бар разворота с подтверждённой визуально ангуляцией, AO ещё падает, но следующая цена НЕ пробила high сигнала. Каков статус входа?",
  "Поставить учебный ценовой триггер чуть выше high и ждать пробоя; исполнение неизвестно.",
  "Считать вход уже исполненным по закрытию сигнальной свечи.",
  "Открыть короткую позицию, потому что AO пока красный."),
 ("WM1_STOP","ENTRY","TC2.WM1.STOP",
  "Достоверно сработал buy-trigger первого мудреца у разворотного бара. Куда книга помещает первый защитный стоп?",
  "За противоположный low сигнального разворотного бара.",
  "Всегда на фиксированные 10 процентов ниже входа.",
  "На максимум того же бара, где был buy-trigger."),
 ("WM1_ANGLE","ENTRY","TC2.WM1.ANG",
  "Форма бычьего бара похожа на WM1, но независимая проверка ангуляции с Аллигатором не сделана. Что правильно?",
  "Оставить WM1 неподтверждённым, запросить проверку расходящегося угла.",
  "Считать цвет свечи достаточным доказательством WM1.",
  "Заменить проверку угла зелёным цветом AO."),
 ("WM2_WAIT","ENTRY","TC2.WM2",
  "На H1 наблюдаются три последовательных зелёных изменения AO, но цена не пересекла максимум третьего сигнального бара. Что это?",
  "Наблюдаемый Super AO, но вход по цене ещё не сработал.",
  "Исполненный покупающий ордер по одному AO.",
  "Готовое подтверждение третьего мудреца без фрактала."),
 ("WM2_FIRST","ENTRY","TC2.WM2",
  "На рынке ещё нет WM1, но первым возник действительный Super AO и потом цена пробила соответствующий сигнальный уровень. Может ли он стать первым сигналом кампании TC2?",
  "Да, WM2 может быть первым сигналом TC2, если выполнены все условия.",
  "Нет, обязательный первый сигнал всегда WM1.",
  "Только после появления NTD AO Saucer."),
 ("WM3_TEETH","ENTRY","TC2.WM3",
  "Пятибарный фрактал завершился с задержкой t+2, но buy-пробой произошёл ниже красных зубов Аллигатора. Как поступить?",
  "Не засчитывать такой buy-пробой как действительный WM3.",
  "Купить на любой пробой фрактала, положение зубов неважно.",
  "Считать сигнал исполненным в центральной свече фрактала."),
 ("WM3_WAIT","ENTRY","TC2.WM3",
  "Есть подтверждённый фрактал с двумя закрытыми барами справа и подходящими зубами Аллигатора; будущая цена ещё не пробила его вершину. Каков статус?",
  "Подтверждённая формация, ожидающая отдельного ценового пробоя.",
  "Исполненный ордер на фрактальном максимуме.",
  "Пятибарная формация означает закрытие текущей сделки."),
 ("WM3_SUPERSEDE","ENTRY","TC2.WM3",
  "До пробоя старого фрактального уровня появился новый более свежий фрактал в ту же сторону. Как объяснить статус?",
  "Новый фрактальный сигнал может заменить прежний неисполненный.",
  "Все старые фракталы остаются вечными торговыми приказами.",
  "Новый фрактал запрещает любые сделки на этом рынке навсегда."),
 ("PYRAMID","MANAGEMENT","TC2.CAMPAIGN",
  "В книге показана обратная пирамида 1-5-4-3-2 контрактов. Как её трактовать применительно к AI и риск-движку?",
  "Как иллюстрацию увеличения прибыльной кампании; объёмы должны пройти независимые лимиты риска.",
  "Как обязательный размер каждой реальной сделки на любой бирже.",
  "Как разрешение усреднять убыток ради средней цены."),
 ("ADD_RED","MANAGEMENT","TC2.CAMPAIGN",
  "Первая позиция убыточная после комиссий. Появился новый индикаторный признак того же направления, но пробоя цены нет. Что допустимо?",
  "Не увеличивать убыточную позицию по одному признаку; ждать фактического подтверждения.",
  "Сразу добавить пять контрактов по книжной пирамиде.",
  "Снять существующий стоп для большей свободы движения."),
 ("SL_WIDEN","MANAGEMENT","TC2.CAMPAIGN",
  "Длинная позиция уже имеет подтверждённый защитный SL на 100. Новый пятибарный swing даёт учебный кандидат SL на 98. Что делать?",
  "Не расширять стоп вниз; сохранить текущую защиту до обоснованного ужесточения.",
  "Передвинуть stop ниже на 98 ради большего убытка.",
  "Удалить stop, если AO зелёный."),
 ("TRAIL","MANAGEMENT","TC2.CAMPAIGN",
  "Кампания LONG прибыльная, биржевой стоп подтверждён; за пятью уже закрытыми свечами появился структурный кандидат SL выше существующего и ниже рынка. Что изучать?",
  "Проверку возможности только подтянуть SL без расширения риска.",
  "Автоматическую отмену SL при любом откате AO.",
  "Немедленный выход при отсутствии нового сигнала мудреца."),
 ("AO_NOT_FILL","MANAGEMENT","TC2.WM2",
  "К сигналу WM2 добавления цена внутри бара дошла до уровня, но нет записи биржи об исполнении. Как учесть объём и среднюю цену?",
  "Не считать биржевое исполнение доказанным по OHLC; нужна сверка ордеров.",
  "Прибавить весь объём к фактической позиции автоматически.",
  "Считать любое пересечение AO доказательством исполнения."),
 ("CORRECTION","MANAGEMENT","TC2.CAMPAIGN",
  "Младший ТФ показывает возможную коррекцию к AO около нуля, но старшая волновая степень не подтверждена. Что делать с существующей кампанией?",
  "Оставить степень волны гипотезой, контролировать SL и структурные подтверждения.",
  "Закрыть всё, потому что любое AO около нуля означает W4.",
  "Разрешить удвоение позиции без ограничения риска."),
 ("OPPOSITE","EXIT","TC2.REVERSE",
  "Открыт защищённый LONG. На H1 только что появился противоположный WM1-кандидат, но его sell-ценовой триггер не исполнен. Нужно ли закрыть LONG?",
  "Нет автоматического выхода по наблюдению: нужен исполнившийся выходной уровень либо самостоятельный риск/структурный повод.",
  "Сразу закрыть LONG по рынку только за смену направления мудреца.",
  "Развернуть в SHORT ещё до пробоя ценового уровня."),
 ("REVERSE","EXIT","TC2.REVERSE",
  "По допустимой TC2-схеме появился и действительно исполнен встречный stop-reverse, закрывающий LONG и открывающий SHORT. Как это интерпретировать?",
  "Как подтверждённый разворот после сверки фактических биржевых исполнений.",
  "Как исполненный разворот ещё на стадии наблюдения нового мудреца.",
  "Как причину игнорировать обновлённые защитные стопы."),
 ("UNPROTECTED","EXIT","TC2.WM1.STOP",
  "На Demo подтверждён вход, но защитный SL отсутствует и биржа не принимает его установку. Что должно превалировать?",
  "Аварийное управление незащищённой позицией и блокировка новых входов.",
  "Ожидать любого нового сигнала WM2 перед защитой.",
  "Считать устное обещание AI равным подтверждению стопа."),
 ("RESTART","EXIT","TC2.CAMPAIGN",
  "После тайм-аута ответа Binance неизвестно, исполнился ли встречный ордер. Что требуется до нового выхода или входа?",
  "Восстановить биржевую позицию, стопы и fill-идентификаторы; не дублировать MARKET вслепую.",
  "Повторять MARKET до положительного HTTP-ответа.",
  "Удалить журнал и предположить, что позиции нет."),
 ("AO_DIVERGENCE","EXIT","TC2.CAMPAIGN",
  "Приблизилась возможная пятая волна, AO показывает дивергенцию, но структурный выход и биржевой SL пока не сработали. Что говорить?",
  "Предупредить об исчерпании импульса; не выдавать гипотезу за биржевое закрытие.",
  "Закрыть любую позицию автоматически по факту дивергенции.",
  "Объявить пятую волну точно известной и отменить SL."),
 ("CANDLE_AMBIGUITY","EXIT","TC2.CAMPAIGN",
  "На одной исторической OHLC-свече одновременно задеты цена пробоя и уровень защитного стопа. Внутрибaрового порядка исполнения нет. Как учесть backtest?",
  "Признать неоднозначность исполнения; использовать консервативный исход и не выдавать прибыль за подтверждённый fill.",
  "Предположить, что прибыльный пробой точно был первым.",
  "Считать любую такую свечу стопроцентно прибыльной сделкой."),
]


def lifecycle_cases(seed: int = 20261024) -> list[dict[str, Any]]:
    """Only the public 3-way question reaches the local model."""
    if not isinstance(seed,int) or isinstance(seed,bool) or not 0<=seed<=2**32-1:
        raise ValueError("INVALID_LIFECYCLE_EXAM_SEED")
    rng=random.Random(seed)
    out=[]
    for id_,stage,rule,question,good,bad1,bad2 in _EXAM:
        answer_options=[good,bad1,bad2]
        rng.shuffle(answer_options)
        public={"id":"TC2-LIFECYCLE-"+id_,"edition":"TC2",
                "question":question,
                "choices":dict(zip("ABC",answer_options))}
        out.append({"public":public,"expected":"ABC"[answer_options.index(good)],
                    "stage":stage,"source_rule_id":rule})
    return out


def grade_lifecycle_under_real_gguf(model: Any, *, seed: int=20261024) -> dict[str, Any]:
    cases=lifecycle_cases(seed)
    grades=[]
    for row in cases:
        try:
            reply=model.answer_exam_case(row["public"])
            chosen=reply.get("choice") if isinstance(reply,dict) else None
            chosen=chosen if chosen in ("A","B","C") else ""
            failure=None if chosen else "MALFORMED_OR_ABSTENTION"
        except (RuntimeError,ValueError,TypeError,KeyError):
            chosen="";failure="GGUF_EXAM_FAILED"
        grades.append({"id":row["public"]["id"],"stage":row["stage"],
                       "rule_id":row["source_rule_id"],
                       "correct":chosen==row["expected"],
                       "abstained":not chosen,"chosen":chosen,
                       # The examiner keeps expected answers out of model input.
                       "expected":row["expected"],"failure":failure})
    counts={}
    for stage in ("ENTRY","MANAGEMENT","EXIT"):
        matched=[x for x in grades if x["stage"]==stage]
        counts[stage]={"correct":sum(x["correct"] for x in matched),
                       "total":len(matched),
                       "abstentions_as_wrong":sum(x["abstained"] for x in matched)}
    correct=sum(x["correct"] for x in grades)
    guesses=Counter(x["chosen"] for x in grades if x["chosen"])
    majority=max(Counter(x["expected"] for x in grades).values())
    payload=json.dumps(cases,ensure_ascii=False,sort_keys=True,
                       separators=(",",":")).encode("utf-8")
    return {
        "mode":"INDEPENDENT_TC2_BLIND_GGUF_TRADE_LIFECYCLE_EXAM",
        "source_edition":"TC2",
        "total":len(grades),"correct":correct,
        "accuracy_all_including_abstention":correct/len(grades),
        "stages":counts,
        "majority_letter_baseline":majority/len(grades),
        "research_only_book_cognition_pass":(
            correct/len(grades)>=0.85
            and all(x["correct"]/x["total"]>=0.75 for x in counts.values())
            and correct>majority
        ),
        "questions_sha256":hashlib.sha256(payload).hexdigest(),
        "results":grades,
        "synthetic_cases_not_actual_exchange_or_historical_trades":True,
        "original_book_pages_not_sent_to_model":True,
        "actual_Qwen_weights_trained_here":False,
        "execution_allowed":False,"trade_action":None,
    }


def main() -> None:
    p=argparse.ArgumentParser(description="Real local GGUF TC2 entrance/campaign/exit exam, no orders")
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--expected-sha256",required=True)
    p.add_argument("--ctx",type=int,default=8192)
    args=p.parse_args()
    from .real_gguf_acceptance import actual_gguf_sha256
    from .local_model import LocalWilliamsModel
    actual_gguf_sha256(args.model,args.expected_sha256)
    model=LocalWilliamsModel(args.model,n_ctx=args.ctx,gpu_layers=0,
                             learning_mode="auto")
    print(json.dumps(grade_lifecycle_under_real_gguf(model),ensure_ascii=False,
                     indent=2,sort_keys=True))


if __name__=="__main__":
    main()
