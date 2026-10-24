"""Four independently sourced William(s) book profiles: blind GGUF exam.

Four volumes never collapse into 'one generic five indicators' strategy.
This assessment tests rule comprehension only, not real-wave eyesight,
LoRA weights, trade fills, expectancy, safe Android or Demo execution.

Correct choices stay on the EXAMINER side, never in model prompts. Questions
and alternate distractors are source-bound to reviewed PDF rule cards. Model
responses can abstain; abstentions count as wrong.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import argparse
import json
from pathlib import Path
import random
from typing import Any

from .book_rules import registry
from .tc2_trade_lifecycle_exam import lifecycle_cases


# (case id, source rule, stage, question, correct text, two wrong alternatives).
# All questions are paraphrases and use only source-verified cards; they are
# exercises written by a developer, NOT independent gold-labelled market charts.
_CASES = [
 ("TC1-W4","TC1.W4","WAVE",
  "После сильной предполагаемой W3 осциллятор 5/34 близок к нулю на H1. Какая оценка корректна?",
  "Это один признак возможной W4 выбранного масштаба; степень и структуру требуется проверить.",
  "AO возле нуля всегда означает точно завершённую четвёртую волну.",
  "Нужно немедленно купить без подтверждения нового импульса."),
 ("TC1-W5","TC1.W5","WAVE",
  "Цена обновила экстремум после W3 и AO показывает дивергенцию. Что известно о завершении W5?",
  "Есть гипотеза истощения; нужны другие признаки, экстремумы и младшие волны.",
  "Любая дивергенция обязательно немедленно разворачивает рынок.",
  "Следует считать волновую степень подтверждённой только AO."),
 ("TC1-TARGET","TC1.MAGIC.TARGET","WAVE",
  "Для гипотетической W5 проекция зоны 62–100% построена от завершения W4. Что это значит?",
  "Исследовательская целевая область, полезная при проверенной структуре W0–W4, а не обязательная точка разворота.",
  "Точная цена выхода, которую рынок обязан достигнуть.",
  "Автоматический сигнал WM3 из Trading Chaos 2."),
 ("TC1-SQUAT","TC1.MAGIC.SQUAT","VOLUME",
  "В криптосвечах Binance есть base volume, но тикового объёма нет. Можно ли подтвердить авторский TC1 Squat?",
  "Нет, для книжного признака требуется проверить настоящий tick volume и отношение MFI к предыдущему бару.",
  "Да, base volume безусловно равен tick volume.",
  "Для Squat достаточно только зелёного столбика AO."),
 ("TC1-MAGIC","TC1.MAGIC.MOMENTUM","WAVE",
  "На предполагаемом конце W5 четыре волшебные пули наблюдаются, но Squat неизвестен; текущий 5/34 пересёк свою SMA5. Как отметить состояние?",
  "Набор признаков неполный; не объявлять пять пуль или исполненный разворот.",
  "Пять пуль точно выполнены, потому что индикатор развернулся.",
  "Изменение импульса автоматически закрывает любую позицию."),
 ("TC1-FRACTAL","TC1.FRACTAL","ENTRY",
  "Центральный фрактальный экстремум определён на текущей H1-свече. Два правых бара ещё не закрыты. Можно ли подтвердить его?",
  "Нет: минимум нужны завершённые подтверждающие правые бары, с учётом вариантов равных уровней.",
  "Да, центральный максимум — окончательный фрактал без правых свечей.",
  "Можно считать новый фрактал биржевым fill."),
 ("TC1-FEAR","TC1.PSY.FEAR","PSYCHOLOGY",
  "После двух убытков модель предлагает убрать подтверждённый SL из-за опасения нового стопа. Как учесть психологический принцип?",
  "Не отменять защиту из-за страха; соблюдать исходный риск и причины сделки.",
  "Удалить SL — это обязательное преодоление страха.",
  "Удвоить позицию после серии убытков."),
 ("TC1-PLAN","TC1.PSY.PLAN","PSYCHOLOGY",
  "Рыночная структура опровергла прежнюю волновую разметку, но модель уверена в себе. Что требуется?",
  "Сменить гипотезу и заново проверить план и защиту, не подгоняя рынок под прогноз.",
  "Усилить старую уверенность и проигнорировать новые данные.",
  "Всегда усреднить сделку на всю свободную маржу."),
 ("NTD-ENTRY","NTD.FRACTAL","ENTRY",
  "По NTD AO положительный и пасть Аллигатора открылась, но первый подходящий внешний фрактал ещё не пробит. Каков допуск?",
  "Начальный вход пока не допущен: нужен действительный пробой подходящего фрактала.",
  "Открыть сделку по одному положительному AO.",
  "Применить ранний TC2 WM1 как обязательное правило NTD."),
 ("NTD-ALLIGATOR","NTD.ALLIGATOR","STRUCTURE",
  "Аллигатор NTD смещён вперёд на графике. Как строить состояние его линии на историческом баре t?",
  "Из уже известных сглаженных значений с видимым смещением, никогда не из данных будущих свечей.",
  "Брать цену из t+8, чтобы график был красивее.",
  "Заменить все три линии средним AO."),
 ("NTD-AO","NTD.AO","INDICATOR",
  "NTD AO Saucer, Twin Peaks и Zero Cross найдены после первоначального фрактального допуска. Они равны Super AO из TC2?",
  "Нет, это собственные схемы NTD и их цены требуют отдельных триггеров.",
  "Да, все три названия означают третий зелёный бар TC2.",
  "Они являются исполненными биржевыми ордерами."),
 ("NTD-AC","NTD.AC","INDICATOR",
  "При поиске покупки по NTD AC находится ниже нуля, последовательные два зелёных столбца уже появились. Сколько растущих столбцов по правилу AC требуется?",
  "Три растущих AC столбца, когда AC ниже нулевой линии.",
  "Два при любом положении AC.",
  "Один столбец, если AO выше нуля."),
 ("NTD-ZONE","NTD.ZONE","INDICATOR",
  "На последнем закрытом баре AO вырос, а AC упал. Цвет зоны согласно NTD?",
  "Серая зона, потому что AO и AC не меняются в одном направлении.",
  "Зелёная зона, поскольку AO вырос.",
  "Красная зона, поскольку AC упал."),
 ("NTD-LINE","NTD.BALANCE","INDICATOR",
  "Пятое измерение NTD Balance Line, цена движется К линии. Есть один новый более высокий максимум после базового бара. Хватит ли для покупки?",
  "Нет: для движения к линии по правилу требуется два новых более высоких максимума.",
  "Да, один новый максимум достаточен и к линии, и от неё.",
  "Balance Line в NTD заменяется фракталом WM3."),
 ("NTD-ZONEADD","NTD.ZONE.ADDED","MANAGEMENT",
  "Подряд завершились пять зелёных зонных баров NTD. AI предлагает продолжать добавлять по зоне на шестом. Что верно?",
  "После пятого подряд зелёного зонного бара дальнейшие зонные добавления прекращаются.",
  "На шестом нужно обязательно добавить вдвое больше.",
  "Цвет зоны никак не влияет на добавления."),
 ("NTD-EXIT","NTD.EXITS","EXIT",
  "В NTD упомянут выход по закрытию относительно Teeth и биржевой защитный stop-market. Это одно и то же?",
  "Нет, stop-on-close — решение по закрытому бару, а stop-market — внутрибaровое условное исполнение.",
  "Да, оба доказаны одним касанием high или low свечи.",
  "Любой AO у нуля означает исполнение stop-market."),
 ("COURSE-W4","COURSE.W4.ZERO","WAVE",
  "На H4 предполагается четвёртая волна, но 5/34 ещё далёк от нуля. Что делать?",
  "Перепроверить степень и выбрать соразмерный TF, не объявлять W4 установленной.",
  "Растянуть рисунок до обязательной W4 независимо от AO.",
  "Закрыть сделку и объявить пятую волну подтверждённой."),
 ("COURSE-RHYTHM","COURSE.RHYTHM","WAVE",
  "В Course описывается примерная длина волновой последовательности в 100–140 баров. Что означает такой ориентир?",
  "Подбор масштаба наблюдения, но не торговый триггер и не точное постоянное число.",
  "Ордер на покупку на 100-й свече без структуры.",
  "Обязательное условие, что каждая W3 длится ровно 140 минут."),
 ("COURSE-NEST","COURSE.VISUAL.NESTED_ABC","WAVE",
  "На H1 видна меньшая коррекция ABC внутри предполагаемого тренда H4. Может ли она сама по себе доказать старшую W5?",
  "Нет, вложенность и степень требуют доступных по времени свингов и конкурентных сценариев.",
  "Да, любой ABC однозначно подтверждает W5 старшей степени.",
  "Степень волны всегда равна числу зелёных AO."),
 ("COURSE-HUMP","COURSE.HUMP","STRUCTURE",
  "На одном баре появился одиночный противоположный фрактал. Достаточно ли для авторской Hump-структуры курса?",
  "Нет, нужно проверить оба противоположных фрактала и перекрытие их баров.",
  "Да, любая свеча с тенью является Hump.",
  "Hump не зависит от двух фракталов."),
 ("COURSE-THUMB","COURSE.THUMB","STRUCTURE",
  "На сканированной странице Course OCR нашёл слово Thumb, но последовательность экстремумов плохо видна. Как подтвердить паттерн?",
  "Визуально проверить графическую структуру и риск-бары, не дописывать модель из OCR.",
  "Признать Thumb доказанным по одному слову OCR.",
  "Заменить его NTD Balance Line, потому что названия похожи."),
 ("COURSE-TP","COURSE.TRADING_PARTNER","MANAGEMENT",
  "Trading Partner изучает старший TF, 5/34/5, Hump/Thumb и уже открытую позицию. Можно ли заменить всю процедуру одним 'AO зелёный'?",
  "Нет, решение зависит от набора структурированных контекстных признаков и кампании.",
  "Да, один зелёный AO всегда даёт готовый торговый ордер.",
  "Вместо контекста нужен только показатель funding."),
 ("COURSE-AO","COURSE.AO5345","WAVE",
  "Profitunity 5/34/5 курса используется для анализа волн. Можно ли без проверки приравнять его к одному торговому AO/AC сигналу из NTD?",
  "Нет, назначение, знак и временное положение индикатора нужно учитывать отдельно.",
  "Да, AC NTD и все компоненты 5/34/5 идентичны.",
  "Это достаточное подтверждение реального биржевого fill."),
 ("COURSE-PROFILE","COURSE.TRADING_PARTNER","MANAGEMENT",
  "Курс показывает самостоятельное управление кампанией. Можно ли одновременно исполнять его и TC2 как два независимых трейдера на одной паре без координатора?",
  "Нет: схемы разных изданий сохраняются отдельно, торговую кампанию и суммарный риск координируют.",
  "Да, две нейросети могут одновременно удваивать позицию без лимитов.",
  "Любое книжное упражнение автоматически разрешает биржевую сделку."),
]


def four_book_blind_cases(*, seed: int=20261024) -> list[dict[str, Any]]:
    if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<=2**32-1:
        raise ValueError("INVALID_FOUR_BOOK_EXAM_SEED")
    rng=random.Random(seed)
    reviewed={x["id"]:x for x in registry()["rule_cards"]
              if x["status"]=="REVIEWED_PAGES"}
    tasks=[]
    for case_id,rule,stage,question,correct,bad1,bad2 in _CASES:
        card=reviewed.get(rule)
        if not card:raise ValueError("UNVERIFIED_SOURCE_RULE:"+rule)
        choices=[correct,bad1,bad2]
        rng.shuffle(choices)
        tasks.append({
            "public":{
                "id":case_id,"edition":card["source"],
                "question":question,"choices":dict(zip("ABC",choices)),
            },
            "expected":"ABC"[choices.index(correct)],
            "stage":stage,
            "source_rule_id":rule,
        })
    # Select exactly 8 scenarios across three TC2 chapters to keep all
    # four book contributions evenly weighted. The dedicated TC2 exam
    # separately covers ALL 20 trading lifecycle scenarios.
    tc2=lifecycle_cases(seed)
    keep=("WM1_WAIT","WM1_STOP","WM2_FIRST","WM3_TEETH",
          "PYRAMID","SL_WIDEN","OPPOSITE","REVERSE")
    for item in tc2:
        if any(item["public"]["id"].endswith(x) for x in keep):
            tasks.append(item)
    assert len(tasks)==32
    assert Counter(x["public"]["edition"] for x in tasks)=={
        "TC1":8,"NTD":8,"COURSE":8,"TC2":8}
    return tasks


def grade_four_book_model(model: Any, *, seed: int=20261024,
                          per_edition: int=8,
                          autonomous_source_retrieval: bool=False)->dict[str,Any]:
    if (isinstance(per_edition,bool) or not isinstance(per_edition,int)
            or not 1<=per_edition<=8):
        raise ValueError("INVALID_PER_EDITION_EXAM_SIZE")
    all_tasks=four_book_blind_cases(seed=seed)
    # Diagnostic-size exams remain balanced across books. The frozen full
    # 8-per-edition variant has no sampling and no heldout-score selection.
    tasks=[]
    for ed in ("TC1","NTD","TC2","COURSE"):
        cohort=[x for x in all_tasks if x["public"]["edition"]==ed]
        selected=[cohort[i] for i in range(8) if (i*per_edition//8)!=((i+1)*per_edition//8)]
        # Pick indices approximately evenly spaced, without consulting answers
        # or outcomes. For a 4-book, 4-per-book pilot this takes 1,3,5,7.
        tasks.extend(selected)
    if len(tasks)!=4*per_edition:
        raise RuntimeError("BALANCED_BOOK_CASE_SELECTION_FAILED")
    grades=[]
    for task in tasks:
        public=task["public"]
        try:
            if autonomous_source_retrieval:
                from .unprompted_source_application import answer_with_self_retrieval
                answer=answer_with_self_retrieval(model,public)
                letter=answer.get("selected")
            else:
                answer=model.answer_exam_case(public)
                letter=answer.get("choice") if isinstance(answer,dict) else None
            letter=letter if letter in ("A","B","C") else ""
            failure=None if letter else "ABSTENTION_OR_MALFORMED_MODEL_REPLY"
        except (RuntimeError,ValueError,KeyError,TypeError):
            letter="";failure="MODEL_CALL_ERROR"
        grades.append({
            "id":public["id"],"edition":public["edition"],
            "stage":task["stage"],"source_rule_id":task["source_rule_id"],
            "chosen":letter,"correct":letter==task["expected"],
            "abstained":letter=="","failure":failure,
        })
    scores={}
    for book in ("TC1","NTD","TC2","COURSE"):
        section=[row for row in grades if row["edition"]==book]
        scores[book]={
            "correct":sum(x["correct"] for x in section),
            "total":len(section),
            "abstentions_counted_wrong":sum(x["abstained"] for x in section),
        }
    correct=sum(x["correct"] for x in grades)
    baseline=max(Counter(x["expected"] for x in tasks).values())
    manifest=sha256(json.dumps(tasks,sort_keys=True,ensure_ascii=False,
                               separators=(",",":")).encode()).hexdigest()
    return {
        "mode":"FOUR_WILLIAMS_BOOKS_BLIND_LOCAL_GGUF_RULE_APPLICATION_EXAM",
        "protocol_sha256":manifest,"seed":seed,"total":len(tasks),
        "per_edition":per_edition,
        "autonomous_question_only_source_retrieval":autonomous_source_retrieval,
        "no_human_selected_rule":autonomous_source_retrieval,
        "full_32_case_exam_completed":per_edition==8,
        "correct":correct,"accuracy_all_including_abstentions":correct/len(tasks),
        "by_edition":scores,
        "majority_letter_baseline":baseline/len(tasks),
        "research_pass_not_book_mastery":(
            correct/len(tasks)>=0.85 and correct>baseline
            and all(x["correct"]/x["total"]>=0.75 for x in scores.values())
        ),
        "results":grades,
        "book_pdf_text_exported":False,
        "reviewed_book_cards_not_sufficient_for_four_book_mastery":True,
        "synthetic_abstract_cases_not_real_trading_charts":True,
        "model_weights_trained_here":False,
        "expert_wave_validation_completed":False,
        "trade_action":None,"execution_allowed":False,
    }


def main()->None:
    p=argparse.ArgumentParser(description="Actual four-book GGUF knowledge exam, no orders")
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--expected-sha256",required=True)
    p.add_argument("--ctx",type=int,default=8192)
    p.add_argument("--seed",type=int,default=20261024)
    p.add_argument("--per-edition",type=int,choices=range(1,9),default=8)
    p.add_argument("--reasoning-budget",type=int,default=1024)
    p.add_argument("--autonomous-source-retrieval",action="store_true",
                   help="Auto lookup reviewed book cards from question text only")
    args=p.parse_args()
    from .real_gguf_acceptance import actual_gguf_sha256
    from .local_model import LocalWilliamsModel
    actual_gguf_sha256(args.model,args.expected_sha256)
    model=LocalWilliamsModel(args.model,n_ctx=args.ctx,gpu_layers=0,
                             learning_mode="auto",
                             reasoning_budget=args.reasoning_budget)
    print(json.dumps(grade_four_book_model(
        model,seed=args.seed,per_edition=args.per_edition,
        autonomous_source_retrieval=args.autonomous_source_retrieval),
                     ensure_ascii=False,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
