"""Focused numeric counterexamples after the first revealed 56-case exam.

The previous 56 are development diagnostics. These authored exercises use new
seeds and separate phrasings; they are not expert-labeled market charts.
"""
from __future__ import annotations

from collections import Counter
from decimal import Decimal, ROUND_FLOOR
import json
import math
import random

from .contrastive_book_practice import RULES, question_rows, questions_digest

INITIAL_SHA = "bf8580fbb74d4ae8b041332146e64408f06900999837826d06ef93b4a6404891"
REFERENCE416_SHA = "3b3030afe4c19fdfd533013309f87e5b544f92ab276e03bfa9ce8a54d8f1c6ba"
PREVIOUS_RECEIPT_SHA = "93ab7953798c59e8c3ba4872ca4290ef39d046f59da390b220f47afa036b16e4"
RUN_SLUG = "williams-ai-focused-2t4-20261027"
CONTROL_URL = "https://raw.githubusercontent.com/19wolfalone92/Williams-AI/research/private-four-book-deep-training-20261027/experiments/focused-training-control.json"


def number(value):
    return format(Decimal(value), "f")


def focused_cases(split: str) -> list[dict]:
    if split not in {"train", "validation", "test"}:
        raise ValueError("EXPLICIT_FOCUSED_SPLIT_REQUIRED")
    seed = {"train": 87210471, "validation": 98321753, "test": 19432861}[split]
    rng = random.Random(seed)
    count = 192 if split == "train" else 6
    forms = {"train": (0, 1, 2), "validation": (3, 4), "test": (5, 6)}[split]
    rows = []
    used_hump_ends = set()

    def add(i, variant, edition, rule, page, text, expected, explanation):
        expected = {**expected, "execution_allowed": False}
        types = ", ".join(k + ": " + ("true/false" if type(v) is bool else "число" if type(v) in (int,float) else "строка") for k,v in expected.items())
        group = "-"+str(variant).rsplit("-",1)[0] if rule=="NTD.AC" else ""
        rows.append({"case_id": f"focused-{split}-{i}-{rule}-{variant}",
            "pair_id": f"focused-{split}-{i}-{rule}{group}", "edition": edition,
            "rule_id": rule, "pdf_page": page,
            "question": text + " Только один JSON: " + types + ". execution_allowed=false. Других полей и пояснений не добавляй.",
            "answer": json.dumps(expected,ensure_ascii=False,separators=(",",":")),
            "teaching_explanation": explanation})

    for i in range(count):
        form = forms[i % len(forms)]
        digits = 2 + i % 5
        end = rng.randrange(10**(digits-1), 10**digits)
        if split == "train" and i < 12:
            end = (9,10,99,100,999,1000,9999,10000,99999,100000,999999,1000000)[i]
        while (form,end) in used_hump_ends:
            end = rng.randrange(10**(digits-1),10**digits)
        used_hump_ends.add((form,end))
        gap = rng.randrange(1,10)
        equity = Decimal(rng.randrange(500,25001))
        fraction = Decimal(rng.choice(("0.0025","0.004","0.005","0.006","0.007","0.008","0.01")))
        budget = equity * fraction
        step = Decimal(rng.choice(("0.01","0.02","0.03","0.05","0.1")))
        distance = Decimal(rng.choice(("0.3","0.5","0.6","1.2","1.5","2.4","2.6","3.5","5.2","7.5")))
        for v in (0,1):
            begin = end + gap if v else end - (0 if i % 3 == 0 else gap)
            hump_text = (
                f"Course PDF72: первая крайняя формация заканчивается баром {end}, вторая начинается баром {begin}. Границы включены.",
                f"Две крайние формации: последняя свеча первой имеет индекс {end}; первая свеча второй — индекс {begin}. Обе входят в формации.",
                f"Проверь Hump: конец первого интервала={end}, начало второго={begin}; оба конца включены.",
                f"Для Hump крайний бар более ранней формации A={end}; крайний начальный бар формации B={begin}. Учитывай оба граничных бара.",
                f"Разрешён ли Hump без перекрытия при inclusive-конце первой группы {end} и inclusive-начале второй группы {begin}?",
                f"Последний включённый в первую формацию номер бара — {end}. Вторую формацию начинают с включённого бара {begin}. Проверь отсутствие пересечения для Hump.",
                f"Course: интервал A до бара {end} включительно; интервал B от бара {begin} включительно. Нужна проверка hump_valid.",
            )[form] + " Противоположные фракталы и остальные условия подтверждены. Оцени только неперекрытие."
            difference = begin-end
            explanation = f"Сравниваю начало второй формации {begin} с концом первой {end}. Разность {begin}-{end}={difference}. "
            explanation += ("Начало строго позже конца, общих баров нет; hump_valid=true."
                            if difference>0 else "Начало не позже конца. При включённых границах есть общий бар; hump_valid=false.")
            add(i,v,"COURSE","COURSE.HUMP",72,hump_text,{"hump_valid":difference>0},explanation)

            d = distance*(v+1)
            lots = (budget/d/step).to_integral_value(rounding=ROUND_FLOOR)
            quantity = lots*step
            risk_text = (
                f"Equity={number(equity)} USDT; риск={number(fraction)} доли; entry-stop={number(d)}; шаг количества={number(step)}.",
                f"На счёте {number(equity)} USDT. Доля риска {number(fraction)}. Расстояние до стопа {number(d)}, шаг объёма {number(step)}.",
                f"Расчёт размера: капитал={number(equity)}, risk_fraction={number(fraction)}, stop_distance={number(d)}, quantity_step={number(step)}.",
                f"Учебный план риска: equity {number(equity)}, fraction {number(fraction)}, расстояние entry–stop {number(d)}, разрешённый шаг {number(step)}.",
                f"Рассчитай объём при капитале {number(equity)}, доле риска {number(fraction)}, стоп-дистанции {number(d)} и лоте с шагом {number(step)}.",
                f"Условия: капитал {number(equity)} USDT; выделяемая доля {number(fraction)}; расстояние между входом и стопом {number(d)}; квант количества {number(step)}.",
                f"Для учебного расчёта доступны equity={number(equity)}, доля={number(fraction)}, distance={number(d)}, step={number(step)}.",
            )[form] + " Стоимость пункта 1 USDT, издержки здесь исключены. Нужны risk_budget и quantity; количество округляй вниз к шагу."
            explanation = (f"Бюджет={number(equity)}*{number(fraction)}={number(budget)}. Один шаг количества рискует {number(d*step)}. "
                f"Число целых шагов=floor({number(budget)}/{number(d*step)})={number(lots)}. Количество={number(lots)}*{number(step)}={number(quantity)}. "
                f"Проверка: риск {number(quantity*d)} не больше бюджета {number(budget)}, а следующий шаг дал бы {number((quantity+step)*d)}. Нельзя округлять вверх или до ближайшего.")
            add(i,v,"TC1","TC1.PSY.PLAN",83,risk_text,{"risk_budget":float(budget),"quantity":float(quantity)},explanation)

        if i < (64 if split == "train" else 2):
            for sign in (-1,1):
                ac = Decimal(sign)*Decimal(rng.randrange(1,99999))/100000
                required = 3 if sign<0 else 2
                for position,green in enumerate((required-1,required,required+1)):
                    ac_text = (
                        f"NTD, покупка по AC: AC={number(ac)}, подряд зелёных={green}, до них красный.",
                        f"Для BUY по NTD AC имеет значение {number(ac)}. После красного появилось {green} зелёных столбцов подряд.",
                        f"AC-buy: осциллятор {number(ac)}; зелёная серия длиной {green} после красного столбца.",
                        f"В задаче о покупке AC равен {number(ac)}; завершённая последовательность зелёных имеет длину {green}, перед ней красный.",
                        f"Проверь готовность покупки NTD: серия {green} зелёных после красного, значение AC {number(ac)}.",
                        f"NTD: сейчас AC={number(ac)}. Последние {green} закрытых столбцов зелёные и следуют за красным. Сколько зелёных требуется для покупки?",
                        f"Задача BUY по AC: текущая величина {number(ac)}, зелёных подряд {green}, предыдущий до серии красный. Определи порог и готовность.",
                    )[form] + " AO=0.8. Нужны required_green и signal_ready; знак AO не заменяет знак AC."
                    explanation = (f"AC {number(ac)} {'ниже' if sign<0 else 'выше'} нуля: для покупки нужно {required} зелёных. "
                        f"Сравнение серии: {green} >= {required} — {'верно' if green>=required else 'неверно'}. "
                        f"required_green={required}, signal_ready={str(green>=required).lower()}. Цвет/значение AO не меняет этот порог.")
                    add(i,f"{sign}-{position}","NTD","NTD.AC",75,ac_text,{"required_green":required,"signal_ready":green>=required},explanation)

        if i >= (64 if split == "train" else 6):
            continue
        tick=Decimal(rng.choice(("0.01","0.05","0.1","0.25","0.5")))
        low=Decimal(rng.randrange(100,30001))*tick
        high=low+Decimal(rng.randrange(2,250))*tick
        hist=Decimal(rng.randrange(-2000,2001))/100
        delta=Decimal(rng.randrange(1,100))/100
        sign=rng.choice((-1,1))
        before=Decimal(sign)*Decimal(rng.randrange(1,1000))/100
        fractal_high=rng.randrange(100,10000)
        for v in (0,1):
            long=v==0
            wm1_intro=("TC2: закрытый","Закрытая сигнальная свеча TC2:","Условия WM1:","План TC2 по закрытой свече:","Для уже подтверждённого WM1:","В учебной ситуации TC2 имеется закрытый","Дано событие TC2:")[form]
            text=f"{wm1_intro} {'бычий' if long else 'медвежий'} WM1; ангуляция подтверждена. Low={number(low)}, high={number(high)}, tick={number(tick)}. "
            text+="Назови entry за нужным экстремумом на tick и stop за противоположным на tick. Ордер не исполнялся."
            add(i,v,"TC2","TC2.WM1.TRIGGER",124,text,{"entry":float(high+tick if long else low-tick),"stop":float(low-tick if long else high+tick)},"")
            after=Decimal(sign if not v else -sign)*Decimal(rng.randrange(1,100))/100
            w4_intro=("Course PDF44","Проверка минимума W4","Условие коррекции Course","Минимальный критерий W4","Волновая учебная ситуация","Проверь условие для четвёртой волны","Задача по Course о W4")[form]
            text=f"{w4_intro}: масштаб 120 баров после предполагаемой W3, 5/34 на закрытиях изменился с {number(before)} на {number(after)}. Завершение младшей c/e не подтверждено. Нужны minimum_w4_met и w4_finished."
            add(i,v,"COURSE","COURSE.W4.ZERO",44,text,{"minimum_w4_met":before*after<0,"w4_finished":False},"")
            signal=hist+(delta if v else -delta)
            ao_intro=("Course PDF45","Моментум по Course","Профиль 5/34/5","Сравнение гистограммы и её сглаживания","Условие импульса Course","Два значения для определения momentum","Учебный профиль Course")[form]
            text=f"{ao_intro}: 5/34 histogram={number(hist)}, собственная SMA(5) этой гистограммы={number(signal)}. Укажи momentum LONG/SHORT из сравнения этих двух величин. Данных для входа нет."
            add(i,v,"COURSE","COURSE.AO5345",45,text,{"momentum":"LONG" if hist>signal else "SHORT"},"")
            right=1+v
            fractal_intro=("TC2 строгий пятибарный фрактал","Пятибарный фрактал HIGH","Условие подтверждения TC2 WM3","Проверка фрактального максимума","Вопрос о зрелости фрактала","Оцени подтверждение экстремума","Фрактальная учебная ситуация")[form]
            text=f"{fractal_intro}: HIGH={fractal_high}, он выше всех допустимых соседей. Два левых бара закрыты, справа закрыто {right} подходящих баров. Неподтверждённые правые данные не использовать. Нужен confirmed сейчас."
            add(i,v,"TC2","TC2.WM3",147,text,{"confirmed":right==2},"")
    expected=1664 if split=="train" else 84
    if len(rows)!=expected or len({r['question'] for r in rows})!=len(rows):
        raise ValueError("EXACT_UNIQUE_FOCUSED_COHORT_REQUIRED")
    return rows


def focused_training_lessons():
    from .contrastive_gpu_research import training_lessons as previous_lessons
    prose,old_typed,old_validation=previous_lessons()
    typed=focused_cases("train")
    groups={}
    for row in typed:
        if row['teaching_explanation']:
            groups.setdefault(row['pair_id'],[]).append(row)
    for i,rows in enumerate(groups.values()):
        row=rows[i%len(rows)]
        prose.append({**row,"question":"Объясни проверяемый расчёт или условие: "+row['question'].split(' Только один JSON:')[0],
            "answer":row['teaching_explanation']+" Итоговый учебный ответ: "+row['answer']})
    return prose,old_typed+typed,old_validation


def checkpoint_selection(grade:dict,max_source_regression:float)->dict:
    if set(grade['per_rule'])!=set(RULES) or not math.isfinite(max_source_regression):
        raise ValueError("ALL_SEVEN_FINITE_DEVELOPMENT_RULES_REQUIRED")
    if any(type(v['correct']) is not int or type(v['total']) is not int or not 0<=v['correct']<=v['total'] or v['total']<=0 for v in grade['per_rule'].values()):
        raise ValueError("NONEMPTY_INTEGER_RULE_COUNTS_REQUIRED")
    per_rule=[v['correct']/v['total'] for v in grade['per_rule'].values()]
    if any(not 0<=v<=1 for v in per_rule) or not 0<=grade['accuracy']<=1:
        raise ValueError("BOUNDED_GENERATED_DEVELOPMENT_REQUIRED")
    collapsed=len(grade['constant_fields_despite_varied_expected_values'])
    minimum=min(per_rule)
    score=minimum+0.1*grade['accuracy']-0.1*collapsed
    if max_source_regression>0.02:score=-3.0-max_source_regression
    return {'score':score,'minimum_rule_accuracy':minimum,'collapsed_rules':collapsed,
        'macro_accuracy':sum(per_rule)/len(per_rule),'source_guard_passed':max_source_regression<=0.02}


def design_manifest():
    splits={s:focused_cases(s) for s in ('train','validation','test')}
    return {'mode':'FOCUSED_RULE_REPAIR_V1','initial_adapter_sha256':INITIAL_SHA,
        'old56_status':'REVEALED_DEVELOPMENT_DIAGNOSTIC_NEVER_REUSED_AS_UNSEEN',
        'new_unique_cases':{s:len(v) for s,v in splits.items()},
        'cases_per_rule':{s:dict(Counter(r['rule_id'] for r in v)) for s,v in splits.items()},
        'questions_sha256':{s:questions_digest(question_rows(v)) for s,v in splits.items()},
        'generated_development_cases_actually_used':84,'new_sealed_test_cases':84,
        'selection':'minimum_rule_accuracy + 0.1*accuracy - 0.1*collapsed_rule_count; source regression <=2%',
        'source_train_repetitions':2,'inference_mode_precommitted':'direct',
        'expert_market_chart_exam':False,'execution_allowed':False}
