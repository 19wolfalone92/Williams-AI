"""Fresh, balanced calculation lessons; machine labels, never expert charts.

The historical October exam is a revealed regression set. New synthetic
development and test seeds are fixed here before another model is trained.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_FLOOR
import json
import random

from .historical_rule_curriculum import FAMILIES, canonical, digest, prompts_only, score
from .book_rules import registry
from .restorative_book_practice import restorative_cases

MODE = "JOINT_BOOK_HISTORY_BOUNDARY_RESEARCH_20261106_V1"
RUN_SLUG = "williams-ai-joint-rules-20261106"
INITIAL_SHA = "f1846adcb666dd31d09a1e17e237fffcd9dd38fa38497014f1f71064520de5f5"
SEEDS = {"train": 106713041, "validation": 106824167, "test": 106935281}
NUMERIC_SEEDS = {"train": 106716037, "validation": 106827149, "test": 106938263}
PROTOCOL = {"mode": "direct", "max_new_tokens": 256, "seconds": 25, "do_sample": False}
LIMITS = {"optimizer_attempts": 2048, "training_seconds": 7200,
          "evaluation_steps": 512, "learning_rate": 5e-6,
          "maximum_kernel_seconds": 14400, "minimum_free_gpu_seconds": 16200}
SOURCES = {card["id"]: (card["source"], card["pdf_pages"][0])
           for card in registry()["rule_cards"] if card["id"] in FAMILIES}



def num(value):
    return format(Decimal(str(value)), "f")


def boundary_cases(split):
    if split not in SEEDS:
        raise ValueError("FROZEN_JOINT_SPLIT_REQUIRED")
    rng = random.Random(SEEDS[split])
    count = {"train": 128, "validation": 8, "test": 16}[split]
    rows = []

    def add(i, rule, operands, question, answer, explanation):
        edition, page = SOURCES[rule]
        # Paraphrases and engineering calculations, no copied book passages.
        prefix = {"train": "Учебная проверка", "validation": "Новые исходные величины",
                  "test": "Применение к другим величинам"}[split]
        question = (prefix + ". " + question + "\nИсходные величины: " + canonical(operands).decode()
                    + "\nВерни только JSON с полями " + ",".join(answer)
                    + ",execution_allowed. execution_allowed=false; без других полей.")
        rows.append({"case_id": f"joint-{split}-{i}-{rule}", "rule_id": rule,
                     "edition": edition, "pdf_page": page, "question": question,
                     "answer": canonical({**answer, "execution_allowed": False}).decode(),
                     "teaching_explanation": explanation, "operands": operands,
                     "label_origin": "AUTHORED_MACHINE_CHECKED_CALCULATION_NOT_EXPERT_CHART"})

    for i in range(count):
        # Easy exact and harder fractional cases. Zero quantity and nondecimal
        # step multiples are deliberately present; risk is never rounded up.
        equity = Decimal(rng.randrange(501, 19001))
        fraction = Decimal(rng.choice(("0.0025", "0.004", "0.005", "0.0075", "0.01")))
        distance = Decimal(rng.choice(("0.3", "0.7", "1.2", "2.6", "5.2", "13.7")))
        step = Decimal(rng.choice(("0.001", "0.01", "0.03", "0.05", "0.1", "0.25")))
        if i % 4 == 0:
            equity, fraction = Decimal(rng.randrange(1, 41)*1000), Decimal("0.005")
            distance, step = Decimal(rng.randrange(1, 11)), Decimal("0.1")
        if i % 8 == 1:
            distance = equity * fraction / step * Decimal("1.1")
        budget = equity*fraction
        quantity = (budget/(distance*step)).to_integral_value(rounding=ROUND_FLOOR)*step
        add(i, "TC1.PSY.PLAN", {"equity": num(equity), "risk_fraction": num(fraction),
            "distance": num(distance), "quantity_step": num(step)},
            "Размер условной позиции. risk_budget=equity*risk_fraction. quantity равен наибольшему "
            "кратному quantity_step, риск которого quantity*distance не превышает бюджет. "
            "Стоимость пункта 1 USDT, издержки в этой учебной арифметике исключены. "
            "Это инженерный расчёт к плану TC1, не исторический фильтр биржи.",
            {"risk_budget": float(budget), "quantity": float(quantity)},
            f"Бюджет {equity}*{fraction}={budget}. Делитель {distance}*{step}={distance*step}. "
            f"floor(бюджет/делитель)*step={quantity}; следующий шаг превышает бюджет.")

        # Factorial side/sign/readiness coverage; a flat step breaks the run.
        side = ("LONG", "SHORT")[(i//2) % 2]
        sign = -1 if i % 2 == 0 else 1
        required = 2 if (sign > 0) == (side == "LONG") else 3
        ready = bool((i//4) % 2)
        run = required if ready else required-1
        unit = Decimal(rng.randrange(1, 27))/100
        last = Decimal(sign)*Decimal(rng.randrange(20, 400))
        direction = 1 if side == "LONG" else -1
        tail = [last-Decimal(direction)*unit*j for j in reversed(range(run+1))]
        values = [tail[0]+Decimal(direction)*unit, *tail]
        if i % 16 == 0:
            values = [last]*5; required = 2 if (last > 0) == (side == "LONG") else 3; ready = False
        elif i % 16 == 1:
            probe = Decimal(rng.randrange(1, 90001)) + Decimal(i+1)/10000
            values = [probe*3, probe*2, probe, Decimal(0)]
            required, ready = None, False
        operands = {"side": side, "ac_oldest_to_latest": [num(v) for v in values]}
        add(i, "NTD.AC", operands,
            "Проверь только AC. Растущий столбец зелёный, падающий красный; равенство разрывает серию. "
            "Для LONG считаем подряд зелёные в конце, для SHORT красные. С направлением знака AC нужно "
            "2 столбца, против знака 3. На точном нуле required_bars=null, signal_ready=false. "
            "Знак AO не задаёт порог и первого входа это не подтверждает.",
            {"required_bars": required, "signal_ready": ready},
            f"Направление {side}, последний AC={values[-1]}; порог {required}; готовность {ready}.")

        base = Decimal(rng.randrange(10, 500001))/100
        delta = Decimal(rng.randrange(1, 99))/100
        layouts = ((base, base+delta, base+2*delta), (base+2*delta, base+delta, base),
                   (base+delta, base, base+2*delta), (base, base, base))
        jaw, teeth, lips = layouts[i % 4]
        close = (min(jaw, teeth, lips)-delta, max(jaw, teeth, lips)+delta, teeth)[(i//4) % 3]
        order = "UP_STACK" if lips>teeth>jaw else "DOWN_STACK" if lips<teeth<jaw else "MIXED"
        add(i, "NTD.ALLIGATOR", {"visible_jaw": num(jaw), "visible_teeth": num(teeth),
            "visible_lips": num(lips), "close": num(close)},
            "Классифицируй видимые линии: order=UP_STACK при lips>teeth>jaw, DOWN_STACK при lips<teeth<jaw, "
            "иначе MIXED. above_all и below_all — строгие сравнения цены со всеми тремя линиями. "
            "Равенство не подходит. Это наблюдение, не самостоятельный вход.",
            {"order": order, "above_all": close>max(jaw, teeth, lips), "below_all": close<min(jaw, teeth, lips)},
            f"Линии {jaw}, {teeth}, {lips}; строгий порядок {order}; цена {close}.")

        previous = Decimal(rng.randrange(-900, 901))/100 + Decimal(i+1)/1000000
        now = Decimal(rng.randrange(-900, 901))/100
        if i % 8 == 0: now = previous
        if i % 8 == 1: now = Decimal(0)
        add(i, "NTD.AO", {"previous_ao": num(previous), "current_ao": num(now)},
            "above_zero означает current_ao>0, rising означает current_ao>previous_ao; "
            "crossed_zero означает строго previous_ao*current_ao<0. Равенство нулю не считается пересечением.",
            {"above_zero": now>0, "rising": now>previous, "crossed_zero": previous*now<0},
            f"Предыдущее {previous}, текущее {now}; равенство не даёт строгий переход.")

        tick = Decimal(rng.choice(("0.001", "0.01", "0.05", "0.25")))
        low = Decimal(rng.randrange(10, 100001))*tick
        high = low + Decimal(rng.randrange(1, 150))*tick
        entry, stop = (high+tick, low-tick) if side == "LONG" else (low-tick, high+tick)
        add(i, "TC2.WM1.TRIGGER", {"side": side, "last_bar_low": num(low),
            "last_bar_high": num(high), "research_tick": num(tick)},
            "Условный WM1: внешняя проверка ангуляции отдельно. Для LONG entry=high+tick, stop=low-tick; "
            "для SHORT entry=low-tick, stop=high+tick. Вычисли уровни entry и stop; исполнения нет.",
            {"entry": float(entry), "stop": float(stop)}, f"{side}: entry={entry}; stop={stop}.")

        kind = "HIGH" if i % 2 == 0 else "LOW"
        center = Decimal(rng.randrange(10, 100001))/100
        offset = Decimal(rng.randrange(1, 200))/100
        neighbor = center-offset if kind == "HIGH" else center+offset
        extrema = [neighbor, neighbor, center, neighbor, neighbor]
        if (i//2) % 2:
            extrema[3] = center if (i//4) % 2 else center+(offset if kind == "HIGH" else -offset)
        right = 1 if (i//4) % 2 else 2
        extrema = extrema[:3+right]
        shape = all((center>v if kind == "HIGH" else center<v) for j,v in enumerate(extrema) if j!=2)
        confirmed = right == 2 and shape
        add(i, "TC2.WM3", {"kind": kind, "closed_extrema_oldest_first": [num(v) for v in extrema],
            "pivot_offset": 2, "right_closed": right},
            "Строгий пятибарный фрактал: нужны два левых и два правых закрытых соседа, центр строго "
            "выше всех для HIGH или ниже всех для LOW. Равенство не подходит, будущих баров нет. Нужен confirmed.",
            {"confirmed": confirmed}, f"Правых закрытых {right}; строгая форма {shape}; confirmed={confirmed}.")

        signal = now if i % 4 == 0 else now+Decimal(rng.choice((-1, 1)))*offset
        momentum = "LONG" if now>signal else "SHORT" if now<signal else "NEUTRAL"
        add(i, "COURSE.AO5345", {"histogram": num(now), "own_sma5": num(signal)},
            "Сравни histogram с его собственной SMA5: выше LONG, ниже SHORT, равен NEUTRAL. "
            "Знак относительно нуля не определяет этот momentum.", {"momentum": momentum},
            f"{now} относительно {signal}: {momentum}.")
        end = (i+1)*10000 + rng.randrange(5, 10001)
        start = end + (-1, 0, 1, 3)[i % 4]
        add(i, "COURSE.HUMP", {"first_formation_inclusive_end": end,
            "second_formation_inclusive_start": start},
            "Противоположные пятибарные формации отдельно подтверждены. Проверь только hump_valid: "
            "первая и вторая не имеют общих включённых баров, если second_start>first_end. Общая граница — перекрытие.",
            {"hump_valid": start>end}, f"Строго {start}>{end}: {start>end}.")
        add(i, "COURSE.W4.ZERO", {"previous_5_34": num(previous), "current_5_34": num(now)},
            "Проверяем только zero_transition_seen: строгое произведение двух 5/34 меньше нуля. "
            "Наблюдение не подтверждает конец коррекции или степень волны. w4_finished=false.",
            {"zero_transition_seen": previous*now<0, "w4_finished": False},
            f"Переход={previous*now<0}; завершение W4 по двум отсчётам не доказано.")
    if len({r["question"] for r in rows}) != len(rows):
        raise ValueError("UNIQUE_JOINT_QUESTIONS_REQUIRED")
    return rows


def numeric_cases(split):
    return restorative_cases(split, seed_override=NUMERIC_SEEDS[split], case_namespace="joint-numeric-20261106")


def grade_groups(groups, responses):
    if set(groups) != set(responses):
        raise ValueError("ALL_JOINT_COHORTS_REQUIRED")
    return {name: score(rows, responses[name]) for name, rows in groups.items()}


def selection(grades, source, practice, references):
    import math
    if set(grades) != {"historical", "numeric", "boundary"}:
        raise ValueError("THREE_DEVELOPMENT_COHORTS_REQUIRED")
    from .deep_book_research import EDITIONS
    ratios = []
    for after, before in ((source, references["focused69_source"]),
                          (source, references["previous40_source"]),
                          (practice, references["focused69_practice"]),
                          (practice, references["previous40_practice"])):
        for edition in EDITIONS:
            a, b = after[edition], before[edition]
            if type(a) not in (int,float) or type(b) not in (int,float) or not math.isfinite(a) or not math.isfinite(b) or min(a,b)<=0:
                raise ValueError("FINITE_POSITIVE_BOOK_LOSSES_REQUIRED")
            ratios.append(a/b-1)
    regression = max(ratios)
    accuracies = [v["correct"]/v["total"] for g in grades.values() for v in g["per_rule"].values()]
    macro = sum(g["accuracy"] for g in grades.values())/len(grades)
    collapsed = sum(len(g["constant_fields_despite_varied_expected_values"]) for g in grades.values())
    value = min(accuracies) + .2*macro - .05*collapsed
    if regression > .02: value = -3-regression
    return {"score": value, "minimum_rule_accuracy": min(accuracies), "macro_accuracy": macro,
            "collapsed_rules": collapsed, "max_any_reference_book_regression": regression,
            "preserved_within_two_percent": regression<=.02}


def manifest(historical):
    boundary = {s: boundary_cases(s) for s in SEEDS}
    numeric = {s: numeric_cases(s) for s in SEEDS}
    all_groups = {s: {"historical": historical[s], "numeric": numeric[s], "boundary": boundary[s]} for s in SEEDS}
    prompt_sets = [{digest([r["rule_id"], r["operands"]]) for r in boundary[s]} for s in SEEDS]
    if any(a & b for i,a in enumerate(prompt_sets) for b in prompt_sets[i+1:]):
        raise ValueError("DISTINCT_JOINT_SPLITS_REQUIRED")
    return {"mode": MODE, "initial_adapter_sha256": INITIAL_SHA, "limits": LIMITS,
            "inference_protocol": PROTOCOL, "seeds": SEEDS, "numeric_seeds": NUMERIC_SEEDS,
            "counts": {s: {g:len(rows) for g,rows in groups.items()} for s,groups in all_groups.items()},
            "questions_sha256": {s: {g:digest(prompts_only(rows)) for g,rows in groups.items()} for s,groups in all_groups.items()},
            "all_nine_families": list(FAMILIES), "historical_test_is_revealed_regression": True,
            "fresh_synthetic_test_not_expert_charts": True, "source_training_repetitions": 2,
            "test_keys_sent_to_worker": False, "expert_chart_supervision_passed": False,
            "trading_qualification": False, "execution_allowed": False}
