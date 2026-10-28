"""Authored arithmetic explanations and a fresh parameter-transfer exam.

The revealed 56/84 exams are development history. These exercises are not
human-labeled market charts. Keys are calculated on the CPU controller only.
"""
from __future__ import annotations
from collections import Counter
from decimal import Decimal, ROUND_FLOOR
from hashlib import sha256
import json
import math
import random
import re

from .contrastive_book_practice import RULES, SYSTEM, question_rows, questions_digest, value_spans

INITIAL_SHA = "00daeb1d26417713334b88580735c944108c1a4d8fa4b5dd3cc3145dfa42533f"
PREVIOUS40_SHA = "bf8580fbb74d4ae8b041332146e64408f06900999837826d06ef93b4a6404891"
PREVIOUS_RECEIPT_SHA = "73f1bc1926a6df26c9361486dfaa13eaed1549190de19df7e5283ee66a07c19a"
RUN_SLUG = "williams-ai-restorative-2t4-20261028"
CONTROL_URL = "https://raw.githubusercontent.com/19wolfalone92/Williams-AI/research/private-four-book-deep-training-20261027/experiments/restorative-training-control.json"


def number(x):
    return format(Decimal(x), "f")


def calculated_answer(rule: str, question: str) -> dict:
    """Recalculate keys from labeled operands, independently of lesson answers."""
    values = {k: Decimal(v) for k, v in re.findall(r"\b([A-Z][A-Z0-9_]*)=(-?\d+(?:\.\d+)?)", question)}
    if rule == "TC1.PSY.PLAN":
        e, f, d, s = (values[k] for k in ("E", "F", "D", "S"))
        if min(e, f, d, s) <= 0:
            raise ValueError("POSITIVE_RISK_INPUTS_REQUIRED")
        b = e * f
        q = (b / (d * s)).to_integral_value(rounding=ROUND_FLOOR) * s
        result = {"risk_budget": float(b), "quantity": float(q)}
    elif rule == "COURSE.HUMP":
        result = {"hump_valid": values["B"] > values["A"]}
    elif rule == "NTD.AC":
        if values["AC"] == 0:
            raise ValueError("ZERO_BOUNDARY_NOT_CLAIMED")
        required = 3 if values["AC"] < 0 else 2
        result = {"required_green": required, "signal_ready": values["G"] >= required}
    elif rule == "TC2.WM1.TRIGGER":
        lo, hi, tick, side = (values[k] for k in ("L", "H", "T", "SIDE"))
        result = {"entry": float(hi + tick if side == 1 else lo - tick),
                  "stop": float(lo - tick if side == 1 else hi + tick)}
    elif rule == "COURSE.W4.ZERO":
        result = {"minimum_w4_met": values["P"] * values["N"] < 0, "w4_finished": False}
    elif rule == "COURSE.AO5345":
        result = {"momentum": "LONG" if values["HIST"] > values["SMA"] else "SHORT"}
    elif rule == "TC2.WM3":
        result = {"confirmed": values["RIGHT"] >= 2}
    else:
        raise ValueError("EXPLICIT_RULE_REQUIRED")
    return {**result, "execution_allowed": False}


def restorative_cases(split: str) -> list[dict]:
    if split not in ("train", "validation", "test"):
        raise ValueError("EXPLICIT_RESTORATIVE_SPLIT_REQUIRED")
    rng = random.Random({"train": 48102731, "validation": 59213843, "test": 60324959}[split])
    count = 64 if split == "train" else 8
    introductions = {
        "train": ("Реши учебное условие.", "Проверь числовую ситуацию.", "Выполни проверку правила."),
        "validation": ("Разбери новый набор исходных величин.", "Оцени условие по заданным обозначениям."),
        "test": ("Независимое задание: сделай вывод из этих данных.", "Дан иной пример применения правила; верни требуемый результат."),
    }[split]
    rows = []
    used_ends = set()
    def add(i, variant, ed, rule, page, text, explanation):
        question = introductions[i % len(introductions)] + " " + text
        result = calculated_answer(rule, question)
        fields = ", ".join(k + ": " + ("true/false" if type(v) is bool else "число" if type(v) in (int, float) else "строка") for k, v in result.items())
        question += " Финальный ответ — один JSON: " + fields + ". execution_allowed=false. Без дополнительных полей."
        rows.append({"case_id": f"restorative-{split}-{i}-{rule}-{variant}", "pair_id": f"restorative-{split}-{i}-{rule}",
            "edition": ed, "rule_id": rule, "pdf_page": page, "question": question,
            "answer": json.dumps(result, ensure_ascii=False, separators=(",", ":")), "teaching_explanation": explanation})
    for i in range(count):
        end = rng.randrange(10 ** (1 + i % 5), 10 ** (2 + i % 5))
        while end in used_ends:
            end = rng.randrange(10 ** (1 + i % 5), 10 ** (2 + i % 5))
        used_ends.add(end)
        gap = rng.randrange(1, 13)
        equity = Decimal(rng.randrange(701, 29001))
        fraction = Decimal(rng.choice(("0.0025", "0.004", "0.005", "0.006", "0.0075", "0.01")))
        distance = Decimal(rng.choice(("0.3", "0.5", "0.6", "1.2", "1.5", "2.4", "2.6", "3.5", "5.2", "7.5")))
        step = Decimal(rng.choice(("0.01", "0.02", "0.03", "0.05", "0.1")))
        tick = Decimal(rng.choice(("0.01", "0.05", "0.1", "0.25", "0.5")))
        low = Decimal(rng.randrange(501, 23001)) * tick
        high = low + Decimal(rng.randrange(2, 151)) * tick
        hist = Decimal(rng.randrange(-1901, 1902)) / 100
        delta = Decimal(rng.randrange(1, 51)) / 100
        prior = Decimal(rng.choice((-1, 1))) * (Decimal(rng.randrange(1, 501)) + Decimal(i)/1000) / 100
        for v in (0, 1):
            begin = end + gap if v else end - (0 if i % 3 == 0 else gap)
            text = (f"Hump: A={end} — последний включённый бар первой крайней формации; B={begin} — первый включённый бар второй. "
                if split == "train" else
                f"Course Hump. Более ранняя крайняя группа содержит свой последний бар A={end}. Более поздняя группа начинается включённым баром B={begin}. ")
            text += "Оба противоположных фрактала и остальные условия уже подтверждены. Нужна только проверка неперекрытия."
            explanation = f"Для включённых интервалов требуется B>A. Здесь {begin}-{end}={begin-end}. " + ("Разность положительна, пересечения нет." if begin > end else "Разность неположительна, есть общий бар; равенство тоже не подходит.")
            add(i, v, "COURSE", "COURSE.HUMP", 72, text, explanation)
            d = distance * (v + 1); budget = equity * fraction
            lots = (budget / (d * step)).to_integral_value(rounding=ROUND_FLOOR); quantity = lots * step
            text = (f"Риск: капитал E={number(equity)}, доля F={number(fraction)}, стоп-дистанция D={number(d)}, шаг количества S={number(step)}. "
                if split == "train" else
                f"Шаг объёма S={number(step)}, расстояние между входом и защитой D={number(d)}. Выделяемая доля F={number(fraction)} от капитала E={number(equity)}. ")
            text += "Пункт стоит 1 USDT; издержки в этой арифметике исключены. Объём округли вниз к шагу."
            explanation = (f"Бюджет={number(equity)}*{number(fraction)}={number(budget)}. Риск одного шага={number(d)}*{number(step)}={number(d*step)}. "
                f"Целых шагов floor({number(budget)}/{number(d*step)})={number(lots)}; объём={number(lots)}*{number(step)}={number(quantity)}. "
                f"Его риск {number(quantity*d)} не превышает {number(budget)}; следующий шаг рискует {number((quantity+step)*d)}, что выше бюджета.")
            add(i, v, "TC1", "TC1.PSY.PLAN", 83, text, explanation)
            ac = Decimal(-1 if v == 0 else 1) * Decimal(rng.randrange(1, 99991)) / 100000
            required = 3 if ac < 0 else 2
            greens = (required-1, required, required+1) if split == "train" else ((1, 2, 3, 4)[i % 4],)
            for j, green in enumerate(greens):
                text = (f"NTD BUY по AC: текущий AC={number(ac)}, зелёных подряд G={green}, перед серией красный. "
                    if split == "train" else
                    f"Для покупки в NTD после красного закрылись G={green} зелёных подряд. Текущее значение AC={number(ac)}. ")
                text += "AO положителен и не определяет порог AC. Назови необходимую длину зелёной серии и её готовность."
                explanation = f"AC {number(ac)} {'ниже' if ac<0 else 'выше'} нуля. Для BUY нужно {required} зелёных. Серия {green} {'достигла' if green>=required else 'не достигла'} порога {required}. Знак AO не меняет правило."
                add(i, f"{v}-{j}", "NTD", "NTD.AC", 75, text, explanation)
            side = 1 if v == 0 else -1
            text = f"TC2 WM1: закрытый {'бычий' if side==1 else 'медвежий'} бар, SIDE={side}; low L={number(low)}, high H={number(high)}, tick T={number(tick)}. Ангуляция подтверждена. Entry на tick за направленным экстремумом, stop за противоположным; исполнения нет."
            explanation = f"{'LONG: entry=H+T, stop=L-T' if side==1 else 'SHORT: entry=L-T, stop=H+T'}. Это уровни условного плана, не сделка."
            add(i, v, "TC2", "TC2.WM1.TRIGGER", 124, text, explanation)
            after = abs(prior) * (1 if prior > 0 else -1) * (1 if v == 0 else -1)
            text = f"Course W4, масштаб 120 баров после предполагаемой W3: прошлое 5/34 P={number(prior)}, текущее N={number(after)}. Младшая c/e не завершена доказательно. Различай минимум W4 и завершение."
            explanation = ("Знаки различаются, минимум перехода нуля выполнен. " if prior*after<0 else "Знак не изменился, минимум перехода нуля не выполнен. ") + "Независимо от этого конец W4 не подтверждён."
            add(i, v, "COURSE", "COURSE.W4.ZERO", 44, text, explanation)
            sma = hist + (delta if v else -delta)
            text = f"Course 5/34/5: гистограмма HIST={number(hist)}, её собственная SMA(5) SMA={number(sma)}. Определи momentum LONG/SHORT по сравнению; данных для входа нет."
            explanation = f"Сравниваю {number(hist)} с {number(sma)}, а не с нулём. Гистограмма {'выше' if hist>sma else 'ниже'} своей SMA, значит momentum {'LONG' if hist>sma else 'SHORT'}."
            add(i, v, "COURSE", "COURSE.AO5345", 45, text, explanation)
            right = 1 + v
            text = f"TC2: строгий пятибарный фрактальный максимум HIGH={end} выше допустимых соседей; слева два закрытых бара, справа закрытых подходящих RIGHT={right}. Открытые правые свечи не использовать. Подтверждён ли фрактал сейчас?"
            explanation = f"Справа нужно два закрытых подходящих бара. Их {right}, поэтому фрактал {'подтверждён' if right==2 else 'пока не подтверждён'}."
            add(i, v, "TC2", "TC2.WM3", 147, text, explanation)
    expected = 1152 if split == "train" else 112
    if len(rows) != expected or len({r['question'] for r in rows}) != len(rows):
        raise ValueError("EXACT_UNIQUE_RESTORATIVE_COHORT_REQUIRED")
    return rows


def tokenize_reasoned(rows, tokenizer, max_tokens=768):
    """Supervise short authored calculations followed by the strict final JSON."""
    result = []
    for row in rows:
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": row["question"]}]
        prefix_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        prefix = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
        completion = "<think>\n" + row["teaching_explanation"] + "\n</think>\n\n" + row["answer"] + tokenizer.eos_token
        rendered = prefix_text + completion
        encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        ids = encoded["input_ids"]
        if not prefix_text.endswith("<|im_start|>assistant\n") or ids[:len(prefix)] != prefix or len(ids) > max_tokens:
            raise ValueError("EXACT_REASONED_TEMPLATE_NO_TRUNCATION_REQUIRED")
        start = rendered.rfind(row["answer"])
        spans = [(a+start, b+start, w) for a,b,w in value_spans(row["answer"])]
        weights = [0.0] * len(prefix)
        for a,b in encoded["offset_mapping"][len(prefix):]:
            weights.append(max([1.0] + [w for lo,hi,w in spans if a<hi and b>lo]))
        result.append({"input_ids": ids, "attention_mask": [1]*len(ids),
            "labels": [-100]*len(prefix)+ids[len(prefix):], "loss_weights": weights})
    return result


def restorative_selection(grade, source_loss, practice_loss, baseline_source, baseline_practice):
    from .focused_book_practice import checkpoint_selection
    editions = ("TC1", "NTD", "TC2", "COURSE")
    if any(not math.isfinite(d[e]) or d[e] <= 0 for d in (source_loss, practice_loss, baseline_source, baseline_practice) for e in editions):
        raise ValueError("FINITE_ALL_BOOK_PRESERVATION_REQUIRED")
    source_regression = max(source_loss[e]/baseline_source[e]-1 for e in editions)
    practice_regression = max(practice_loss[e]/baseline_practice[e]-1 for e in editions)
    selection = checkpoint_selection(grade, max(source_regression, practice_regression))
    return {**selection, "max_source_regression": source_regression, "max_practice_regression": practice_regression}


def design_manifest():
    rows = {s: restorative_cases(s) for s in ("train", "validation", "test")}
    return {"mode": "RESTORATIVE_REASONED_AND_DIRECT_REPLAY_V1", "initial_adapter_sha256": INITIAL_SHA,
        "previous_receipt_sha256": PREVIOUS_RECEIPT_SHA, "new_unique_cases": {s:len(v) for s,v in rows.items()},
        "cases_per_rule": {s:dict(Counter(r['rule_id'] for r in v)) for s,v in rows.items()},
        "questions_sha256": {s:questions_digest(question_rows(v)) for s,v in rows.items()},
        "old56_and_old84_are_revealed_development": True, "inference_mode_precommitted": "thinking",
        "selected_on_test": False, "all_122_practice_and_178_source_windows_guard_checkpoints": True,
        "maximum_regression_any_book": 0.02, "training_answer_modes": ["direct", "short_authored_reasoning"],
        "expert_market_chart_exam": False, "execution_allowed": False}
