"""Machine-checked book-rule exercises grounded in real closed OHLCV.

This is a separate arithmetic/indicator research protocol. It neither supplies
nor invents the two human approvals required by chart_supervision. Automated
labels are not expert wave counts, future returns, trade signals or fills.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_FLOOR
from hashlib import sha256
import gzip
import json
import math
from pathlib import Path
import random

from .book_rules import registry
from .chart_reader import Candle, indicators
from .contrastive_book_practice import SYSTEM, correct_answer, strict_answer

SCHEMA = "HISTORICAL_BOOK_RULE_CALCULATION_RESEARCH_V1"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT", "AVAXUSDT", "LTCUSDT")
INTERVALS = {"1h": (3600000, 1200, "H1", 20), "4h": (14400000, 480, "H4", 8), "1d": (86400000, 320, "D1", 4)}
STARTS = {"train": "2022-01-01", "validation": "2024-01-01", "test": "2025-01-01"}
FAMILIES = ("TC1.PSY.PLAN", "NTD.AC", "NTD.ALLIGATOR", "NTD.AO", "TC2.WM1.TRIGGER", "TC2.WM3", "COURSE.AO5345", "COURSE.HUMP", "COURSE.W4.ZERO")
PER_FAMILY = {"train": 160, "validation": 8, "test": 16}


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(obj):
    return sha256(canonical(obj)).hexdigest()


def iso(milliseconds):
    return datetime.fromtimestamp(milliseconds / 1000, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def number(value):
    return float(format(float(value), ".9g"))


def checked_pack(pack):
    if (not isinstance(pack, dict) or set(pack) != {"schema", "split", "datasets"}
            or pack["schema"] != SCHEMA or pack["split"] not in STARTS
            or not isinstance(pack["datasets"], list) or len(pack["datasets"]) != 30):
        raise ValueError("EXACT_DECLARED_PUBLIC_HISTORY_PACK_REQUIRED")
    start = int(datetime.fromisoformat(STARTS[pack["split"]]).replace(tzinfo=timezone.utc).timestamp() * 1000)
    seen = set()
    for dataset in pack["datasets"]:
        if (not isinstance(dataset, dict) or set(dataset) != {"symbol", "interval", "bars"}
                or dataset["symbol"] not in SYMBOLS or dataset["interval"] not in INTERVALS):
            raise ValueError("ONLY_DECLARED_REAL_USDM_SERIES_ALLOWED")
        key = dataset["symbol"], dataset["interval"]
        if key in seen:
            raise ValueError("DUPLICATE_MARKET_SERIES")
        seen.add(key)
        width, count, _, _ = INTERVALS[key[1]]
        if not isinstance(dataset["bars"], list) or len(dataset["bars"]) != count:
            raise ValueError("INCOMPLETE_CLOSED_HISTORY")
        for i, row in enumerate(dataset["bars"]):
            if (not isinstance(row, list) or len(row) != 7 or type(row[0]) is not int
                    or row[0] != start + (i + 1) * width - 1
                    or any(not isinstance(v, str) for v in row[1:])):
                raise ValueError("CONTIGUOUS_ALIGNED_FINAL_CANDLES_ONLY")
            try:
                op, hi, lo, cl, vol = map(Decimal, row[1:6])
                if (any(not v.is_finite() for v in (op, hi, lo, cl, vol))
                        or min(op, hi, lo, cl) <= 0 or vol < 0
                        or lo > min(op, cl) or hi < max(op, cl) or lo > hi
                        or row[6] != "BINANCE_FINAL"):
                    raise ValueError("INVALID_REAL_OHLCV")
            except Exception as error:
                raise ValueError("INVALID_REAL_OHLCV") from error
    return pack


def pack_from_raw(root: Path, split: str):
    datasets = []
    for symbol in SYMBOLS:
        for interval in INTERVALS:
            raw = json.loads((root / split / f"{symbol}_{interval}.json").read_bytes())
            if raw["provider"] != "BINANCE_PUBLIC_CONNECTOR" or raw["split"] != split:
                raise ValueError("REAL_CONNECTOR_PROVENANCE_REQUIRED")
            datasets.append({"symbol": symbol, "interval": interval, "bars": [
                [r[6], *r[1:6], "BINANCE_FINAL"] for r in raw["klines"]]})
    return checked_pack({"schema": SCHEMA, "split": split, "datasets": datasets})


def load_pack(path: Path):
    return checked_pack(json.loads(gzip.decompress(path.read_bytes())))


def check_partitions(packs):
    if set(packs) != set(STARTS):
        raise ValueError("ALL_FROZEN_MARKET_SPLITS_REQUIRED")
    ranges = {}
    for split, pack in packs.items():
        checked_pack(pack)
        if pack["split"] != split:
            raise ValueError("PARTITION_CANNOT_BE_RELABELLED")
        ranges[split] = (min(d["bars"][0][0] - INTERVALS[d["interval"]][0] + 1 for d in pack["datasets"]),
                         max(d["bars"][-1][0] for d in pack["datasets"]))
    for first, second in (("train", "validation"), ("validation", "test")):
        if ranges[first][1] + 24 * 3600000 >= ranges[second][0]:
            raise ValueError("GLOBAL_CROSS_ASSET_CROSS_TIMEFRAME_EMBARGO_REQUIRED")
    return {s: {"start": iso(a), "end": iso(b)} for s, (a, b) in ranges.items()}


def _window_rows(dataset, end):
    """Slice BEFORE computing anything. Later candles cannot affect a row."""
    raw = dataset["bars"][end - 219:end + 1]
    if len(raw) != 220:
        raise ValueError("220_PAST_CANDLES_REQUIRED")
    candles = [Candle(datetime.fromtimestamp(r[0] / 1000, timezone.utc), *map(float, r[1:6])) for r in raw]
    line = indicators(candles)
    cards = {c["id"]: c for c in registry()["rule_cards"]}
    window = {"symbol": dataset["symbol"], "interval": dataset["interval"], "bars": raw}
    identity = digest(window)
    current = raw[-1]
    side = "LONG" if int(identity[:8], 16) % 2 else "SHORT"
    tick = Decimal("0.1" if dataset["symbol"] in {"BTCUSDT", "ETHUSDT", "BNBUSDT"} else "0.001")
    low, high = Decimal(current[3]), Decimal(current[2])
    entry, stop = (high + tick, low - tick) if side == "LONG" else (low - tick, high + tick)
    intro = f"Реальная история Binance USD-M, {dataset['symbol']} {INTERVALS[dataset['interval']][2]}, закрытие {iso(current[0])}. Последние 220 закрытых свечей; будущего нет. "

    def add(rule, observations, instruction, expected, explanation, stratum):
        card = cards[rule]
        if card["status"] != "REVIEWED_PAGES":
            raise ValueError("SOURCE_REFERENCED_RULE_REQUIRED")
        answer = {**expected, "execution_allowed": False}
        question = (intro + f"Источник {card['source']}, PDF {','.join(map(str, card['pdf_pages']))}. "
            + "Пересказ проверенного правила: " + card["summary"] + "\nВидимые данные: "
            + canonical(observations).decode() + "\n" + instruction
            + " Верни один JSON с полями " + ",".join(answer) + "; execution_allowed=false. Без других полей.")
        return {"case_id": identity[:20] + ":" + rule, "rule_id": rule, "edition": card["source"],
            "pdf_page": card["pdf_pages"][0], "question": question, "answer": canonical(answer).decode(),
            "teaching_explanation": explanation, "window_sha256": identity,
            "symbol": dataset["symbol"], "timeframe": INTERVALS[dataset["interval"]][2],
            "start": iso(raw[0][0] - INTERVALS[dataset["interval"]][0] + 1), "cutoff": iso(current[0]),
            "label_origin": "DETERMINISTIC_CLOSED_BAR_RULE_CALCULATION_NOT_EXPERT", "stratum": stratum}

    equity = Decimal((1000, 2000, 3000, 5000, 8000, 10000)[int(identity[8:12], 16) % 6])
    fraction = Decimal(("0.0025", "0.005", "0.0075", "0.01")[int(identity[12:16], 16) % 4])
    step = Decimal("0.001" if dataset["symbol"] in {"BTCUSDT", "ETHUSDT"} else "0.01")
    budget, distance = equity * fraction, abs(entry - stop)
    units = (budget / (distance * step)).to_integral_value(rounding=ROUND_FLOOR)
    quantity = units * step
    yield add("TC1.PSY.PLAN", {"equity": str(equity), "risk_fraction": str(fraction), "conditional_side": side,
        "conditional_entry": str(entry), "conditional_stop": str(stop), "research_quantity_step": str(step)},
        "Это условный расчёт размера, наличие WM1 не утверждается. Цена пункта 1 USDT. Издержки исключены только для упражнения; шаг учебный, не исторический фильтр биржи. Бюджет=капитал*доля; объём=floor(бюджет/abs(entry-stop)/step)*step. Вычисли risk_budget и quantity.",
        {"risk_budget": float(budget), "quantity": float(quantity)},
        f"Бюджет {equity}*{fraction}={budget}; дистанция {distance}; риск шага {distance*step}; целых шагов floor({budget}/{distance*step})={units}; количество {quantity}. Следующий шаг превышает бюджет. Это инженерная арифметика к плану TC1, не дословная формула книги.", side)

    ac = [number(x) for x in line["ac"][-8:]]
    count = 0
    for a, b in reversed(list(zip(ac, ac[1:]))):
        if (b > a if side == "LONG" else b < a): count += 1
        else: break
    required = None if ac[-1] == 0 else (2 if (ac[-1] > 0) == (side == "LONG") else 3)
    ready = required is not None and count >= required
    yield add("NTD.AC", {"side": side, "ac_oldest_to_latest": ac},
        "Посчитай цветную серию из видимых AC: LONG растёт, SHORT падает; равенство прерывает серию. С направлением знака AC нужны 2 столбца, против него 3. На точном нуле required_bars=null и signal_ready=false. Нужны required_bars и signal_ready; это только условие AC, не первый вход NTD.",
        {"required_bars": required, "signal_ready": ready},
        f"Направление {side}; последний AC={ac[-1]}; цветная серия={count}; порог={required}; готовность={ready}. AO не выбирает порог AC.", f"{side}:{required}:{ready}")

    jaw, teeth, lips = (number(line[k][-1]) for k in ("jaw", "teeth", "lips"))
    close = number(current[4])
    order = "UP_STACK" if lips > teeth > jaw else "DOWN_STACK" if lips < teeth < jaw else "MIXED"
    yield add("NTD.ALLIGATOR", {"visible_jaw": jaw, "visible_teeth": teeth, "visible_lips": lips, "close": close},
        "Классифицируй только порядок линий: UP_STACK при lips>teeth>jaw, DOWN_STACK при lips<teeth<jaw, иначе MIXED. Это инженерные имена наблюдений, не доказательство фазы/входа. Нужны order и above_all, below_all для close относительно трёх видимых линий.",
        {"order": order, "above_all": close > max(jaw, teeth, lips), "below_all": close < min(jaw, teeth, lips)},
        f"Видимые линии вычислены со сдвигом из прошлого. Сравниваю {lips}, {teeth}, {jaw}; порядок {order}; цена {close}. Один порядок не подтверждает вход.", order)

    previous, now = (number(x) for x in line["ao"][-2:])
    yield add("NTD.AO", {"previous_ao": previous, "current_ao": now},
        "Нужны above_zero, rising, crossed_zero. Строгое пересечение: previous_ao*current_ao<0. Равенство нулю здесь пересечением не считается. Самостоятельный вход не утверждается.",
        {"above_zero": now > 0, "rising": now > previous, "crossed_zero": previous * now < 0},
        f"Сравниваю AO={now} с нулём и предыдущим {previous}; произведение знаков определяет только пересечение, не TC2 Super AO.", f"{now>0}:{now>previous}:{previous*now<0}")

    yield add("TC2.WM1.TRIGGER", {"side": side, "last_bar_low": str(low), "last_bar_high": str(high), "research_tick": str(tick)},
        "Условно вычисли entry и stop, ЕСЛИ отдельная проверка подтвердит WM1 и ангуляцию. Эти свечи сами не доказывают ангуляцию. Вход за направленным экстремумом, защита за противоположным на учебный tick. Это уровни плана, не сигнал или fill.",
        {"entry": float(entry), "stop": float(stop)}, f"{side}: entry={entry}, stop={stop}. Ангуляция в этом упражнении не размечена.", side)

    pivot = len(raw) - (3 if int(identity[16:20], 16) % 2 else 2)
    kind = "HIGH" if int(identity[20:24], 16) % 2 else "LOW"
    index = 2 if kind == "HIGH" else 3
    local = raw[pivot-2:min(pivot+3, len(raw))]
    values = [float(r[index]) for r in local]
    right = len(local) - 3
    shape = all((values[2] > v if kind == "HIGH" else values[2] < v) for i, v in enumerate(values) if i != 2)
    confirmed = right == 2 and shape
    yield add("TC2.WM3", {"kind": kind, "closed_extrema_oldest_first": values, "pivot_offset": 2, "right_closed": right},
        "Строгий пятибарный фрактал требует двух левых и двух правых закрытых соседей и строгого экстремума центра. Нужен confirmed. Подтверждение формации не равно допустимости пробоя относительно Teeth или биржевому исполнению.",
        {"confirmed": confirmed}, f"Справа закрыто {right}; строгая форма={shape}; подтверждение={confirmed}. Будущих правых баров не добавляю.", str(confirmed))

    signal = number(sum(line["ao"][-5:]) / 5)
    momentum = "LONG" if now > signal else "SHORT" if now < signal else "NEUTRAL"
    yield add("COURSE.AO5345", {"histogram": now, "own_sma5": signal},
        "Нужен momentum: LONG выше собственной SMA5, SHORT ниже, NEUTRAL при равенстве. Нулевой уровень отвечает на другой вопрос; вход не дан.",
        {"momentum": momentum}, f"Сравниваю {now} с {signal}: {momentum}. Степень волны этим не доказана.", momentum)

    fractals = []
    for i in range(2, len(raw)-2):
        others = [raw[j] for j in (i-2, i-1, i+1, i+2)]
        hi = all(Decimal(raw[i][2]) > Decimal(r[2]) for r in others)
        lo = all(Decimal(raw[i][3]) < Decimal(r[3]) for r in others)
        if hi != lo: fractals.append((i, "HIGH" if hi else "LOW"))
    pairs = [(a, b) for a, b in zip(fractals, fractals[1:]) if a[1] != b[1]]
    if pairs:
        a, b = pairs[-1]
        end_a, start_b = a[0] + 2, b[0] - 2
        valid = start_b > end_a
        yield add("COURSE.HUMP", {"first_kind": a[1], "second_kind": b[1], "first_formation_inclusive_end": end_a, "second_formation_inclusive_start": start_b},
            "Обе противоположные строгие формации подтверждены на этом закрытом участке. Проверь только hump_valid по неперекрытию всех пяти баров формаций. Общий граничный бар тоже является перекрытием. Действительный торговый вход не утверждается.",
            {"hump_valid": valid}, f"Сравниваю включённые границы {start_b}>{end_a}: {valid}. Речь о геометрическом условии, не готовой кампании.", str(valid))

    yield add("COURSE.W4.ZERO", {"previous_5_34": previous, "current_5_34": now},
        "Условный урок о W4 после предполагаемой W3. Эти два отсчёта не устанавливают саму W3 или степень волны. Нужны zero_transition_seen (строгое произведение<0) и w4_finished=false: наблюдаемый переход не доказывает завершение коррекции.",
        {"zero_transition_seen": previous * now < 0, "w4_finished": False},
        f"Произведение двух видимых значений {previous}*{now}; строгий переход={previous*now<0}. Завершение младшей структуры неизвестно.", str(previous * now < 0))


def cases_from_pack(pack):
    checked_pack(pack)
    split = pack["split"]
    pools = defaultdict(lambda: defaultdict(list))
    for dataset in pack["datasets"]:
        stride = INTERVALS[dataset["interval"]][3]
        for end in range(219, len(dataset["bars"]), stride):
            for row in _window_rows(dataset, end):
                pools[row["rule_id"]][row["stratum"]].append(row)
    selected = []
    for family in FAMILIES:
        groups = pools[family]
        rng = random.Random(f"historical-20261029:{split}:{family}")
        for values in groups.values(): rng.shuffle(values)
        output = []
        while len(output) < PER_FAMILY[split]:
            added = False
            for stratum in sorted(groups):
                if groups[stratum] and len(output) < PER_FAMILY[split]:
                    output.append(groups[stratum].pop()); added = True
            if not added: raise ValueError("INSUFFICIENT_REAL_MARKET_STRATA")
        selected.extend(output)
    if len({r["case_id"] for r in selected}) != len(selected):
        raise ValueError("UNIQUE_HISTORICAL_CASE_IDENTITIES_REQUIRED")
    return selected


def prompts_only(rows):
    return [{"case_id": r["case_id"], "domain": r["rule_id"], "prompt": r["question"]} for r in rows]


def score(rows, responses):
    expected = {r["case_id"]: r for r in rows}
    if (len(expected) != len(rows) or len(responses) != len(rows)
            or {r["case_id"] for r in responses} != set(expected)
            or len({r["case_id"] for r in responses}) != len(responses)):
        raise ValueError("EXACT_HISTORICAL_GENERATION_COHORT_REQUIRED")
    counts = defaultdict(lambda: {"correct": 0, "total": 0})
    fields, keys, details = defaultdict(list), defaultdict(list), []
    for response in responses:
        row = expected[response["case_id"]]
        if any(type(response.get(k)) is not bool for k in ("thinking_complete", "response_time_limit_reached")):
            raise ValueError("REAL_BOOLEAN_COMPLETION_FLAGS_REQUIRED")
        good = (response["thinking_complete"] and not response["response_time_limit_reached"]
                and correct_answer(response["response"], row["answer"]))
        counts[row["rule_id"]]["total"] += 1
        counts[row["rule_id"]]["correct"] += int(good)
        fields[row["rule_id"]].append(strict_answer(response["response"]))
        keys[row["rule_id"]].append(json.loads(row["answer"]))
        details.append({"case_id": row["case_id"], "correct": bool(good)})
    collapsed = {}
    for family, values in keys.items():
        for key in values[0]:
            wanted = {canonical(v[key]) for v in values}
            got = {canonical(v.get(key)) for v in fields[family] if v is not None}
            if len(wanted) > 1 and len(got) == 1: collapsed.setdefault(family, []).append(key)
    total = len(rows)
    return {"correct": sum(v["correct"] for v in counts.values()), "total": total,
        "accuracy": sum(v["correct"] for v in counts.values()) / total,
        "per_rule": dict(counts), "constant_fields_despite_varied_expected_values": collapsed,
        "timeouts": sum(r["response_time_limit_reached"] for r in responses),
        "incomplete": sum(not r["thinking_complete"] for r in responses), "cases": details,
        "expert_market_chart_exam": False, "trading_qualification": False, "execution_allowed": False}


def audit_cases(rows):
    return {"cases": len(rows), "unique_windows": len({r["window_sha256"] for r in rows}),
        "per_rule": dict(Counter(r["rule_id"] for r in rows)),
        "per_timeframe": dict(Counter(r["timeframe"] for r in rows)),
        "assets": sorted({r["symbol"] for r in rows}), "questions_sha256": digest(prompts_only(rows)),
        "calculation_labels_not_expert_chart_labels": True, "execution_allowed": False}
