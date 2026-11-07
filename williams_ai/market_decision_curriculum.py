"""Causal chart interpretation exercises with traceable, limited answer keys."""
from __future__ import annotations

from bisect import bisect_right
from decimal import Decimal
import json

from .causal_timeframes import aggregate_closed_h1
from .chart_reader import indicators, validate_closed_bars
from .historical_rule_curriculum import canonical, digest, iso, number, prompts_only
from .joint_rule_curriculum import boundary_cases
from .restorative_book_practice import restorative_cases
from .wave_evidence import confirmed_price_swings

COHORTS = ("context", "waves", "campaign", "precision")
WINDOWS = {"train": 36, "validation": 8, "test": 12}
SEEDS = {"train": 71126037, "validation": 71127191, "test": 71128301}
HOUR = 3_600_000
WARMUP = 90 * 24


def _number(value):
    return Decimal(str(value))


def _features(line, candle, i):
    return {k: number(v[i]) for k, v in line.items()} | {"close": number(candle.close)}


def _stack(frame):
    jaw, teeth, lips = (frame[k] for k in ("jaw", "teeth", "lips"))
    return "LONG" if lips > teeth > jaw else "SHORT" if lips < teeth < jaw else "MIXED"


def _tail(values):
    changes = [int(b > a) - int(b < a) for a, b in zip(values, values[1:])]
    sign = changes[-1]
    count = 0
    for change in reversed(changes):
        if sign == 0 or change != sign:
            break
        count += 1
    return sign, count


def _context(operands):
    ac = operands["ac_h1"]
    sign, count = _tail(ac)
    favorable = ac[-1] * sign > 0
    threshold = None if sign == 0 or ac[-1] == 0 else 2 if favorable else 3
    ready = threshold is not None and count >= threshold
    high = [_stack(operands[k]) for k in ("h4", "d1")]
    ao, average = (operands[k] for k in ("oscillator_5_34", "its_sma5"))
    return {
        "ac_side": ("LONG" if sign == 1 else "SHORT") if ready else "NONE",
        "ac_required": threshold,
        "h1_stack": _stack(operands["h1"]),
        "h4_d1_agree": high[0] if high[0] == high[1] and high[0] != "MIXED" else "NONE",
        "course_rhythm": "LONG" if ao > average else "SHORT" if ao < average else "NEUTRAL",
        "ntd_next": "REVIEW_ADD" if ready and operands["first_fractal_fill"] else
                    "WAIT_FIRST_FRACTAL" if ready else "NO_AC_SETUP",
    }


def _geometry(candidate):
    p = [_number(x) for x in candidate["prices_0_to_5"]]
    s = 1 if candidate["first_is_low"] else -1
    w1, w3, w5 = (s * (p[j] - p[j - 1]) for j in (1, 3, 5))
    for failed, name in (
        (min(w1, w3, w5) <= 0, "LEG_DIRECTION"),
        (s * (p[2] - p[0]) <= 0, "W2_ORIGIN"),
        (s * (p[3] - p[1]) <= 0, "W3_EXTENSION"),
        (w3 < min(w1, w5), "W3_SHORTEST"),
        (s * (p[4] - p[1]) <= 0, "W4_OVERLAP"),
    ):
        if failed:
            return name
    return "NONE"


def _waves(operands):
    a, b = (_geometry(operands[k]) for k in ("A", "B"))
    kept = ("A" if a == "NONE" else "") + ("B" if b == "NONE" else "")
    return {
        "compatible": {"": "NONE", "A": "A", "B": "B", "AB": "BOTH"}[kept],
        "reject_a": a,
        "reject_b": b,
        "w4_zero_a": operands["A"]["w4_ao_min"] <= 0 <= operands["A"]["w4_ao_max"],
        "w4_zero_b": operands["B"]["w4_ao_min"] <= 0 <= operands["B"]["w4_ao_max"],
        "squat_verified": False,
        "wave_confirmed": False,
    }


def _campaign(operands):
    state = operands["journal"]
    prices = [[_number(x) for x in bar] for bar in operands["h1_high_low_close"]]
    h, l, close = prices[-1]
    tick = _number(operands["research_tick"])
    out = {"decision": "WAIT", "rule": "TC2.CAMPAIGN", "side": state["side"], "trigger": None, "stop": None}
    if state["stop_fill"]:
        return out | {"decision": "RECONCILE_FLAT"}
    if state["fills"] and not state["stop_ack"]:
        return out | {"decision": "AUDIT_PROTECTION"}
    if state["fills"] == 0:
        prev_h, prev_l, _ = prices[-2]
        lines = [_number(x) for x in operands["h1_visible_mouth"]]
        mid = (h + l) / 2
        bullish = l < prev_l and close > mid and h < min(lines)
        bearish = h > prev_h and close < mid and l > max(lines)
        out["rule"] = "TC2.WM1"
        if state["angulation_review"] and (bullish or bearish):
            out.update(decision="PLAN_STOP_ENTRY", side="LONG" if bullish else "SHORT",
                       trigger=float(h + tick if bullish else l - tick),
                       stop=float(l - tick if bullish else h + tick))
        return out
    long = state["side"] == "LONG"
    sign = 1 if long else -1
    if state["trail_review"]:
        current = _number(state["stop"])
        candidate = min(p[1] for p in prices[-5:]) - tick if long else max(p[0] for p in prices[-5:]) + tick
        if sign * (close - current) <= 0:
            return out | {"decision": "AUDIT_PROTECTION"}
        tighter = sign * (candidate - current) > 0 and sign * (close - candidate) > 0
        return out | {"decision": "REVIEW_TIGHTEN" if tighter else "KEEP_STOP",
                      "stop": float(candidate if tighter else current)}
    if state["fills"] == 1:
        direction, length = _tail(operands["h1_ao"])
        out["rule"] = "TC2.WM2"
        if direction == sign and length == 3 and state["add_risk_ok"]:
            out.update(decision="PLAN_ADD_STOP", trigger=float(h + tick if long else l - tick))
        return out
    out["rule"] = "TC2.WM3"
    window = prices[-5:]
    h0, l0, _ = window[2]
    neighbors = window[:2] + window[3:]
    high = h0 > max(x[0] for x in neighbors)
    low = l0 < min(x[1] for x in neighbors)
    if high != low and (high if long else low) and state["add_risk_ok"]:
        out.update(decision="WATCH_FRACTAL_BREAK", trigger=float(h0 + tick if long else l0 - tick))
    return out


_CONTEXT_QUESTION = (
    "Разбери закрытый участок. В NTD найди направление готовой серии AC или NONE; "
    "ac_required это число нужных столбцов для текущего цвета AC (null для нуля или равных последних). "
    "h1_stack это LONG, SHORT или MIXED по порядку Lips/Teeth/Jaw; h4_d1_agree это совпавшее LONG/SHORT "
    "или NONE. Отдельно определи Course course_rhythm LONG/SHORT/NEUTRAL относительно собственной SMA5. "
    "ntd_next выбери из REVIEW_ADD, WAIT_FIRST_FRACTAL, NO_AC_SETUP. REVIEW_ADD разрешает только "
    "изучить доливку. Учитывай заполненный журнал первой фрактальной сделки."
)
_WAVES_QUESTION = (
    "Сравни два предложенных классических импульса A и B на одном закрытом H1. "
    "prices_0_to_5 это подтверждённые локальные экстремумы, а не доказанная степень. "
    "compatible выбери A, B, BOTH или NONE. Для каждого reject_a/reject_b укажи первое "
    "нарушение в порядке LEG_DIRECTION, W2_ORIGIN, W3_EXTENSION, W3_SHORTEST, W4_OVERLAP либо NONE. "
    "Здесь равенство границе W1 считаем перекрытием, усечение W5 само гипотезу не снимает. "
    "Укажи w4_zero_a/w4_zero_b, наблюдалось ли касание или переход 5/34 через ноль в интервале W4. "
    "Есть биржевой базовый объём, нет канонического тикового объёма и доказанной младшей разметки. "
    "Оцени squat_verified и wave_confirmed."
)
_CAMPAIGN_QUESTION = (
    "Выбери следующий шаг TC2 по закрытому H1 и условному журналу. fills=0,1,2 это число подтверждённых "
    "этапов WM1/WM2, stop_fill это подтверждённое исполнение защиты, stop_ack это сверенная защита. "
    "При первом открытом этапе показанная серия AO началась после заполнения WM1. При trail_review "
    "проверь уровень за последними пятью барами с указанным учебным шагом. Ожидание WM3 требует "
    "свежего однозначного фрактала. Teeth на фактическом пробое будут проверены отдельно. "
    "decision из WAIT, PLAN_STOP_ENTRY, PLAN_ADD_STOP, WATCH_FRACTAL_BREAK, KEEP_STOP, "
    "REVIEW_TIGHTEN, AUDIT_PROTECTION, RECONCILE_FLAT. rule из TC2.WM1, TC2.WM2, TC2.WM3, TC2.CAMPAIGN. "
    "side из LONG, SHORT, NONE. trigger/stop число либо null. Решение описывает наблюдение или план."
)


def _row(identity, rule, edition, page, question, operands, answer, window):
    fields = ",".join(answer) + ",execution_allowed"
    return {"case_id": identity, "rule_id": rule, "edition": edition, "pdf_page": page,
            "question": question + "\n" + canonical(operands).decode() +
                        "\nВерни только JSON с полями " + fields + ". execution_allowed=false.",
            "answer": canonical(answer | {"execution_allowed": False}).decode(),
            "operands": operands, "window": window,
            "label_origin": "CAUSAL_OBSERVATIONS_AND_RESEARCH_POLICY_NOT_HUMAN_GOLD"}


def _window(symbol, raw, i):
    return {"symbol": symbol, "cutoff": iso(raw[i][0]), "prefix_sha256": digest(raw[:i + 1])}


def _from_window(lane, window, rule, edition, page, question, operands, answer):
    return _row(digest(window)[:24] + "-" + lane, rule, edition, page,
                question, operands, answer, window)


def _context_at(raw, candles, hourly, frames, i, sequence):
    cutoff = raw[i][0]
    visible = {k: _features(line, bars[j], j) for k, (closes, bars, line) in frames.items()
               if (j := bisect_right(closes, cutoff) - 1) >= 44}
    if set(visible) != {"h4", "d1"}:
        raise ValueError("CLOSED_HIGHER_TIMEFRAME_WARMUP_REQUIRED")
    ac = [number(x) for x in hourly["ac"][i - 4:i + 1]]
    ao = [number(x) for x in hourly["ao"][i - 4:i + 1]]
    return {"h1": _features(hourly, candles[i], i), **visible, "ac_h1": ac,
            "oscillator_5_34": ao[-1], "its_sma5": number(sum(ao) / 5),
            "first_fractal_fill": sequence % 3 != 0}


def _waves_at(candles, hourly, i, compatible_count):
    start = i - 159
    points = confirmed_price_swings(candles[start:i + 1])[-8:]
    if len(points) < 8:
        return None

    def alternative(pivots):
        lo, hi = (start + pivots[j]["index"] for j in (3, 4))
        oscillator = hourly["ao"][lo:hi + 1]
        return {"bars_ago": [i - start - p["index"] for p in pivots],
                "prices_0_to_5": [p["price"] for p in pivots],
                "first_is_low": pivots[0]["type"] == "L",
                "w4_ao_min": number(min(oscillator)), "w4_ao_max": number(max(oscillator))}

    # Alternatives end at the same visible pivot and vary earlier boundaries.
    alternatives = [alternative(points[:j] + points[j + 2:]) for j in range(5)]
    eligible = [x for x in alternatives if _geometry(x) == "NONE"]
    excluded = [x for x in alternatives if _geometry(x) != "NONE"]
    if len(eligible) < compatible_count or len(excluded) < 2 - compatible_count:
        return None
    a, b = eligible[:compatible_count] + excluded[:2 - compatible_count]
    return {"A": a, "B": b}


def _campaign_at(raw, hourly, i, phase, side, tighter):
    recent = [[r[2], r[3], r[4]] for r in raw[i - 4:i + 1]]
    places = max(0, *(-_number(value).normalize().as_tuple().exponent for bar in recent for value in bar))
    tick = Decimal(1).scaleb(-places)
    fills = (0, 0, 1, 1, 2, 2, 2, 2, 2)[phase]
    close = _number(recent[-1][-1])
    sign = 1 if side == "LONG" else -1
    edge = (min(_number(r[1]) for r in recent) - tick if sign == 1
            else max(_number(r[0]) for r in recent) + tick)
    old = (max(tick, edge - sign * tick * 2) if tighter
           else (close + edge) / 2)
    journal = {"fills": fills, "side": side if fills else "NONE", "stop_ack": fills > 0 and phase != 7,
               "stop_fill": phase == 8, "angulation_review": phase < 2, "trail_review": phase == 6,
               "add_risk_ok": phase in {2, 3, 4, 5}, "stop": str(old) if fills else None}
    return {"journal": journal, "h1_high_low_close": recent, "research_tick": str(tick),
            "h1_visible_mouth": [number(hourly[k][i]) for k in ("jaw", "teeth", "lips")],
            "h1_ao": [number(x) for x in hourly["ao"][i - 4:i + 1]]}


def _ordered_cutoffs(origin, end):
    return (*range(origin, end), *range(WARMUP, origin))


def _chart_rows(symbol, raw, candles, hourly, frames, split, asset):
    seen_waves, seen_campaigns = set(), set()
    n = WINDOWS[split]
    for j in range(n):
        fixed = WARMUP + (len(raw) - WARMUP - 1) * j // n
        seq = asset + j
        context = _context_at(raw, candles, hourly, frames, fixed, seq)
        yield "context", _from_window("context", _window(symbol, raw, fixed), "NTD.COURSE.CONTEXT", "NTD", 75,
                                      _CONTEXT_QUESTION, context, _context(context))
        count = (0, 1, 1, 2)[seq % 4]
        for i in _ordered_cutoffs(fixed, len(raw)):
            if i in seen_waves:
                continue
            operands = _waves_at(candles, hourly, i, count)
            if operands is not None:
                break
        else:
            raise ValueError("MISSING_WAVE_COMPATIBILITY_STRATUM")
        seen_waves.add(i)
        if (seq // 4) % 2:
            operands = {"A": operands["B"], "B": operands["A"]}
        yield "waves", _from_window("waves", _window(symbol, raw, i), "COURSE.TC1.WAVE_ALTERNATIVES", "COURSE", 44,
                                    _WAVES_QUESTION, operands, _waves(operands))

        phase = seq % 9
        side = "LONG" if (asset + j // 9) % 2 == 0 else "SHORT"
        expected_setup = ("PLAN_STOP_ENTRY", "PLAN_ADD_STOP", "WATCH_FRACTAL_BREAK")[phase // 2] if phase < 6 else None
        for i in _ordered_cutoffs(fixed, len(raw)):
            if i in seen_campaigns:
                continue
            plan = _campaign_at(raw, hourly, i, phase, side, (asset + j // 9) % 3 != 0)
            eligible = _campaign(plan)
            if expected_setup is None or eligible["decision"] == expected_setup and eligible["side"] == side:
                break
        else:
            raise ValueError("MISSING_OBSERVED_CAMPAIGN_SETUP_STRATUM")
        seen_campaigns.add(i)
        if phase in {1, 3, 5}:
            plan["journal"]["angulation_review" if phase == 1 else "add_risk_ok"] = False
        yield "campaign", _from_window("campaign", _window(symbol, raw, i), "TC2.CAUSAL_CAMPAIGN", "TC2", 124,
                                       _CAMPAIGN_QUESTION, plan, _campaign(plan))


def build_curriculum(history):
    groups = {}
    data_manifest = {}
    for split, series in history.items():
        rows = {cohort: [] for cohort in COHORTS}
        data_manifest[split] = {}
        for asset, (symbol, raw) in enumerate(sorted(series.items())):
            if any(b[0] - a[0] != HOUR for a, b in zip(raw, raw[1:])):
                raise ValueError("CONTIGUOUS_HOURLY_ARCHIVE_REQUIRED")
            bars = [{"close_time": iso(r[0]), "open": float(r[1]), "high": float(r[2]), "low": float(r[3]),
                     "close": float(r[4]), "volume": float(r[5]), "closed": True} for r in raw]
            candles, gaps = validate_closed_bars(bars, "H1", bars[-1]["close_time"])
            if gaps or len(bars) <= WARMUP + 1:
                raise ValueError("COMPLETE_90_DAY_HOURLY_WARMUP_REQUIRED")
            frames = {}
            for tf in ("H4", "D1"):
                native = aggregate_closed_h1(bars, tf)
                closed, gaps = validate_closed_bars(native, tf, bars[-1]["close_time"])
                if gaps:
                    raise ValueError("HIGHER_TIMEFRAME_GAPS")
                frames[tf.lower()] = ([round(c.at.timestamp() * 1000) for c in closed], closed, indicators(closed))
            hourly = indicators(candles)
            for cohort, row in _chart_rows(symbol, raw, candles, hourly, frames, split, asset):
                rows[cohort].append(row)
            data_manifest[split][symbol] = {"bars": len(raw), "first_close": iso(raw[0][0]),
                "last_close": iso(raw[-1][0]), "canonical_sha256": digest(raw)}
        boundary = boundary_cases(split, seed_override=SEEDS[split], case_namespace="market-precision")
        numeric = restorative_cases(split, seed_override=SEEDS[split] + 1, case_namespace="market-numeric")
        rows["precision"] = [r for r in boundary + numeric if r["rule_id"] in {"TC1.PSY.PLAN", "NTD.AC"}]
        groups[split] = rows
    for first, second in (("train", "validation"), ("validation", "test")):
        if max(x["last_close"] for x in data_manifest[first].values()) >= min(
                x["first_close"] for x in data_manifest[second].values()):
            raise ValueError("GLOBAL_ASSET_AND_TIMEFRAME_SPLIT_OVERLAP")
    manifest = {"market_history": data_manifest, "splits": {
        s: {c: {"questions": len(r), "prompt_sha256": digest(prompts_only(r)),
                "answer_sha256": digest([x["answer"] for x in r])}
            for c, r in rows.items()} for s, rows in groups.items()},
        "expert_gold": False, "observable_setup_strata": True, "future_outcomes_in_tasks": False, "execution_allowed": False}
    return groups, manifest
