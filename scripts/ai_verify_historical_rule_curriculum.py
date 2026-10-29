"""Independent Decimal/key and raw-market audit; no model or credentials."""
from __future__ import annotations
from decimal import Decimal, ROUND_FLOOR
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from williams_ai.historical_rule_curriculum import (FAMILIES, INTERVALS, canonical,
    cases_from_pack, check_partitions, digest, iso, load_pack, score)


def operands(row):
    return json.loads(row["question"].split("\nВидимые данные: ", 1)[1].split("\n", 1)[0])


def independent_key(row):
    v = operands(row); family = row["rule_id"]
    dec = lambda key: Decimal(str(v[key]))
    if family == "TC1.PSY.PLAN":
        money = dec("equity") * dec("risk_fraction")
        distance = abs(dec("conditional_entry") - dec("conditional_stop"))
        step = dec("research_quantity_step")
        if min(money, distance, step) <= 0:
            raise ValueError("POSITIVE_ENGINEERING_RISK_INPUTS_REQUIRED")
        quantity = (money / distance / step).to_integral_value(rounding=ROUND_FLOOR) * step
        if not quantity * distance <= money < (quantity + step) * distance:
            raise ValueError("RISK_FLOOR_INVARIANT_FAILED")
        answer = {"risk_budget": float(money), "quantity": float(quantity)}
    elif family == "NTD.AC":
        values = list(map(lambda x: Decimal(str(x)), v["ac_oldest_to_latest"]))
        signed = values[-1]
        required = None if signed == 0 else (2 if (signed > 0) == (v["side"] == "LONG") else 3)
        consecutive = 0
        for j in range(len(values)-1, 0, -1):
            change = values[j] - values[j-1]
            if change == 0 or (change > 0) != (v["side"] == "LONG"): break
            consecutive += 1
        answer = {"required_bars": required, "signal_ready": required is not None and consecutive >= required}
    elif family == "NTD.ALLIGATOR":
        jaw, teeth, lips, close = (dec(k) for k in ("visible_jaw", "visible_teeth", "visible_lips", "close"))
        up = lips > teeth and teeth > jaw; down = jaw > teeth and teeth > lips
        answer = {"order": "UP_STACK" if up else "DOWN_STACK" if down else "MIXED",
                  "above_all": all(close > x for x in (jaw, teeth, lips)), "below_all": all(close < x for x in (jaw, teeth, lips))}
    elif family == "NTD.AO":
        before, now = dec("previous_ao"), dec("current_ao")
        answer = {"above_zero": now > 0, "rising": now > before, "crossed_zero": (before < 0 < now) or (now < 0 < before)}
    elif family == "TC2.WM1.TRIGGER":
        lo, hi, tick = dec("last_bar_low"), dec("last_bar_high"), dec("research_tick")
        answer = {"entry": float(hi+tick if v["side"] == "LONG" else lo-tick),
                  "stop": float(lo-tick if v["side"] == "LONG" else hi+tick)}
    elif family == "TC2.WM3":
        points = list(map(Decimal, map(str, v["closed_extrema_oldest_first"])))
        good = len(points) == 5 and v["right_closed"] == 2 and all(
            (points[2] > x if v["kind"] == "HIGH" else points[2] < x) for j, x in enumerate(points) if j != 2)
        answer = {"confirmed": good}
    elif family == "COURSE.AO5345":
        difference = dec("histogram") - dec("own_sma5")
        answer = {"momentum": "NEUTRAL" if difference == 0 else "LONG" if difference > 0 else "SHORT"}
    elif family == "COURSE.HUMP":
        answer = {"hump_valid": v["second_formation_inclusive_start"] - v["first_formation_inclusive_end"] > 0}
    elif family == "COURSE.W4.ZERO":
        a, b = dec("previous_5_34"), dec("current_5_34")
        answer = {"zero_transition_seen": (a < 0 < b) or (b < 0 < a), "w4_finished": False}
    else: raise ValueError("UNKNOWN_HISTORICAL_RULE")
    return {**answer, "execution_allowed": False}


def verify_market_observations(row, dataset):
    end = next(i for i, r in enumerate(dataset["bars"]) if iso(r[0]) == row["cutoff"])
    raw = dataset["bars"][end-219:end+1]
    if len(raw) != 220 or digest({"symbol": dataset["symbol"], "interval": dataset["interval"], "bars": raw}) != row["window_sha256"]:
        raise ValueError("CLOSED_WINDOW_HASH_OR_CUTOFF_MISMATCH")
    med = [(Decimal(r[2])+Decimal(r[3]))/2 for r in raw]
    ao = {i: sum(med[i-4:i+1])/5 - sum(med[i-33:i+1])/34 for i in range(33, 220)}
    ac = {i: ao[i]-sum(ao[j] for j in range(i-4, i+1))/5 for i in range(37, 220)}
    v = operands(row); family = row["rule_id"]
    def near(a, b):
        if not math.isclose(float(a), float(b), rel_tol=8e-8, abs_tol=2e-8):
            raise ValueError("OBSERVATION_NOT_DERIVED_FROM_THE_RAW_CLOSED_MARKET")
    if family == "NTD.AC":
        for got, i in zip(v["ac_oldest_to_latest"], range(212, 220)): near(got, ac[i])
    elif family in {"NTD.AO", "COURSE.W4.ZERO"}:
        names = ("previous_ao", "current_ao") if family == "NTD.AO" else ("previous_5_34", "current_5_34")
        for name, i in zip(names, (218, 219)): near(v[name], ao[i])
    elif family == "COURSE.AO5345":
        near(v["histogram"], ao[219]); near(v["own_sma5"], sum(ao[i] for i in range(215, 220))/5)
    elif family == "NTD.ALLIGATOR":
        for name, period, shift in (("visible_jaw", 13, 8), ("visible_teeth", 8, 5), ("visible_lips", 5, 3)):
            value = sum(med[:period])/period
            for i in range(period, 220-shift): value = (value*(period-1)+med[i])/period
            near(v[name], value)
        near(v["close"], raw[-1][4])
    elif family in {"TC1.PSY.PLAN", "TC2.WM1.TRIGGER"}:
        if family == "TC2.WM1.TRIGGER":
            near(v["last_bar_low"], raw[-1][3]); near(v["last_bar_high"], raw[-1][2])
        else:
            side = v["conditional_side"]
            tick = Decimal("0.1" if dataset["symbol"] in {"BTCUSDT", "ETHUSDT", "BNBUSDT"} else "0.001")
            hi, lo = Decimal(raw[-1][2]), Decimal(raw[-1][3])
            near(v["conditional_entry"], hi+tick if side == "LONG" else lo-tick)
            near(v["conditional_stop"], lo-tick if side == "LONG" else hi+tick)
    elif family == "TC2.WM3":
        center = 217 if v["right_closed"] == 2 else 218
        column = 2 if v["kind"] == "HIGH" else 3
        target = [float(r[column]) for r in raw[center-2:min(center+3, 220)]]
        if target != v["closed_extrema_oldest_first"]:
            raise ValueError("FRACTAL_NEIGHBORS_NOT_FROM_VISIBLE_CANDLES")
    elif family == "COURSE.HUMP":
        a = v["first_formation_inclusive_end"] - 2
        b = v["second_formation_inclusive_start"] + 2
        if not 2 <= a < b <= 217 or v["first_kind"] == v["second_kind"]:
            raise ValueError("CONFIRMED_OPPOSITE_FORMATIONS_REQUIRED")
        for center, kind in ((a, v["first_kind"]), (b, v["second_kind"])):
            column = 2 if kind == "HIGH" else 3
            target = Decimal(raw[center][column])
            for j in (center-2, center-1, center+1, center+2):
                if not (target > Decimal(raw[j][column]) if kind == "HIGH" else target < Decimal(raw[j][column])):
                    raise ValueError("STRICT_FRACTAL_FORMATION_WAS_NOT_CONFIRMED")


def audit(root: Path):
    packs = {s: load_pack(root/(s+".json.gz")) for s in ("train", "validation", "test")}
    ranges = check_partitions(packs); report = {}
    all_ids, all_prompts = set(), set()
    for split, pack in packs.items():
        rows = cases_from_pack(pack)
        datasets = {(r["symbol"], INTERVALS[r["interval"]][2]): r for r in pack["datasets"]}
        for row in rows:
            if independent_key(row) != json.loads(row["answer"]):
                raise ValueError("INDEPENDENT_VISIBLE_OPERAND_KEY_MISMATCH")
            verify_market_observations(row, datasets[(row["symbol"], row["timeframe"])])
            if row["case_id"] in all_ids or row["question"] in all_prompts:
                raise ValueError("MARKET_CASE_REPLAY_ACROSS_PARTITIONS")
            all_ids.add(row["case_id"]); all_prompts.add(row["question"])
        report[split] = {"verified_keys": len(rows), "raw_closed_market_observations_verified": len(rows)}
    return {"partitions": ranges, "splits": report, "independent_decimal_checks": True,
        "expert_wave_or_profitability_claim": False, "execution_allowed": False}


if __name__ == "__main__":
    print(json.dumps(audit(Path(sys.argv[1] if len(sys.argv)>1 else "examples/historical_rule_20261029")), ensure_ascii=False, indent=2))
