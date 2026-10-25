"""Typed, read-only stop reasoning and an independent numerical critic.

The source chapter describes a campaign; monotonic protection is a separate
engineering risk invariant. A journal claim is never exchange verification.
No decision from this module can replace/cancel a stop or authorize an order.
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone
from hashlib import sha256
import json
from typing import Any

from .decision_contract import _as_utc

SCHEMA = "TC2_CLOSED_BAR_STOP_REVIEW_V1"
SOURCE = "SUPPLIED_JOURNAL_AND_CLOSED_BARS_NOT_BROKER_AUTHENTICATED"
INPUT_KEYS = {"schema", "symbol", "as_of", "side", "current_stop",
              "candidate_stop", "last_closed_price", "source",
              "execution_allowed", "trade_action"}
OUTPUT_KEYS = {"level_relation", "risk_direction", "candidate_protects_side",
               "assessment", "reason"}
RISK_POLICY = (
    "Independent engineering risk policy, not a quotation from a book: "
    "LONG protection may only move UP; SHORT protection may only move DOWN. "
    "A lower LONG stop or higher SHORT stop increases risk. Equal means no change. "
    "A LONG stop must remain BELOW the observed price; a SHORT stop ABOVE it. "
    "If the EXISTING stop is already at/beyond that price, request reconciliation: "
    "a closed candle cannot prove whether an exchange stop filled. "
    "Never cancel protection or infer broker acknowledgements from these facts."
)


def price(value: Any) -> Decimal:
    if not isinstance(value, str) or not 1 <= len(value) <= 64:
        raise ValueError("STOP_REVIEW_REQUIRES_DECIMAL_STRINGS")
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("STOP_REVIEW_INVALID_PRICE") from exc
    if (not number.is_finite() or number <= 0 or
            not -18 <= number.adjusted() <= 18 or len(number.as_tuple().digits) > 36):
        raise ValueError("STOP_REVIEW_INVALID_OR_UNBOUNDED_PRICE")
    return number


def normalize_context(value: Any) -> dict[str, Any]:
    if (not isinstance(value, dict) or set(value) != INPUT_KEYS
            or value.get("schema") != SCHEMA or value.get("source") != SOURCE
            or value.get("execution_allowed") is not False
            or value.get("trade_action") is not None
            or not isinstance(value.get("side"),str)
            or value.get("side") not in {"LONG", "SHORT"}):
        raise ValueError("INVALID_READ_ONLY_STOP_REVIEW_CONTEXT")
    symbol = value["symbol"]
    if (not isinstance(symbol, str) or not symbol.isascii() or
            not symbol.endswith("USDT") or not symbol.isalnum()
            or not 5 <= len(symbol) <= 24 or symbol != symbol.upper()):
        raise ValueError("INVALID_STOP_REVIEW_SYMBOL")
    cutoff = _as_utc(value["as_of"])
    if cutoff is None:
        raise ValueError("INVALID_STOP_REVIEW_CUTOFF")
    out = {**value, "as_of":datetime.fromisoformat(
        value["as_of"].replace("Z","+00:00")).astimezone(timezone.utc).isoformat()}
    for key in ("current_stop", "candidate_stop", "last_closed_price"):
        out[key] = format(price(value[key]), "f")
    return out


def _response_shape(value: Any) -> bool:
    return (isinstance(value, dict) and set(value) == OUTPUT_KEYS and
        isinstance(value.get("level_relation"), str) and
        value["level_relation"] in {"LOWER","HIGHER","EQUAL","UNKNOWN"} and
        isinstance(value.get("risk_direction"), str) and
        value["risk_direction"] in {"TIGHTER","WIDER","UNCHANGED","UNKNOWN"} and
        isinstance(value.get("assessment"), str) and
        value["assessment"] in {"KEEP_PROTECTION","REVIEW_TIGHTENING","HALT_FOR_RECONCILIATION","UNKNOWN"} and
        type(value.get("candidate_protects_side")) is bool and
        isinstance(value.get("reason"), str) and 1<=len(value["reason"])<=300)


def _call_metadata(model: Any) -> dict[str, Any] | None:
    raw=getattr(model,"latest_stop_review_call",None)
    if not isinstance(raw,dict) or raw.get("mode")!="SCHEMA_CONSTRAINED_STOP_REVIEW":
        return None
    usage=raw.get("usage",{})
    return {"mode":"SCHEMA_CONSTRAINED_STOP_REVIEW",
        "finish_reason":raw.get("finish_reason") if raw.get("finish_reason") in {"stop","length"} else "UNKNOWN",
        "usage":{key:usage[key] for key in ("prompt_tokens","completion_tokens","total_tokens")
            if isinstance(usage,dict) and type(usage.get(key)) is int and usage[key]>=0},
        "case_specific_answer_enums_supplied":False if raw.get("case_specific_answer_enums_supplied") is False else None}


def audit_stop_review(context: dict[str, Any], response: Any) -> dict[str, Any]:
    context = normalize_context(context)
    old, new, close = (price(context[k]) for k in
                       ("current_stop", "candidate_stop", "last_closed_price"))
    relation = "HIGHER" if new > old else "LOWER" if new < old else "EQUAL"
    tighter = new > old if context["side"] == "LONG" else new < old
    direction = "UNCHANGED" if new == old else "TIGHTER" if tighter else "WIDER"
    protects = new < close if context["side"] == "LONG" else new > close
    current_protects = old < close if context["side"] == "LONG" else old > close
    expected = {"level_relation":relation, "risk_direction":direction,
                "candidate_protects_side":protects,
                "assessment":("HALT_FOR_RECONCILIATION" if not current_protects else
                    "REVIEW_TIGHTENING" if tighter and protects else "KEEP_PROTECTION")}
    failures = []
    shape = _response_shape(response)
    if not shape:
        failures.append("INVALID_MODEL_STOP_REVIEW_FIELDS")
    else:
        for key, reference in expected.items():
            if response[key] != reference:
                failures.append("STOP_REVIEW_MISMATCH_" + key.upper())
        if response["assessment"] == "REVIEW_TIGHTENING":
            if direction == "WIDER":
                failures.append("MODEL_ATTEMPTED_TO_WIDEN_STOP")
            if not protects:
                failures.append("MODEL_CANDIDATE_ON_WRONG_PRICE_SIDE")
        if not current_protects and response["assessment"] != "HALT_FOR_RECONCILIATION":
            failures.append("MODEL_IGNORED_POSSIBLY_TRIGGERED_STOP")
    return {"verdict":"NUMERIC_RESEARCH_CONSISTENT" if not failures else "UNSUPPORTED_STOP_REASONING",
            "hard_failures":failures, "reference":expected,
            "reason_text_not_semantically_verified":True,
            "broker_authentication_performed":False,
            "execution_allowed":False, "trade_action":None}


def review_with_model(model: Any, context: dict[str, Any]) -> dict[str, Any]:
    context = normalize_context(context)
    digest = sha256(json.dumps(context, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    response = None
    error = None
    if not callable(getattr(model, "review_protective_stop", None)):
        error = "MODEL_STOP_REVIEW_INTERFACE_UNAVAILABLE"
    else:
        try:
            response = model.review_protective_stop(deepcopy(context))
        except (ValueError, TypeError, KeyError, RuntimeError):
            error = "MODEL_STOP_REVIEW_CALL_FAILED"
    audit = audit_stop_review(context, response)
    if error:
        audit["hard_failures"] = [error]
        audit["verdict"] = "MODEL_STOP_REVIEW_ERROR"
    # Invalid output is not repaired into a plausible safe answer or exported
    # wholesale (it could contain arbitrary fields/private runtime strings).
    safe_response = response if _response_shape(response) and not error else None
    return {"mode":"INDEPENDENT_NUMERIC_STOP_RESEARCH_REVIEW",
            "context_sha256":digest, "cutoff":context["as_of"],
            "model_response":safe_response, "audit":audit,
            "source_rule_id":"TC2.CAMPAIGN",
            "risk_policy_id":"MONOTONIC_PROTECTION_V1",
            "model_call_metadata":_call_metadata(model),
            "needs_source_or_reasoning_review":bool(audit["hard_failures"]),
            "model_weights_updated":False, "execution_allowed":False,
            "trade_action":None}


def diagnostic_cases() -> list[dict[str, Any]]:
    # Frozen synthetic arithmetic examples, NOT historical market holdouts.
    rows = (
        ("LONG_WIDER", "LONG", "100", "97", "125"),
        ("SHORT_WIDER", "SHORT", "110", "113", "90"),
        ("LONG_TIGHTER", "LONG", "100", "103", "125"),
        ("SHORT_TIGHTER", "SHORT", "110", "108", "90"),
        ("LONG_EQUAL", "LONG", "0.00100", "0.00100", "0.00125"),
        ("SHORT_EQUAL", "SHORT", "0.00110", "0.00110", "0.00090"),
        ("LONG_WRONG_SIDE", "LONG", "100", "130", "125"),
        ("SHORT_WRONG_SIDE", "SHORT", "110", "85", "90"),
        ("LONG_EXISTING_CROSSED", "LONG", "130", "140", "125"),
        ("SHORT_EXISTING_CROSSED", "SHORT", "85", "80", "90"),
    )
    return [{"id":ident, "context":normalize_context({
        "schema":SCHEMA, "symbol":"BTCUSDT", "as_of":"2026-10-20T00:59:59.999000+00:00",
        "side":side, "current_stop":old, "candidate_stop":new,
        "last_closed_price":close, "source":SOURCE,
        "execution_allowed":False, "trade_action":None})}
        for ident, side, old, new, close in rows]


def diagnose_model(model: Any) -> dict[str, Any]:
    results = []
    for case in diagnostic_cases():
        result = review_with_model(model, case["context"])
        results.append({"id":case["id"], **result})
    return {"mode":"ACTUAL_MODEL_TYPED_STOP_RISK_DIAGNOSTIC",
            "cases":len(results), "consistent":sum(not r["audit"]["hard_failures"] for r in results),
            "results":results, "synthetic_arithmetic_not_market_charts":True,
            "explicit_risk_policy_supplied_not_unassisted_exam":True,
            "different_protocol_not_comparable_to_previous_mcq_score":True,
            "schema_constrained_decoder_not_a_correctness_oracle":True,
            "expected_case_answers_not_supplied_to_model":True,
            "model_weights_updated":False, "market_profitability_proven":False,
            "execution_allowed":False, "trade_action":None}


def main() -> None:
    import argparse
    from pathlib import Path
    from .local_model import LocalWilliamsModel
    from .real_gguf_acceptance import actual_gguf_sha256
    parser = argparse.ArgumentParser(description="Actual model numeric stop review, no orders")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--reasoning-budget", type=int, default=1536)
    args = parser.parse_args()
    actual_gguf_sha256(args.model, args.expected_sha256)
    model = LocalWilliamsModel(args.model, n_ctx=args.ctx,
        learning_mode="auto", reasoning_budget=args.reasoning_budget)
    print(json.dumps(diagnose_model(model), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
