"""One bounded correction after an ACTUAL model's unsupported claim.

The independent auditor, not the GGUF, decides whether a claim has observable
support. A second attempt receives ONLY a short, source-scoped explanation of
the first failure, never future OHLC, user secrets, exchange orders or
the expected answer. Both attempts remain visible in the report.

This is in-context SUPERVISED PRACTICE, not updated GGUF weights, a blinded
independent exam or evidence of profitable market prediction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .algorithm_guard import audit_reasoning


# Terse explanations drawn from previously source-reviewed rule cards.
# Each lesson source is the matching model edition; the local GGUF checks that.
LESSONS: dict[str,tuple[str,str]]={
    "WM1_REVERSAL_BAR_NOT_OBSERVED":(
        "TC2.WM1.BAR",
        "A TC2 WM1 requires a real reversal bar visible at the current cutoff; do not invent one."
    ),
    "WM1_REVERSAL_BAR_FROM_PAST_NOT_CURRENT_CANDLE":(
        "TC2.WM1.BAR",
        "A past reversal bar is not a current signal. Recheck the last closed candle time."
    ),
    "WM2_SUPER_AO_NOT_OBSERVED":(
        "TC2.WM2",
        "A TC2 WM2 requires an actually observed same-side Super AO formation; AO direction alone is not WM2."
    ),
    "WM3_NO_PRICE_TOUCH_ON_THIS_CLOSED_BAR":(
        "TC2.WM3",
        "A prior WM3 price touch is not a fresh current-bar breakout. If a confirmed valid fractal is still pending, report PENDING_TRIGGER; otherwise abstain."
    ),
    "WM3_ENTRY_STATE_MUST_BE_PENDING_OR_FRESH_CLOSED_TOUCH":(
        "TC2.WM3",
        "WM3 requires an explicit confirmed pending fractal (PENDING_TRIGGER) or a verified price touch on the last closed bar (CLOSED_BAR_TOUCH_ONLY). Other states do not prove WM3."
    ),
    "WM3_NO_CONFIRMED_FRACTAL_IN_OBSERVATION":(
        "TC2.WM3",
        "A TC2 WM3 needs a confirmed same-side fractal on already closed candles."
    ),
    "WM3_NO_CURRENT_PENDING_VALID_FRACTAL":(
        "TC2.WM3",
        "Do not claim pending entry if the latest valid confirmed fractal is not actually awaiting its trigger."
    ),
    "WM3_NO_VERIFIED_TEETH_PASS_AT_PRICE_TOUCH":(
        "TC2.WM3",
        "At the actual fractal breakout, the price-versus-Teeth filter must pass; formation alone is not enough."
    ),
    "NTD_SIGNAL_NOT_OBSERVED":(
        "NTD.AO",
        "A directional NTD signal must exist in the observed closed data; other source rules cannot replace it."
    ),
    "TC1_FRACTAL_NOT_OBSERVED_ON_CLOSED_BARS":(
        "TC1.FRACTAL",
        "An initiating TC1 fractal must be confirmed on available closed bars and with the proposed side."
    ),
    "FIVE_MAGIC_BULLETS_NOT_ALL_OBSERVED":(
        "TC1.MAGIC.SQUAT",
        "TC1 Squat requires explicit tick volume; Binance base-asset volume must not prove all five bullets."
    ),
}
MALFORMED_CODES={
    "GGUF_EMPTY_DECISION_RESPONSE",
    "GGUF_MALFORMED_JSON_RESPONSE",
    "GGUF_RESPONSE_IS_NOT_AN_OBJECT",
    "GGUF_MISSING_MANDATORY_DECISION_FIELDS",
    "GGUF_CHART_REASONING_OUTPUT_TRUNCATED",
    "GGUF_CHART_STRUCTURED_OUTPUT_TRUNCATED",
    "GGUF_CHART_FINAL_SUMMARY_INVALID",
    "GGUF_RUNTIME_CONTEXT_WINDOW_EXCEEDED",
    "GGUF_RUNTIME_GRAMMAR_ERROR",
    "GGUF_RUNTIME_VALUE_ERROR",
    "MODEL_CONTEXT_BUDGET_EXCEEDED_AFTER_CAUSAL_COMPACTION",
}


def safe_error_code(error:Exception)->str:
    code=str(error)
    return code if code in MALFORMED_CODES else "UNCLASSIFIED_MODEL_ERROR"


@dataclass
class GuidedReadOnlyModel:
    """Wrap original actual GGUF, recording both decisions and corrections."""

    model:Any
    latest:dict[str,Any]|None=None

    def propose_chart_reasoning(self, observation:dict[str,Any], *,
                                edition:str="TC2")->dict[str,Any]:
        self.latest=None
        source_rule_errors=[]
        first_audit=None
        first=None
        error=None
        try:
            first=self.model.propose_chart_reasoning(observation,edition=edition)
            first_audit=audit_reasoning(observation,first)
            source_rule_errors=list(first_audit["hard_failures"])
        except (ValueError,TypeError,KeyError) as exc:
            error=safe_error_code(exc)
            source_rule_errors=[error]
        first_mode=(
            "MODEL_ERROR" if error else
            "SOURCE_UNSUPPORTED" if source_rule_errors else "SOURCE_CONSISTENT"
        )
        result={
            "first_attempt_verdict":first_mode,
            "first_attempt_failures":source_rule_errors,
            "first_attempt_setup":first_audit.get("setup") if first_audit else None,
            "first_attempt_is_safe_abstention":bool(
                first_audit and first_audit.get("setup") in ("NO_SETUP","UNKNOWN")
                and not source_rule_errors
            ),
            "retry_performed":False,
            "retry_verdict":None,
            "retry_setup":None,
            "retry_hard_failures":[],
            "same_closed_cutoff":True,
            "no_future_bars_or_live_orders":True,
            "neural_weights_modified":False,
        }
        if not source_rule_errors:
            self.latest=result
            return first
        rule_cards=[]
        # When the model's prose literally denies its selected JSON setup,
        # the correction must be about the SAME selected book signal, not a
        # generic unrelated indicator. Source edits remain edition-bound.
        if "EXPLANATION_CONTRADICTS_SELECTED_SETUP" in source_rule_errors:
            chosen=first_audit.get("setup") if first_audit else None
            rule_by_setup={
                "WM1":"TC2.WM1.BAR","WM2":"TC2.WM2","WM3":"TC2.WM3",
                "TC1_FRACTAL":"TC1.FRACTAL",
                "TC1_FIVE_MAGIC_BULLETS":"TC1.MAGIC.DIVERGENCE",
                "AO_ZERO_CROSS":"NTD.AO","AO_SAUCER":"NTD.AO",
                "AO_TWIN_PEAKS":"NTD.AO","AC":"NTD.AC",
                "ZONE":"NTD.ZONE","BALANCE_LINE":"NTD.BALANCE",
            }
            rule=rule_by_setup.get(chosen)
            if rule is not None and rule.startswith(edition+"."):
                rule_cards.append({
                    "source":f"{edition}:RULE:{rule}",
                    "excerpt":"Your structured setup and plain explanation contradicted each other. If the confirmed source setup is present, name and explain it consistently; otherwise choose NO_SETUP. Never claim actual execution.",
                })
        for code in source_rule_errors:
            item=LESSONS.get(code)
            if item and item[0].startswith(edition+"."):
                rule_cards.append({
                    "source":f"{edition}:RULE:{item[0]}",
                    "excerpt":item[1],
                })
            if len(rule_cards)>=2:
                break
        # Malformed output: a re-request may help but a magic retry must
        # never be counted as first-try knowledge.
        if not rule_cards:
            if edition=="TC2":rule="TC2.WM1.BAR"
            elif edition=="NTD":rule="NTD.AO"
            elif edition=="TC1":rule="TC1.FRACTAL"
            else:rule="COURSE.RHYTHM"
            rule_cards=[{
                "source":f"{edition}:RULE:{rule}",
                "excerpt":"Return a well-formed source-specific JSON decision only. If observations do not support a concrete setup, explicitly abstain. Never claim live orders."
            }]
        result["retry_performed"]=True
        try:
            second=self.model.propose_chart_reasoning(
                observation,edition=edition,prior_lessons=rule_cards,
            )
            verdict=audit_reasoning(observation,second)
            result["retry_verdict"]=verdict["verdict"]
            result["retry_setup"]=verdict.get("setup")
            result["retry_hard_failures"]=list(verdict["hard_failures"])
            result["retry_is_safe_abstention"]=bool(
                verdict["verdict"]=="PROVISIONAL_REASONING_ONLY"
                and verdict.get("setup") in ("NO_SETUP","UNKNOWN")
            )
            result["retry_supported_substantive_claim"]=bool(
                verdict["verdict"]=="PROVISIONAL_REASONING_ONLY"
                and verdict.get("setup") not in ("NO_SETUP","UNKNOWN")
            )
            self.latest=result
            return second
        except (ValueError,TypeError,KeyError) as exc:
            result["retry_verdict"]="MODEL_ERROR"
            result["retry_hard_failures"]=[safe_error_code(exc)]
            self.latest=result
            # Caller gets explicit error; no fake "safe NO_SETUP" repair.
            raise
