"""Bill Williams TC2: source-grounded causal lifecycle lesson for local GGUF.

Research-only. Produces a *teacher's policy and counterfactual review* of
entry, management and exit decisions: it never produces executable orders.
Derived from the user's private TC2 source review (chapters 9--12) and the
page-reviewed rule cards. The first edition, NTD and scanned Course have
distinct policies and are NOT silently treated as TC2.

Only closed, gap-free OHLCV is supplied. Signal / price-touch / exchange-fill
and protective-stop ACK are mutually different levels of evidence.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

from .book_rules import registry
from .campaign_audit import audit_campaign
from .chart_reader import validate_closed_bars, analyze_chart


RULE_IDS = ("TC2.WM1.BAR", "TC2.WM1.ANG", "TC2.WM1.TRIGGER",
            "TC2.WM1.STOP", "TC2.WM2", "TC2.WM3", "TC2.REVERSE",
            "TC2.CAMPAIGN")
PYRAMID_EXAMPLE = (1, 5, 4, 3, 2)  # Book's ILLUSTRATION, not live sizing.
_ALLOWED_STATES = ("AWAITING_PRICE_TRIGGER",
                   "CLOSED_BAR_TOUCH_PASSES_TEETH_NOT_EXCHANGE_FILL")


def _decimal(value: Any) -> Decimal:
    if isinstance(value, bool):
        raise ValueError("INVALID_TC2_PRICE_OR_TICK")
    try:
        out = Decimal(str(value))
    except (ValueError, TypeError, InvalidOperation) as exc:
        raise ValueError("INVALID_TC2_PRICE_OR_TICK") from exc
    if not out.is_finite() or out <= 0:
        raise ValueError("INVALID_TC2_PRICE_OR_TICK")
    return out


def _no_trade() -> dict[str, Any]:
    return {"mode": "TC2_BOOK_GROUNDED_PAPER_LIFECYCLE_TUTOR",
            "edition": "TC2", "trade_action": None,
            "execution_allowed": False, "auto_orders": False,
            "book_rules_do_not_authenticate_exchange_orders": True,
            "model_weights_modified": False}


def teach_tc2_trade_lifecycle(
    observation: dict[str, Any], bars: list[dict[str, Any]],
    *, tick_size: float, journal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Explain three Wise Men, optional single-campaign SL and exit discipline.

    `observation` MUST be a independently reconstructed chart_reader view.
    A passed journal is a *claimed* fill/ACK sequence, not verified broker data.
    Training-only stop references never authorize moving a real stop.

    Training states:
       flat => plan eligible observations, await price trigger;
       open + missing SL ACK => HALT_FOR_PROTECTION_AUDIT;
       open + confirmed journal SL => review trailing (never widen);
       opposite signal without independently filled reverse => HOLD/REVIEW.
    """
    if not isinstance(observation, dict) or observation.get("execution_allowed") is not False:
        raise ValueError("INDEPENDENT_READ_ONLY_CHART_REQUIRED")
    if observation.get("gaps") != 0 or observation.get("no_lookahead") is not True:
        raise ValueError("NON_CAUSAL_OR_GAPPED_CHART")
    symbol = observation.get("symbol")
    timeframe = observation.get("timeframe")
    as_of = observation.get("as_of")
    candles, gaps = validate_closed_bars(bars, timeframe, as_of)
    if (gaps or len(candles) < 45 or
            candles[-1].at.isoformat() != as_of or
            observation.get("last_closed_at") != as_of or
            not isinstance(symbol, str) or not symbol.endswith("USDT")):
        raise ValueError("CHART_CUTOFF_OR_PRICE_HISTORY_UNVERIFIED")
    tick = _decimal(tick_size)
    # Reject invented signal candidates and indicator context. The tutor must
    # not simply believe a pasted GGUF/chart JSON: recompute the exact
    # cutoff with the same causal, displaced-Alligator engine.
    rechecked = analyze_chart(symbol, timeframe, bars, as_of=as_of,
                              tick_size=float(tick))
    if (rechecked["gaps"] or
            observation.get("candidates") != rechecked["candidates"] or
            observation.get("evidence") != rechecked["evidence"]):
        raise ValueError("TC2_LIFECYCLE_CANDIDATES_NOT_INDEPENDENTLY_REPRODUCED")
    known = {card["id"]:card for card in registry()["rule_cards"]
             if card["source"] == "TC2" and card["status"] == "REVIEWED_PAGES"}
    if any(rule not in known for rule in RULE_IDS):
        raise ValueError("SOURCE_RULE_CARD_NOT_PAGE_REVIEWED")
    cards = [{"rule_id":name, "pdf_pages":known[name]["pdf_pages"]}
             for name in RULE_IDS]
    candidates = observation.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("MALFORMED_CLOSED_BAR_SIGNALS")
    actions = []
    for raw in candidates:
        if not isinstance(raw, dict):
            continue
        side = raw.get("side")
        if side not in ("LONG", "SHORT"):
            continue
        kind = raw.get("type")
        if kind not in ("WM1", "WM2_SUPER_AO", "WM3_FRACTAL"):
            continue
        item = {"side":side, "signal":("WM1" if kind=="WM1" else
                "WM2" if kind=="WM2_SUPER_AO" else "WM3"),
                "setup_seen_at":as_of, "status":"OBSERVATION_ONLY",
                "price_breakout_requires_next_observation":True,
                "exchange_fill_known":False}
        if kind == "WM1":
            if raw.get("bar_time") != as_of or raw.get("angle_heuristic") is not True:
                continue
            item["status"] = "REVERSAL_BAR_HUMAN_ANGLULATION_REVIEW_REQUIRED"
            item["source_rule_ids"] = ["TC2.WM1.BAR","TC2.WM1.ANG",
                                       "TC2.WM1.TRIGGER","TC2.WM1.STOP"]
            last = candles[-1]
            item["breakout_price_reference"] = str(
                Decimal(str(last.high)) + tick if side=="LONG"
                else Decimal(str(last.low)) - tick)
            item["first_sl_reference"] = str(
                Decimal(str(last.low)) - tick if side=="LONG"
                else Decimal(str(last.high)) + tick)
        elif kind == "WM2_SUPER_AO":
            if raw.get("state") != "MOMENTUM_CANDIDATE_NOT_ORDER":
                continue
            item["status"] = "THIRD_AO_MOMENTUM_BAR_AWAIT_ACTUAL_PRICE_BREAK"
            item["source_rule_ids"] = ["TC2.WM2"]
            last=candles[-1]
            item["breakout_price_reference"] = str(
                Decimal(str(last.high)) + tick if side=="LONG"
                else Decimal(str(last.low)) - tick)
            # 3--5 recently CLOSED bars, not a magically known wave extremum.
            recent=candles[-5:]
            item["first_sl_reference"] = str(
                min(Decimal(str(b.low)) for b in recent)-tick if side=="LONG"
                else max(Decimal(str(b.high)) for b in recent)+tick)
            item["stop_basis"] = "RECENT_FIVE_CLOSED_BARS_TC2_PROXY"
        elif kind == "WM3_FRACTAL":
            from .decision_contract import legal_wm3_discussion_states
            legal=legal_wm3_discussion_states({
                "as_of":as_of,"candidates":[raw],
                "execution_allowed":False,"gaps":0})
            states={x["entry_state"] for x in legal}
            if "PENDING_TRIGGER" in states:
                item["status"]="CONFIRMED_T_PLUS_2_FRACTAL_AWAIT_PRICE_BREAK"
            elif "CLOSED_BAR_TOUCH_ONLY" in states:
                item["status"]="CLOSED_BAR_TOUCH_NOT_EXCHANGE_FILL"
            else:
                continue
            item["source_rule_ids"]=["TC2.WM3"]
            if not isinstance(raw.get("entry_trigger"),(int,float)):
                continue
            item["breakout_price_reference"]=str(_decimal(raw["entry_trigger"]))
            item["teeth_condition_requires_price_trigger_validation"]=True
        # Different directions may be under observation at once; without
        # valid real price triggers, not even a *flat* account takes orders.
        actions.append(item)

    state = "FLAT_READ_ONLY_STUDY"
    management: dict[str, Any] = {
        "status":"AWAIT_VERIFIED_BROKER_CAMPAIGN_STATE",
        "source_rule_ids":["TC2.CAMPAIGN","TC2.REVERSE"],
        "opposite_wise_man_signal_alone_is_not_an_exit":True,
        "stop_must_never_be_widened":True,
        "unconfirmed_wave_hypotheses_are_not_exit_orders":True,
        "stop_review_context":None,
    }
    audit = None
    if journal is not None:
        if (not isinstance(journal,dict)
                or journal.get("symbol")!=symbol
                or journal.get("edition")!="TC2"
                or journal.get("as_of")!=as_of):
            raise ValueError("TC2_JOURNAL_CHART_CUTOFF_MISMATCH")
        audit=audit_campaign(journal)
        critical=[x["code"] for x in audit["findings"]
                  if x["severity"]=="CRITICAL"]
        management.update({"journal_only_not_broker_verified":True,
                           "campaign_audit_critical_findings":critical})
        if audit["position"]["status"]=="OPEN":
            state = "CLAIMED_OPEN_POSITION_REQUIRE_EXCHANGE_RECONCILIATION"
            side=audit["position"]["side"]
            sl=audit["position"]["stop_price"]
            opposing=[x["signal"] for x in actions if x["side"]!=side]
            management.update({
                "side":side, "opposing_signals_observed_only":opposing,
            })
            if sl is None or critical:
                management["status"]="HALT_FOR_PROTECTION_OR_FILL_AUDIT"
                management["trailing_stop_paper_reference"]=None
            else:
                old_sl=_decimal(sl)
                # TC2 book describes protecting a profitable campaign with
                # a stop behind the last 3--5 bars. Use ONLY closed bars and
                # show a 5-bar conservative illustration, no order instruction.
                window=candles[-5:]
                candidate=(min(Decimal(str(b.low)) for b in window)-tick
                           if side=="LONG" else
                           max(Decimal(str(b.high)) for b in window)+tick)
                current_price=Decimal(str(candles[-1].close))
                tighter=(candidate>old_sl if side=="LONG" else candidate<old_sl)
                valid=(candidate<current_price if side=="LONG"
                       else candidate>current_price)
                current_valid=(old_sl<current_price if side=="LONG" else old_sl>current_price)
                management["status"]=("HALT_FOR_STOP_RECONCILIATION" if not current_valid else
                    "REVIEW_MONOTONIC_TRAILING_STOP"
                    if tighter and valid else "HOLD_PROTECTIVE_STOP")
                management["trailing_stop_paper_reference"]=(
                    str(candidate) if tighter and valid and current_valid else None)
                management["existing_journal_stop"]=str(old_sl)
                management["opposite_signal_alone_must_not_flatten"]=True
                if candidate>0:
                    from .stop_review import SCHEMA, SOURCE, normalize_context
                    # Preserve even a rejected candidate as an observation.
                    # The LLM must compare it; never provide the teacher's
                    # expected assessment as the model's own answer.
                    management["stop_review_context"]=normalize_context({
                        "schema":SCHEMA,"symbol":symbol,"as_of":as_of,
                        "side":side,"current_stop":str(old_sl),
                        "candidate_stop":str(candidate),
                        "last_closed_price":str(current_price),"source":SOURCE,
                        "execution_allowed":False,"trade_action":None})
        else:
            state="CLAIMED_FLAT_REQUIRES_EXCHANGE_RECONCILIATION"
            management["status"]=("HALT_FOR_CAMPAIGN_AUDIT" if critical else
                                  "NO_OPEN_POSITION_IN_SUPPLIED_JOURNAL")
    # The numbers 1--5--4--3--2 are a book example, NEVER mandatory live
    # contracts: hard portfolio risk and confirmed improving trade must gate
    # every add. This teaching artifact is not a position-size proposal.
    return {**_no_trade(),
            "symbol":symbol,"timeframe":timeframe,"as_of":as_of,
            "source_rule_cards":cards,
            "entry_observations":actions[-12:],
            "campaign_state":state,
            "management":management,
            "inverse_pyramid_book_illustration":list(PYRAMID_EXAMPLE),
            "pyramiding_requires": [
                "ACTUAL_SAME_DIRECTION_PRICE_TRIGGER_AND_BROKER_FILL",
                "OPEN_CAMPAIGN_ALREADY_PROFITABLE_AFTER_COSTS",
                "CONFIRMED_PROTECTIVE_STOP_AND_SHARED_PORTFOLIO_RISK_ROOM",
                "NO_AVERAGING_INTO_LOSS",
            ],
            "exit_requires_one_of":[
                "AUTHENTICATED_PROTECTIVE_SL_FILL",
                "AUTHENTICATED_OPPOSITE_STOP_REVERSE_TRIGGER_AND_FILL",
                "SEPARATELY_AUTHORIZED_STRUCTURAL_OR_EMERGENCY_EXIT",
            ],
            "historical_wave_degree_verified":False,
            "no_lookahead":True,
            "journal_ack_flags_not_broker_authenticated":True,
            "not_an_entry_exit_or_sl_command":True}
