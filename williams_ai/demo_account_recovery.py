"""Read-only, authenticated Binance Futures DEMO recovery diagnostics.

This checks the exchange after a phone/process crash or uncertain submission.
It intentionally NEVER removes the journal lock, cancels stops or sends orders.
An open protective stop on the account cannot be attributed to our pilot by
symbol/side alone: ownership and a full exchange fill audit are still required.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

from .demo_futures import DemoSafetyError
from .demo_pilot_readiness import PAIR, OneCampaignReservation


def _position_decimal(v: Any) -> Decimal:
    if isinstance(v, bool):
        raise DemoSafetyError("DEMO_POSITION_AMOUNT_INVALID")
    try:
        amount = Decimal(str(v))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise DemoSafetyError("DEMO_POSITION_AMOUNT_INVALID") from exc
    if not amount.is_finite():
        raise DemoSafetyError("DEMO_POSITION_AMOUNT_INVALID")
    return amount


def inspect_demo_after_restart(client: Any, journal: OneCampaignReservation) -> dict[str, Any]:
    """ONLY signed GETs; no guesswork and no implicit authority to trade.

    client must be a dedicated DEMO-only DemoHttp. The persisted journal
    preserves an UNKNOWN effect until a human reconciles order identities.
    """
    reserved = journal.status().get("reservations")
    if not isinstance(reserved, list) or len(reserved) > 1:
        raise DemoSafetyError("AMBIGUOUS_PILOT_JOURNAL")
    result: dict[str, Any] = {
        "mode": "BINANCE_FUTURES_DEMO_SIGNED_READONLY_RECOVERY",
        "venue": "BINANCE_FUTURES_DEMO",
        "symbol": PAIR,
        "journal_unchanged": True,
        "manual_exchange_reconciliation_required": True,
        "stop_acknowledged": False,
        "execution_allowed": False,
        "trade_action": None,
    }
    if not reserved:
        return {**result, "status": "NO_RESERVED_PILOT_DO_NOT_ASSUME_FLAT",
                "authenticated_checks_performed": False}
    row = reserved[0]
    if (not isinstance(row, dict) or row.get("symbol") != PAIR or
            row.get("state") != "UNKNOWN_UNTIL_EXCHANGE_RECONCILED" or
            row.get("side") not in ("BUY", "SELL")):
        raise DemoSafetyError("PILOT_JOURNAL_INCONSISTENT")
    expected = _position_decimal(row.get("quantity"))
    if expected <= 0:
        raise DemoSafetyError("PILOT_JOURNAL_INVALID_QUANTITY")
    # The current supervised CLI atomically seals deterministic stop + entry
    # client IDs before submitting any POST. Legacy rows have no IDs and
    # cannot prove that a merely similar stop belongs to our attempt.
    bound = journal.bound_client_ids()
    expected_stop_price = (_position_decimal(bound["expected_stop_price"])
                           if bound is not None else None)
    # Neither the LLM nor any untrusted snapshot can supply alternate API
    # paths, symbols, keys, account identities or exchange host.
    side_mode = client.request("GET", "/fapi/v1/positionSide/dual", signed=True)
    if not isinstance(side_mode, dict) or side_mode.get("dualSidePosition") is not False:
        raise DemoSafetyError("DEMO_POSITION_MODE_NOT_ONE_WAY")
    positions = client.request("GET", "/fapi/v3/positionRisk",
                               {"symbol": PAIR}, signed=True)
    if (not isinstance(positions, list) or len(positions) != 1 or
            not isinstance(positions[0], dict) or
            positions[0].get("symbol") != PAIR or
            positions[0].get("positionSide") != "BOTH"):
        raise DemoSafetyError("DEMO_POSITION_IDENTITY_UNVERIFIED")
    amount = _position_decimal(positions[0].get("positionAmt"))
    # Current ALL open conditional orders; do not mistake ordinary openOrders
    # or a pre-entry ACK for the durable protective algo STOP_MARKET.
    open_algos = client.request("GET", "/fapi/v1/openAlgoOrders",
                                {"symbol": PAIR}, signed=True)
    if not isinstance(open_algos, list) or len(open_algos) > 200:
        raise DemoSafetyError("DEMO_OPEN_ALGO_ORDER_LIST_UNTRUSTED")
    protective_side = "SELL" if row["side"] == "BUY" else "BUY"
    candidates = 0
    bound_matches = 0
    for algo in open_algos:
        if not isinstance(algo, dict) or algo.get("symbol") != PAIR:
            raise DemoSafetyError("DEMO_ALGO_ORDER_IDENTITY_UNVERIFIED")
        same_id = (bound is not None and
                   algo.get("clientAlgoId") == bound["stop_client_algo_id"])
        valid = (algo.get("side") == protective_side
                 and algo.get("orderType") == "STOP_MARKET"
                 and algo.get("positionSide") == "BOTH"
                 and algo.get("closePosition") is True
                 and algo.get("workingType") == "MARK_PRICE"
                 and algo.get("algoStatus") == "NEW")
        # An identical clientAlgoId with unexpected stop/side/status is a
        # critical conflict: do not "select" another order as protection.
        if same_id:
            if (not valid or _position_decimal(algo.get("triggerPrice"))
                    != expected_stop_price):
                raise DemoSafetyError("BOUND_DEMO_STOP_ID_HAS_WRONG_PARAMETERS")
            bound_matches += 1
        if valid and _position_decimal(algo.get("triggerPrice")) > 0:
            candidates += 1
    if bound_matches > 1:
        raise DemoSafetyError("DUPLICATE_BOUND_DEMO_STOP_ID")
    if amount == 0:
        status = ("FLAT_WITH_OPEN_STOP_CANDIDATE_MANUAL_REVIEW" if candidates
                  else "FLAT_BUT_PILOT_JOURNAL_LOCKED")
    elif ((amount > 0) != (row["side"] == "BUY")):
        status = "EXCHANGE_POSITION_DIRECTION_CONFLICT"
    elif abs(amount) != expected:
        status = "EXCHANGE_POSITION_SIZE_MISMATCH"
    elif bound_matches == 1 and candidates == 1:
        status = "POSITION_WITH_SIGNED_MATCHING_BOUND_STOP_NEEDS_FILL_AUDIT"
    elif bound_matches == 1 and candidates > 1:
        # Other conditional stops may trigger too; require manual review.
        status = "POSITION_WITH_EXTRA_OPEN_STOP_CANDIDATES_REVIEW"
    elif candidates == 1 and bound is None:
        status = "POSITION_WITH_POSSIBLE_STOP_OWNERSHIP_NOT_PROVEN"
    else:
        status = "POSITION_WITH_UNVERIFIED_PROTECTION"
    confirmed = bool(amount != 0 and bound is not None and bound_matches == 1
                     and status == "POSITION_WITH_SIGNED_MATCHING_BOUND_STOP_NEEDS_FILL_AUDIT")
    return {**result, "status": status, "authenticated_checks_performed": True,
            "signed_position_amount": str(amount),
            "possible_protective_stops": candidates,
            "bound_exchange_stop_identity_matched":bool(bound_matches==1),
            "stop_acknowledged":confirmed,
            "stop_belongs_to_pilot_unverified":not confirmed,
            "entry_exchange_fill_not_independently_reconciled":True,
            "no_orders_or_cancellations_submitted": True}
