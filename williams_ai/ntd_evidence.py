"""NTD five-dimensions evidence, strictly separate from TC2 Wise Men.

Signal patterns are candidate observations from *closed* indicator bars.
They are deliberately NOT exchange orders and never skip the initial
external confirmed fractal admission gate of the NTD profile.
Sources: NTD PDF 53-67 (AO), 72-80 (AC), 91-93 (zones).
"""
from __future__ import annotations
from typing import Any


def _finite(values: list[float | None], length: int) -> bool:
    if len(values) < length or any(v is None for v in values[-length:]):
        return False
    from math import isfinite
    return all(isfinite(float(v)) for v in values[-length:])


def ntd_closed_bar_evidence(
    ao: list[float | None], ac: list[float | None],
    *, initial_fractal_triggered: bool = False
) -> dict[str, Any]:
    """No current-price trigger, no actual trading authority, no look-ahead."""
    if not isinstance(ao, list) or not isinstance(ac, list) or len(ao) != len(ac):
        raise ValueError("AO and AC must be equally long arrays")
    signals: list[dict[str, Any]] = []
    def record(name: str, side: str, pages: str, note: str) -> None:
        signals.append({
            "profile": "NTD_FIVE_DIMENSIONS", "name": name, "side": side,
            "formation": "CLOSED_INDICATOR_CANDIDATE",
            "admission": (
                "INITIAL_FRACTAL_PROVEN_EXTERNALLY" if initial_fractal_triggered
                else "WAIT_FOR_FIRST_CONFIRMED_OUTSIDE_ALLIGATOR_FRACTAL_BREAKOUT"
            ),
            "trade_action": None,
            "source": "NTD PDF " + pages,
            "limitations": note,
        })
    if _finite(ao, 3):
        a, b, c = ao[-3:]
        # NTD 57: simple three-bar saucer subset, not multi-red generalization.
        if a > b and c > b and min(a, b, c) > 0:
            record("AO_SAUCER_THREE_BAR", "LONG", "56-58",
                   "Simplified form. Multiple intervening bars not analyzed.")
        if a < b and c < b and max(a, b, c) < 0:
            record("AO_SAUCER_THREE_BAR", "SHORT", "61-63",
                   "Simplified form. Multiple intervening bars not analyzed.")
    if _finite(ao, 2):
        before, now = ao[-2:]
        if before < 0 < now:
            record("AO_ZERO_CROSS", "LONG", "58-59",
                   "Price stop trigger and last valid signal must be checked separately.")
        if before > 0 > now:
            record("AO_ZERO_CROSS", "SHORT", "63-64",
                   "Price stop trigger and last valid signal must be checked separately.")
    # AO two peaks: a confirmed second valley on last-1, current AO
    # turns away, earlier extremum occurred at least two bars earlier.
    if _finite(ao, 6):
        now, second = ao[-1], ao[-2]
        for side in ("LONG", "SHORT"):
            if side == "LONG" and not (second < 0 and now > second):
                continue
            if side == "SHORT" and not (second > 0 and now < second):
                continue
            for p in range(len(ao)-5, max(0, len(ao)-62)-1, -1):
                older = ao[p]
                if older is None or p < 1 or ao[p-1] is None or ao[p+1] is None:
                    continue
                between = ao[p+1:-2]
                if any(v is None for v in between):
                    continue
                if side == "LONG":
                    older_turn = older < ao[p-1] and older < ao[p+1]
                    valid = older_turn and older < second and all(v < 0 for v in between)
                else:
                    older_turn = older > ao[p-1] and older > ao[p+1]
                    valid = older_turn and older > second and all(v > 0 for v in between)
                if valid:
                    record("AO_TWIN_PEAKS_POSSIBLE", side,
                           "60-61" if side == "LONG" else "65-67",
                           "First local extremum and second turn found; complete oscillator "
                           "and Alligator context still require graph review.")
                    break
    if _finite(ac, 4) and _finite(ao, 1):
        momentum = ao[-1]
        for side in ("LONG", "SHORT"):
            # NTD AC: 2 recent color bars if momentum aligns; 3 otherwise.
            needed = 2 if (momentum > 0 if side == "LONG" else momentum < 0) else 3
            sequence = ac[-(needed+1):]
            if all((sequence[j] > sequence[j-1] if side == "LONG"
                    else sequence[j] < sequence[j-1])
                   for j in range(1, len(sequence))):
                record("AC_ACCELERATION_CANDIDATE", side, "72-80",
                       "AO sign determines required bar count; AC zero crossing "
                       "itself is not a trading signal.")
    zone = "UNKNOWN"
    if _finite(ac,2) and _finite(ao,2):
        da, dc = ao[-1]-ao[-2], ac[-1]-ac[-2]
        zone = "GREEN" if da>0 and dc>0 else "RED" if da<0 and dc<0 else "GREY"
    return {
        "profile": "NTD_FIVE_DIMENSIONS", "zone": zone, "signals": signals,
        "entry_admitted": initial_fractal_triggered,
        "trade_action": None,
        "warning": "NTD candidates are not TC2 Wise Men or order instructions.",
    }
