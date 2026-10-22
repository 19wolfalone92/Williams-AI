"""Fast, source-bounded visual wave reference lookup (not trained neural weights).

Reference: user's 6-page chart memo after D. Vozny, SHA256 in JSON.
It is distinct from Bill Williams TC1/NTD/TC2/COURSE. No copied scans
or copyrighted pages are shipped. Every pattern is an *archetype* rather
than a validated label on actual OHLCV.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import json
import re
from typing import Any

_REFERENCE = Path(__file__).resolve().parents[1] / "knowledge/WAVE_REFERENCE_VOZNY_RU.json"
_ALIASES = {
    "волна":"волны", "волн":"волны", "elliott":"волны",
    "импульсный":"импульс", "зигзага":"зигзаг",
    "флэт":"плоскость", "wxy":"wxy", "triangle":"треугольник",
    "диагональ":"диагональ", "волновой":"волны",
    "triangle":"треугольник", "zigzag":"зигзаг",
}

@lru_cache(maxsize=1)
def registry() -> dict[str, Any]:
    data=json.loads(_REFERENCE.read_text(encoding="utf-8"))
    assert data["reference"]["total_pages"]==6
    assert len({item["id"] for item in data["cards"]})==len(data["cards"])
    for item in data["cards"]:
        assert 1<=item["page"]<=6 and item["execution_authority"] is False
    return data


def _terms(s: str) -> set[str]:
    terms=set(re.findall(r"[^\W_]+",s.casefold()))
    result=set()
    for term in terms:
        result.add(_ALIASES.get(term,term))
        if term.startswith(("волн","коррек","импульс","треуголь","диагонал")):
            result.add(term[:5])
    return result


def recall_wave(reference_query: str, *, limit: int=3) -> list[dict[str,Any]]:
    """A small deterministic sparse index: O(12 cards), cached JSON."""
    if not isinstance(reference_query,str) or len(reference_query)>1600:
        raise ValueError("Invalid wave question")
    if not isinstance(limit,int) or isinstance(limit,bool) or not 1<=limit<=6:
        raise ValueError("Invalid limit")
    terms=_terms(reference_query)
    if not terms:
        return []
    ranked=[]
    for card in registry()["cards"]:
        keywords=_terms(card["keywords"]+" "+card["title"]+" "+card["id"])
        overlap=terms & keywords
        if overlap:
            ranked.append((len(overlap),card))
    ranked.sort(key=lambda item:(-item[0],item[1]["page"],item[1]["id"]))
    return [{
        "source":f'VoznyWaveMemo: PDF {c["page"]} / {c["id"]}',
        "excerpt":c["summary"],
        "status":"WAVE_ARCHETYPE_UNVERIFIED_ON_MARKET",
        "checks":c["checks"],
        "authoritative_for_trade":False,
    } for _,c in ranked[:limit]]


@lru_cache(maxsize=1)
def _chart_cases()->dict[str,Any]:
    source=Path(__file__).resolve().parents[1]/"knowledge/USER_ANNOTATED_CHART_CASES.json"
    pack=json.loads(source.read_text(encoding="utf-8"))
    assert pack["schema"]=="WILLIAMS_USER_ANNOTATED_CHART_CASES_V1"
    return pack


def recall_annotated_charts(question:str, *,limit:int=1)->list[dict[str,Any]]:
    """Old hand-marked examples are hypotheses, never a chart ground truth."""
    if not isinstance(question,str) or len(question)>1600:
        raise ValueError("Invalid chart question")
    if not 1<=limit<=4:
        raise ValueError("limit must be 1..4")
    terms=_terms(question)
    ranks=[]
    for card in _chart_cases()["cases"]:
        haystack=_terms(card["title"]+" "+card["description"]+" "+card["id"])
        score=len(terms & haystack)
        if score:
            ranks.append((score,card))
    ranks.sort(key=lambda z:(-z[0],z[1]["id"]))
    return [{
        "source":"ANNOTATED_USER_CHART:"+v["id"],
        "excerpt":v["description"],
        "status":"VISUAL_ANNOTATION_UNVERIFIED_NUMERICALLY",
        "authoritative_for_trade":False,
    } for _,v in ranks[:limit]]


def structural_exam(claim: dict[str,Any]) -> dict[str,Any]:
    """Reject demonstrably invalid geometric claims, never confirm wave truth.

    A strict impulse cannot overlap W1/W4. This function cannot infer
    internal 5-3-5-3-5 wave degrees from six arbitrary extrema.
    """
    if not isinstance(claim,dict):
        raise ValueError("Claim must be object")
    pattern=claim.get("pattern")
    pivots=claim.get("pivots")
    if pattern not in {"IMPULSE","LEADING_DIAGONAL","ENDING_DIAGONAL"}:
        raise ValueError("Unsupported pattern")
    if not isinstance(pivots,list) or len(pivots)!=6:
        raise ValueError("Exactly six price turning points are needed")
    from math import isfinite
    if any(isinstance(x,bool) or not isinstance(x,(float,int)) or not isfinite(x) or x<=0 for x in pivots):
        raise ValueError("Prices must be finite positive")
    direction=1 if pivots[1]>pivots[0] else -1
    w1,w3,w5=(direction*(pivots[b]-pivots[a]) for a,b in ((0,1),(2,3),(4,5)))
    violations=[]
    if min(w1,w3,w5)<=0: violations.append("MOTIVE_LEG_DIRECTION")
    if direction*(pivots[2]-pivots[0])<=0: violations.append("W2_BREACHES_ORIGIN")
    if direction*(pivots[3]-pivots[1])<=0: violations.append("W3_DOES_NOT_EXTEND_W1")
    if w3<min(w1,w5): violations.append("W3_SHORTEST")
    overlap=direction*(pivots[4]-pivots[1])<=0
    if pattern=="IMPULSE" and overlap: violations.append("W4_OVERLAPS_W1_CLASSIC_IMPULSE")
    return {
        "pattern":pattern,"geometry_status":"REJECTED" if violations else "UNVERIFIED_CANDIDATE",
        "violations":violations,"w4_overlaps_w1":overlap,
        "internal_subwaves_verified":False,"wave_degree":"UNKNOWN",
        "reference":"VoznyWaveMemo pages 1-2 (distinct from Williams TC1/NTD/TC2)",
        "trade_action":None,"execution_allowed":False,
    }
