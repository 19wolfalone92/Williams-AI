"""Explicitly provisional roles for a few directly inspected Course pages.

No copyright images or long transcriptions are stored in this module.
These are *visual layout observations*, not authenticated author rules.
Unreviewed pages remain UNKNOWN; no inference may promote them to gold.
"""
from __future__ import annotations

VISIBLE_LAYOUT_DRAFTS={
    47:("WAVE_CHARACTERISTICS_DIAGRAM","An annotated first-wave/Point Zero example"),
    56:("NESTED_ELLiOTT_DEGREES_DIAGRAM","Several nested wave degrees 0..5 and A/B/C"),
    60:("CORRECTION_CHECKLIST_FLOWCHART","Sideways correction decision columns and arrows"),
    71:("FRACTAL_THUMB_PATTERN_DEFINITIONS","Two-bar reversal, fractal and thumb schematic"),
    90:("EMPTY_TRADING_WORKSHEET","Blank multi-market trading planning grid"),
    120:("EMPTY_REFLECTION_WORKSHEET","Blank private daily reflection/planning form"),
    121:("EMPTY_REFLECTION_WORKSHEET","Blank reflection, mistakes and goals form"),
}


def private_course_page_role(pdf_page:int)->dict:
    if type(pdf_page) is not int or not 1<=pdf_page<=122:
        raise ValueError("COURSE_PAGE_NUMBER_RANGE_REQUIRED")
    row=VISIBLE_LAYOUT_DRAFTS.get(pdf_page)
    if row is None:
        return {"role":"UNKNOWN_UNREVIEWED","description":None,
                "source_review_status":"NO_DIRECT_VISUAL_OBSERVATION"}
    role,description=row
    return {"role":role,"description":description,
            "source_review_status":"DIRECT_VISUAL_DRAFT_NOT_INDEPENDENTLY_VERIFIED",
            "automatic_market_gold_allowed":False,
            "blank_form_not_a_trading_rule":role.startswith("EMPTY_")}


def require_not_an_empty_rule_template(pdf_page:int):
    result=private_course_page_role(pdf_page)
    if result.get("blank_form_not_a_trading_rule"):
        raise ValueError("BLANK_WORKSHEET_IS_NOT_TRADING_SOURCE_GOLD")
    return result
