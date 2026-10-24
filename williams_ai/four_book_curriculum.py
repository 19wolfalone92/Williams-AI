"""Four BOOKS, FOUR distinct modes: a source-verified Williams learning passport.

This deliberately teaches the *boundaries* between the original works rather
than merging them into an imaginary one-size-fits-all bot strategy.

The complete PDFs and private images are NOT copied into the GitHub code, and
the page-reviewed cards are not a substitute for real image verification,
independent expert charts, trainable GPU weights, or broker fills.
"""
from __future__ import annotations

from typing import Any

from .book_rules import registry


# Evidence gates are source-specific. Any rule omitted from the reviewed
# registry blocks its entire teaching topic instead of silently substituting
# from another book. A named task describes what a student must demonstrate,
# not an assertion that the model can already do it.
_COURSE = {
    "TC1": (
        ("wave_hierarchy", ("TC1.W3","TC1.W4","TC1.W5","TC1.RHYTHM"),
         "Предлагай конкурирующие степени W1-W5 и ABC; AO у нуля — не доказательство W4."),
        ("five_magic_bullets", ("TC1.MAGIC.DIVERGENCE","TC1.MAGIC.TARGET",
         "TC1.MAGIC.FRACTAL","TC1.MAGIC.SQUAT","TC1.MAGIC.MOMENTUM"),
         "Пять пуль: реальная дивергенция, зона цели, фрактал, Squat с TICK volume, моментум 5/34/5; неполные признаки нельзя назвать полным набором."),
        ("fractal_and_psychology", ("TC1.FRACTAL","TC1.PSY.FEAR","TC1.PSY.GREED",
         "TC1.PSY.BELIEF","TC1.PSY.PLAN"),
         "Фрактал подтверждается правыми барами; страх/FOMO/месть не повод закрыть позицию или увеличить риск."),
    ),
    "NTD": (
        ("alligator_first_entry", ("NTD.ALLIGATOR","NTD.FRACTAL"),
         "Первый допуск профиля NTD — подходящий ВНЕ пасти фрактал с пробоем цены; видимые смещённые линии не читают будущие бары."),
        ("five_dimensions", ("NTD.AO","NTD.AC","NTD.ZONE",
         "NTD.ZONE.ADDED","NTD.BALANCE"),
         "Пять измерений раздельны: NTD AO Saucer/Twin Peaks/Zero Cross не равны TC2 WM2; AC над/под нулём, зона AO+AC, линия баланса."),
        ("edition_specific_exits", ("NTD.EXITS",),
         "Выходы NTD: Teeth close для тренда, зона 5 баров, Lips при быстром движении, действительный встречный сигнал; stop-on-close не равен биржевому внутрибaровому SL."),
    ),
    "TC2": (
        ("first_wise_man", ("TC2.WM1.BAR","TC2.WM1.ANG",
         "TC2.WM1.TRIGGER","TC2.WM1.STOP"),
         "WM1: разворотный бар плюс проверяемая ангуляция, ценовой пробой, первоначальный SL за другим концом сигнального бара."),
        ("other_wise_men", ("TC2.WM2","TC2.WM3"),
         "WM2: три изменения Super AO и отдельный ценовой пробой; WM3: подтверждённый фрактал t+2, Teeth в момент пробоя. Не путай наблюдение и реальный fill."),
        ("campaign_entry_add_exit", ("TC2.CAMPAIGN","TC2.REVERSE"),
         "Первым может быть WM1/WM2/WM3; обратная пирамида только после прибыли и лимитов; trailing не расширяет риск; противоположный мудрец без исполненного reverse НЕ закрывает позицию."),
    ),
    "COURSE": (
        ("wave_degree_and_zero", ("COURSE.AO5345","COURSE.W4.ZERO",
         "COURSE.RHYTHM","COURSE.VISUAL.NESTED_ABC"),
         "Волну исследуй на соразмерных ~100-140 закрытых барах; AO к нулю для вероятной W4 — контекст; вложенные ABC и 1-5 остаются гипотезами."),
        ("hump_and_thumb", ("COURSE.HUMP","COURSE.THUMB"),
         "Hump и Thumb — самостоятельные рисунки Course, проверяемые визуально, не фрактал WM3/NTD Balance Line по умолчанию."),
        ("trading_partner", ("COURSE.TRADING_PARTNER",),
         "Trading Partner оценивает старший ТФ, 5/34/5, Hump/Thumb и открытую кампанию, но не выдаёт биржевую команду."),
    ),
}


def four_book_teaching_passport(*, edition: str | None = None) -> dict[str, Any]:
    """Produce teachable reviewed-card objectives and honest coverage states.

    No private text, future bars, execution keys, or personal data is returned.
    A partial knowledge card set is explicitly represented, never filled in.
    """
    if edition is not None and edition not in _COURSE:
        raise ValueError("UNKNOWN_OR_MIXED_WILLIAMS_BOOK_EDITION")
    cards = registry()["rule_cards"]
    ids = {x["id"]:x for x in cards}
    selections = [edition] if edition else list(_COURSE)
    modules: dict[str, list[dict[str, Any]]] = {}
    for ed in selections:
        group=[]
        for stage, references, objective in _COURSE[ed]:
            checked=[]
            for rule_id in references:
                card=ids.get(rule_id)
                if (card is None or card["source"]!=ed
                        or card["status"]!="REVIEWED_PAGES"
                        or not card["pdf_pages"]):
                    raise ValueError("RULE_NOT_INDEPENDENTLY_PAGE_REVIEWED:"+rule_id)
                checked.append({"id":rule_id,"pdf_pages":list(card["pdf_pages"])})
            group.append({
                "stage":stage,"competency_objective":objective,
                "source_ed":ed,"reviewed_source_rule_references":checked,
                "text_reading_not_model_mastery":True,
                "expert_chart_acceptance_done":False,
                "weight_training_done_here":False,
            })
        modules[ed]=group
    return {
        "mode":"FOUR_WILLIAMS_BOOKS_SOURCE_BOUND_CURRICULUM",
        "selected_edition":edition or "ALL_DISTINCT_EDITIONS",
        "modules":modules,
        "comparison_policy":{
            "TC1":"Original Trading Chaos; waves, Magic Bullets and psychology.",
            "NTD":"New Trading Dimensions; first external fractal and five dimensions.",
            "TC2":"Trading Chaos 2; Three Wise Men, campaign and stop/reverse.",
            "COURSE":"Profitunity course; scaled waves, Hump/Thumb, Trading Partner.",
        },
        "cross_edition_execution_rules_must_not_merge":True,
        "no_trade_is_always_allowed":True,
        "trade_action":None,"execution_allowed":False,
        "book_mastery_percent":None,"GGUF_weights_trained_here":False,
    }


def single_book_model_lesson(edition: str) -> dict[str, Any]:
    """Compact bounded review objectives for a REAL GGUF prompt.

    Does not attach source text or blindly assert full book coverage.
    """
    passport=four_book_teaching_passport(edition=edition)
    return {
        "edition":edition,
        "correct_rule_family_must_be_used":True,
        "all_competing_wave_degrees_remain_unverified":True,
        "lessons":[{
            "stage":x["stage"],"objective":x["competency_objective"],
            "rule_ids":[y["id"] for y in x["reviewed_source_rule_references"]],
        } for x in passport["modules"][edition]],
        "cross_edition_rule_substitution_forbidden":True,
        "broker_execution_not_proved_by_chart":True,
        "trading_authority":False,
    }
