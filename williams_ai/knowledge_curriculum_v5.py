"""Williams AI 5.0 — reproducible source-grounded SFT data, NOT chart gold.

Only publicly reviewed *paraphrase cards* and pinned original PDF metadata
are used. Original copyrighted books, OCR, images and private client data
are NOT included. Exclude visually unverified diagrams, unknown labels,
future market information, synthetic PnL or invented expert approvals.

Split on the WHOLE (edition, rule_id), not on paraphrased row IDs.
Validation and test labels never enter training. Revisioned SQLite contains
independently queryable provenance and full rule-disjoint split receipts.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import argparse
import json
from pathlib import Path
import random
import sqlite3

from .book_rules import registry

EDITIONS=("TC1","NTD","TC2","COURSE")
SPLITS=("train","validation","test")
EXCLUDE={"COURSE.VISUAL.NESTED_ABC"}  # requires actual figure + independent expert visual review
SEED=20261111
SCHEMA="WILLIAMS_V5_SOURCE_CARDS_ONLY_RULE_DISJOINT_SFT"
_VARIANTS=12
SAFETY=("Ни одна книжная карточка сама по себе не подтверждает текущую рыночную "
        "сделку, цену исполнения, волну Эллиотта, биржевой SL или доходность.")


def canonical(obj)->bytes:
    return json.dumps(obj,sort_keys=True,ensure_ascii=False,
                      allow_nan=False,separators=(",",":")).encode("utf-8")


def digest(obj)->str:
    return sha256(canonical(obj)).hexdigest()


def _fold(card,position):
    if card["source"]=="TC2" and position==0:
        return "validation"
    if position == 0:
        return "validation"
    if position in (1,2) and card["source"]!="COURSE":
        return "test"
    if position == 1 and card["source"]=="COURSE":
        return "test"
    return "train"


def _target(card,style,correct_letter=None):
    if style=="MCQ":
        if correct_letter is None:
            raise ValueError("CORRECT_MCQ_LETTER_REQUIRED")
        return json.dumps({
            "choice":correct_letter,
            "reason":("В "+card["source"]+" это правило имеет ограничения; "
                      "см. PDF "+str(card["pdf_pages"][0])+"."),
        },ensure_ascii=False,separators=(",",":"))
    if style=="KNOWLEDGE":
        return json.dumps({
            "status":"SOURCE_SCOPED_RULE_NOT_MARKET_CONFIRMATION",
            "edition":card["source"],"rule_id":card["id"],
            "pdf_pages":card["pdf_pages"],
            "rule":card["summary"],"limits":SAFETY,
            "execution_allowed":False,
        },ensure_ascii=False,separators=(",",":"))
    if style=="BOUNDARY":
        return json.dumps({
            "status":"INSUFFICIENT_MARKET_CONFIRMATION",
            "edition":card["source"],"rule_id":card["id"],
            "evidence":card["summary"],
            "missing":"Закрытые согласованные H1/H4/D1, подтверждения структуры и биржевые статусы.",
            "research_only":True,"execution_allowed":False,
        },ensure_ascii=False,separators=(",",":"))
    raise ValueError("UNRECOGNIZED_TRAINING_TASK")


def _sample(card,n):
    ed,rule,title=card["source"],card["id"],card["title"]
    p=card["pdf_pages"][0]
    rng=random.Random(int(digest({"rule":rule,"n":n,"seed":SEED})[:16],16))
    if n%3==0:
        style="MCQ"
        correct=card["summary"]
        # Two intentionally false claims about automatic execution and
        # edition conflation; never invent a second plausible expert rule.
        wrongs=[
            ("Достаточно заметить термин «"+title+
             "»: он уже доказывает открытие позиции и фактический fill."),
            ("Любой сигнал из "+ed+
             " без дополнительной проверки всегда совпадает с правилами других книг."),
        ]
        choices=[correct]+wrongs
        rng.shuffle(choices)
        stems=[
            f"Для книги {ed}: какое утверждение о «{title}» соответствует карточке?",
            f"Выбери строго книжное определение «{title}» в издании {ed}.",
            f"Какое из трёх утверждений о «{title}» в {ed} не обещает рыночный fill?",
            f"Отдели правило «{title}» ({ed}) от неподтверждённых торговых выводов."
        ]
        prompt={"edition":ed,"question":stems[n//3],
                "choices":dict(zip("ABC",choices))}
        answer=_target(card,style,"ABC"[choices.index(correct)])
    elif n%3==1:
        style="KNOWLEDGE"
        stems=[
            f"Объясни «{title}» ({ed}) и укажи страницы источника.",
            f"Назови ограничения правила «{title}» из книги {ed}.",
            f"Отдели проверяемые источники от гипотез для «{title}» ({ed}).",
            f"Что известно про «{title}» в {ed}, а что остаётся неизвестным?"
        ]
        prompt={"edition":ed,"question":stems[n//3]}
        answer=_target(card,style)
    else:
        style="BOUNDARY"
        stems=[
            f"Есть только карточка «{title}» ({ed}). Допустим ли торговый вход?",
            f"Нет закрытых H1/H4/D1: как исследовать «{title}» ({ed}) без догадок?",
            f"Книжный термин «{title}» виден, но статус ордера отсутствует. Что известно?",
            f"Применение «{title}» из {ed} без рыночных подтверждений: вердикт?"
        ]
        prompt={"edition":ed,"question":stems[n//3],
                "reviewed_source_context":{
                    "rule_id":rule,"pages":card["pdf_pages"],
                    "paraphrased_rule":card["summary"],
                    "not_expert_market_gold":True}}
        answer=_target(card,style)
    return {"schema":SCHEMA,
        "case_id":"V5-"+digest({"r":rule,"v":n,"seed":SEED})[:16],
        "rule_id":rule,"edition":ed,"pdf_pages":list(card["pdf_pages"]),
        "source_sha256":card["source_sha256"],
        "review_status":"PAGE_CHECKED_RULE_PARAPHRASE_NOT_INDEPENDENT_MARKET_GOLD",
        "split":card["split"],"variant":n,"task_type":style,
        "prompt":prompt,"answer":answer,
        "source_excerpt_in_training":False,
        "visual_semantics_independently_verified":False,
        "chart_expert_gold":False,
        "eligible_for_sft":card["split"]=="train",
        "execution_allowed":False}


def build():
    source=registry()
    books={b["id"]:b for b in source["books"]}
    rules=[r for r in source["rule_cards"]
           if r["status"]=="REVIEWED_PAGES" and r["id"] not in EXCLUDE]
    omitted=[r["id"] for r in source["rule_cards"] if r not in rules]
    mapped=[]
    for ed in EDITIONS:
        cards=sorted((c for c in rules if c["source"]==ed),key=lambda c:c["id"])
        if len(cards)<4:
            raise ValueError("INSUFFICIENT_PER_BOOK_CHECKED_RULE_CARDS")
        # Seeded selection BEFORE generating a single paraphrased question.
        order=cards[:]
        random.Random(SEED+EDITIONS.index(ed)*1009).shuffle(order)
        for i,c in enumerate(order):
            if any(not isinstance(p,int) or not 1<=p<=books[ed]["pages"]
                   for p in c["pdf_pages"]):
                raise ValueError("UNPINNED_RULE_CARD_PAGE")
            mapped.append({**c,"source_sha256":books[ed]["sha256"],
                           "split":_fold(c,i)})
    counts=Counter((c["source"],c["split"]) for c in mapped)
    if any(counts[(ed,split)]<=0 for ed in EDITIONS for split in SPLITS):
        raise ValueError("ALL_FOUR_BOOKS_IN_EACH_FROZEN_SPLIT_REQUIRED")
    out=[_sample(card,v) for card in mapped for v in range(_VARIANTS)]
    checks={part:[x for x in out if x["split"]==part] for part in SPLITS}
    ids=[x["case_id"] for x in out]
    prompts=[digest({"prompt":x["prompt"],"edition":x["edition"]}) for x in out]
    owners={}
    for x in out:
        old=owners.setdefault((x["edition"],x["rule_id"]),x["split"])
        if old!=x["split"]:
            raise ValueError("TRAIN_VALIDATION_TEST_RULE_LEAKAGE")
    if len(ids)!=len(set(ids)) or len(prompts)!=len(set(prompts)):
        raise ValueError("DUPLICATED_QUESTION_OR_CASE_ACROSS_FROZEN_SPLITS")
    if min(len(checks[x]) for x in SPLITS)<10:
        raise ValueError("INSUFFICIENT_DEV_OR_HOLDOUT")
    manifest={
        "schema":SCHEMA,"seed":SEED,"variants_per_rule":_VARIANTS,
        "source_registry_sha256":digest(source),
        "sources":[{"book":b["id"],"pdf_sha256":b["sha256"],
                    "pages":b["pages"]} for b in source["books"]],
        "excluded_rules":omitted,
        "approved_market_labels":0,
        "independently_reviewed_diagrams":0,
        "split_rule_ids":{part:sorted(x["id"] for x in mapped
                    if x["split"]==part) for part in SPLITS},
        "split_rows":{part:len(checks[part]) for part in SPLITS},
        "split_prompt_sha256":{part:digest([x["prompt"] for x in checks[part]])
                               for part in SPLITS},
        "split_gold_sha256":{part:digest([x["answer"] for x in checks[part]])
                             for part in SPLITS},
        "data_quality_gate":"RULE_CARDS_ONLY_CONDITIONAL_NOT_EXPERT_CHART_GOLD",
        "future_data_used":False,"test_answers_allowed_into_gpu":False,
        "source_pdf_text_or_images_in_dataset":False,
        "test_locked_before_training":True,
        "live_trading_allowed":False}
    return out,manifest




# V6 is a source-only extension of the existing curriculum architecture.
# It does not import chart labels, expert claims, or the old exposed exam cohort.
V6_SCHEMA="WILLIAMS_V6_SOURCE_ONLY_CURRICULUM"
V6_MODULES=(
    "market_structure","alligator","angulation","ao","ac","fractals","waves",
    "wm1","wm2","wm3","wise_men_interaction","multitimeframe",
    "position_management","risk_management","uncertainty",
)
V6_EXCLUDED_VISUAL={"V7.COURSE.WAVE_VISUAL.NESTED_ABC_DEGREES"}
V6_LESSON_MODULES={
 "V7.TC1.FRACTAL.START_SIGNAL_CANCEL":["fractals","wm3","position_management"],
 "V7.TC1.WAVE.CAMPAIGN":["market_structure","waves","position_management"],
 "V7.NTD.BALANCE.BASE.COUNT":["market_structure","position_management"],
 "V7.NTD.BALANCE.PENDING_STOP":["market_structure","position_management"],
 "V7.TC2.WM1.ANGULATION":["wm1","angulation"],
 "V7.TC2.WM2.AO_THREE_BARS":["wm2","ao"],
 "V7.TC2.WM3.TEETH_ACTIVATION":["wm3","fractals","alligator"],
 "V7.TC2.OPPOSITE.SIGNAL_POSITION":["wise_men_interaction","position_management"],
 "V7.COURSE.W4.ZERO_CROSS":["waves","ao","uncertainty"],
 "V7.COURSE.WAVE_DEGREE.W5_TARGET":["waves","uncertainty"],
 "V7.NTD.ALLIGATOR.THREE_BALANCE_LINES":["alligator","market_structure"],
 "V7.NTD.AO.SAUCER_CROSS_TWIN_PEAKS":["ao","market_structure"],
 "V7.NTD.AC.ACCELERATION_SIGNAL_COUNTS":["ac","ao"],
 "V7.NTD.ZONES.AO_AC_PRICE_ADDITION":["ao","ac","market_structure"],
 "V7.NTD.BALANCE.ZONE.ALLIGATOR_FILTERS":["market_structure","alligator","position_management"],
 "V7.NTD.EXITS.FOUR_MARKET_DECISIONS":["position_management","alligator","ao","ac"],
 "V7.TC1.MFI.PROFITUNITY_WINDOWS.AIRBAG":["market_structure","risk_management"],
 "V7.TC1.WAVE.MFI.MACD.W4.W5":["waves","uncertainty","ao"],
 "V7.TC1.FIVE_MAGIC_BULLETS.ZERO_POINT":["wise_men_interaction","waves","uncertainty"],
 "V7.TC2.WISE_MEN.FIRST_SIGNAL_ALTERNATIVES":["wise_men_interaction","wm1","wm2","wm3"],
 "V7.TC2.CAMPAIGN.REVERSE_PYRAMID_AND_TRAILING":["position_management"],
 "V7.TC2.EXITS.TRAILING_VS_REVERSE_AND_PARTIAL_CLOSE":["position_management","wise_men_interaction"],
 "V7.COURSE.MFI.PROFITUNITY_WINDOWS":["market_structure","risk_management"],
 "V7.COURSE.THUMB.EARLY_FRACTAL_ENTRY":["fractals","wm1","wm3"],
 "V7.COURSE.HUMP.OPPOSITE_NONOVERLAPPING_FRACTALS":["fractals","wm3"],
 "V7.COURSE.TRADING_PARTNER.MULTITIMEFRAME_TABLE":["multitimeframe"],
 "V7.TC1.PSYCHOLOGY.FOUR_TRADING_PROGRAMS":["risk_management","uncertainty"],
 "V7.TC1.PSYCHOLOGY.INSIDE_RISK_STRUCTURE_I_CAN":["risk_management","uncertainty"],
}

V6_PRACTICAL_TASKS=[
 {"module":"market_structure","source_refs":[{"edition":"NTD","pages":[99,100,101,102,103,104,105]}],"question":"На закрытых H1 swing-high равны 100, 104, 109, а swing-low — 92, 95, 99. Какова минимально подтверждаемая структура?","answer":{"decision":"HIGHER_HIGHS_AND_HIGHER_LOWS","interpretation":"Восходящая структура HH/HL; сама по себе не является сигналом входа.","limits":"Нужны отдельные условия Мудрецов и контекст старших таймфреймов."},"evidence_class":"MATHEMATICALLY_CHECKABLE"},
 {"module":"alligator","source_refs":[{"edition":"NTD","pages":[34,35,36,37,38,39,40]}],"question":"Линия Jaw отображается на баре 230 со сдвигом на 8 баров вперёд, а Teeth — со сдвигом на 5. Какие исходные индексы расчёта соответствуют отображаемому бару 230?","answer":{"jaw_source_index":222,"teeth_source_index":225,"limits":"Сдвиг визуализации не разрешает использовать будущие значения. Спорный сдвиг Lips здесь не используется."},"evidence_class":"MATHEMATICALLY_CHECKABLE"},
 {"module":"angulation","source_refs":[{"edition":"TC2","pages":[121,122,123,124]}],"question":"На закрытом участке наклон ценовой траектории +3.6, наклон пасти Alligator +1.1; расхождение углов увеличивается. Что можно заключить без сигнала входа?","answer":{"decision":"ANGULATION_HYPOTHESIS_SUPPORTED","reason":"Геометрия указывает на увеличение расхождения цены и пасти; это подтверждает гипотезу ангуляции, но не создаёт самостоятельный вход.","limits":"Нужны корректные закрытые бары, структура и отдельный допустимый сигнал."},"evidence_class":"BOOK_RULE_APPLICATION"},
 {"module":"ao","source_refs":[{"edition":"TC2","pages":[135,137,138,139]}],"question":"Для уже рассчитанной медианной цены SMA5=127.4, SMA34=126.8. Чему равен AO в этой точке по принятой в проекте реализации?","answer":{"ao":0.6,"calculation":"127.4 - 126.8","limits":"Формула рисунка TC2 имеет зафиксированное расхождение с кодом приложения; этот пример проверяет арифметику готовых SMA, а не разрешает конфликт первоисточника."},"evidence_class":"MATHEMATICALLY_CHECKABLE"},
 {"module":"ac","source_refs":[{"edition":"NTD","pages":[72,73,74,75,76,77,78,79,80]}],"question":"AO равен 1.2, а SMA5 от AO равна 0.7. Рассчитай AC.","answer":{"ac":0.5,"calculation":"1.2 - 0.7"},"evidence_class":"MATHEMATICALLY_CHECKABLE"},
 {"module":"fractals","source_refs":[{"edition":"TC1","pages":[72,73,74]},{"edition":"COURSE","pages":[69,70,71]}],"question":"Пять последовательных закрытых баров имеют high: 11, 13, 19, 16, 14. Подтверждён ли верхний фрактал на третьем баре?","answer":{"confirmed":True,"reason":"Экстремум центрального бара выше двух баров слева и двух закрытых баров справа.","limits":"Если хотя бы один из двух правых баров ещё открыт, фрактал не подтверждён."},"evidence_class":"MATHEMATICALLY_CHECKABLE"},
 {"module":"waves","source_refs":[{"edition":"COURSE","pages":[43,44,45]},{"edition":"TC1","pages":[49,50,51,52,53,54,55]}],"question":"AO на последовательных закрытых барах: +2.0, 0.0, +0.4. Пересекла ли кривая AO нулевую линию или только коснулась её?","answer":{"decision":"TOUCH_NOT_CROSS","reason":"Значение достигло нуля, но не перешло в отрицательную область.","limits":"Одного касания недостаточно, чтобы объявить завершение волны или выбрать единственный волновой сценарий."},"evidence_class":"MATHEMATICALLY_CHECKABLE"},
 {"module":"wm1","source_refs":[{"edition":"TC2","pages":[121,122,123,124]}],"question":"Предыдущий минимум равен 112. Текущий закрытый бар имеет low=108, high=124, close=120. Выполнено ли условие бычьего разворотного бара относительно midpoint?","answer":{"midpoint":116,"close_above_midpoint":True,"decision":"BULLISH_REVERSAL_BAR_CONDITION_MET","limits":"Это проверка бара, а не подтверждение исполнения STOP-входа."},"evidence_class":"BOOK_RULE_APPLICATION"},
 {"module":"wm2","source_refs":[{"edition":"TC2","pages":[135,137,138,139]}],"question":"Четыре последовательных закрытых значения AO равны -9, -7, -5, -2. Есть ли три последовательных повышения AO, заканчивающихся на последнем баре?","answer":{"consecutive_rises":3,"decision":"THREE_BAR_AO_MOMENTUM_CONDITION_MET","limits":"Это условие импульса WM2; отдельно проверяются предусмотренный ценовой STOP-триггер и контекст."},"evidence_class":"BOOK_RULE_APPLICATION"},
 {"module":"wm3","source_refs":[{"edition":"TC2","pages":[147,148,149]}],"question":"Верхний фрактал равен 250, buy STOP установлен на 250.5, а Teeth в момент предполагаемого срабатывания равен 252. Выполнено ли условие LONG WM3 относительно Teeth?","answer":{"decision":"NO_VALID_WM3_LONG_ACTIVATION","reason":"Уровень активации 250.5 не находится выше Teeth 252 в момент срабатывания."},"evidence_class":"BOOK_RULE_APPLICATION"},
 {"module":"wise_men_interaction","source_refs":[{"edition":"TC2","pages":[150,151,152,153,154,155,156,157,158,159,160,161,162]}],"question":"Первым допустимым сигналом в новой кампании появился корректный WM2, тогда как WM1 и WM3 ещё не возникали. Нужно ли ждать WM1?","answer":{"decision":"WM2_MAY_BE_FIRST_VALID_SIGNAL","reason":"Порядок трёх Мудрецов не является обязательной очередью; любой допустимый Мудрец может оказаться первым."},"evidence_class":"BOOK_RULE_APPLICATION"},
 {"module":"multitimeframe","source_refs":[{"edition":"COURSE","pages":[88,89]}],"question":"По пользовательской политике D1/H4 задают глобальный контекст, H1 — принятие решений, M15/M5 — диагностику. Обязан ли сигнал H1 ждать подтверждения M15?","answer":{"decision":"NO_M15_MANDATORY_CONFIRMATION","reason":"M15/M5 используются для наблюдения и диагностики, а не как обязательный фильтр входа."},"evidence_class":"USER_APPROVED_POLICY"},
 {"module":"position_management","source_refs":[{"edition":"TC2","pages":[154,155,156,157,158,159,160,161,162,163,164,165]}],"question":"Для LONG выбран TC2 trailing по трём завершённым барам. Их low: 104, 102, 106; минимальный тик 0.5. Какой уровень STOP получается при размещении на тик ниже минимума?","answer":{"lowest_low":102,"stop":101.5,"exit_reason":"THREE_BAR_TRAILING_STOP","limits":"Не заменять книжное сопровождение ATR и не переносить STOP только из-за текущей прибыли."},"evidence_class":"BOOK_RULE_APPLICATION"},
 {"module":"risk_management","source_refs":["USER_POLICY"],"question":"При базе 5000 USDT рассчитай дневной лимит 3%, аварийный breaker 5% и максимальную маржу 50%. Также укажи блокировку после трёх подряд убыточных сделок.","answer":{"daily_loss_limit_usdt":150,"circuit_breaker_usdt":250,"maximum_margin_usdt":2500,"consecutive_losses_lockout":3},"evidence_class":"USER_APPROVED_POLICY"},
 {"module":"uncertainty","source_refs":["USER_POLICY","SOURCE_CONFLICT_REGISTER"],"question":"Два волновых сценария одинаково совместимы с доступными закрытыми H1 барами. Что должен выдать аналитик вместо принудительного единственного счёта?","answer":{"decision":"PRESERVE_ALTERNATIVE_HYPOTHESES","required":["описать оба сценария","указать подтверждающие и противоречащие факты","дать уровни/условия инвалидации каждого сценария","обозначить неопределённость"],"forbidden":"Выдавать один сценарий за доказанный без достаточных данных."},"evidence_class":"UNCERTAINTY_POLICY"},
]


def _v6_row(case_id, modules, task_type, question, answer, *, edition,
            pages, source_sha256, evidence_class, source_refs):
    if not modules or any(x not in V6_MODULES for x in modules):
        raise ValueError("V6_UNKNOWN_OR_EMPTY_COMPETENCY_MODULE")
    prompt={"module_ids":list(modules),"question":question,
            "source_refs":source_refs,"evidence_class":evidence_class}
    return {
        "schema":V6_SCHEMA,"case_id":case_id,"module_ids":list(modules),
        "edition":edition,"pdf_pages":list(pages),"source_sha256":source_sha256,
        "split":"train","task_type":task_type,"prompt":prompt,
        "answer":json.dumps(answer,ensure_ascii=False,sort_keys=True,separators=(",",":")),
        "evidence_class":evidence_class,
        "review_status":"SOURCE_SPOTCHECKED_PARAPHRASE_NOT_EXPERT_MARKET_GOLD",
        "source_excerpt_in_training":False,"chart_expert_gold":False,
        "expert_market_gold":False,"visual_semantics_independently_verified":False,
        "eligible_for_sft":True,"execution_allowed":False,
    }


def source_only_admission():
    """Validate source provenance for book-only SFT; never authorizes market gold."""
    root=Path(__file__).resolve().parent.parent
    candidate=json.loads((root/"knowledge"/"WILLIAMS_V7_SOURCE_CURRICULUM_CANDIDATES_20261115.json").read_text(encoding="utf-8"))
    conflicts=json.loads((root/"research"/"WILLIAMS_V7_SOURCE_CONFLICTS_20261117.json").read_text(encoding="utf-8"))
    crosswalk=json.loads((root/"knowledge"/"TC2_EN_CROSS_REFERENCE_20261124.json").read_text(encoding="utf-8"))
    course_audit=json.loads((root/"research"/"WILLIAMS_AI_COURSE_OCR_VISUAL_MATCH_20261124.json").read_text(encoding="utf-8"))
    pack=registry()
    books={b["id"]:b for b in pack["books"]}
    lessons=candidate.get("lesson_cards",[])
    if len(lessons)!=29 or candidate.get("status",{}).get("training_permitted") is not False:
        raise ValueError("EXPECTED_29_SOURCE_CANDIDATES_WITH_EXPERT_GATE_STILL_CLOSED")
    if conflicts.get("expert_approved") is not False or conflicts.get("sft_ready") is not False:
        raise ValueError("SOURCE_CONFLICTS_MUST_REMAIN_UNRESOLVED")
    if len(conflicts.get("issues",[]))!=4 or conflicts.get("labels_generated")!=0:
        raise ValueError("EXPECTED_FOUR_EXPLICIT_SOURCE_CONFLICTS_WITHOUT_LABELS")
    tc2_ru=crosswalk.get("canonical_edition",{})
    tc2_en=crosswalk.get("cross_reference_edition",{})
    if (tc2_ru.get("sha256")!=books["TC2"]["sha256"] or tc2_ru.get("pages")!=books["TC2"]["pages"]):
        raise ValueError("TC2_RU_CANONICAL_HASH_OR_PAGE_COUNT_MISMATCH")
    variants=books["TC2"].get("cross_reference_editions",[])
    if not any(v.get("id")=="TC2_EN" and v.get("sha256")==tc2_en.get("sha256") and v.get("pages")==tc2_en.get("pages") for v in variants):
        raise ValueError("TC2_EN_VARIANT_NOT_PINNED_IN_SOURCE_REGISTRY")
    if tc2_en.get("sha256")!="eb734089bb19e72a92e023602bf558e43b7cb20aecf77469fdfe28f8e6cd3523" or tc2_en.get("pages")!=257:
        raise ValueError("TC2_EN_CROSS_REFERENCE_HASH_OR_PAGE_COUNT_MISMATCH")
    if (course_audit.get("original",{}).get("sha256")!=books["COURSE"]["sha256"] or
        course_audit.get("original",{}).get("pages")!=books["COURSE"]["pages"]):
        raise ValueError("COURSE_ORIGINAL_HASH_OR_PAGE_COUNT_MISMATCH")
    for issue in conflicts["issues"]:
        ed=issue["source_edition"]
        pages=issue["source_pages"]
        if ed=="TC2_EN":
            expected=tc2_en["sha256"]; max_page=tc2_en["pages"]
        else:
            if ed not in books:
                raise ValueError("UNKNOWN_SOURCE_CONFLICT_EDITION")
            expected=books[ed]["sha256"]; max_page=books[ed]["pages"]
        if issue["source_pdf_sha256"]!=expected or any(type(p) is not int or not 1<=p<=max_page for p in pages):
            raise ValueError("SOURCE_CONFLICT_PROVENANCE_MISMATCH")
        if issue.get("training_approved") is not False:
            raise ValueError("SOURCE_CONFLICT_MUST_NOT_BE_TRAINED_AS_GOLD")
    if (course_audit.get("comparison",{}).get("pages_compared")!=122 or
        course_audit.get("comparison",{}).get("pixel_identical_pages")!=122 or
        course_audit.get("comparison",{}).get("result")!="PASS_IMAGE_LAYER_IDENTICAL"):
        raise ValueError("COURSE_OCR_IMAGE_LAYER_NOT_VERIFIED")
    for card in lessons:
        ed=card.get("edition")
        if ed not in books or card.get("source_pdf_sha256")!=books[ed]["sha256"]:
            raise ValueError("V6_SOURCE_FILE_SHA256_MISMATCH")
        pages=card.get("pages")
        if (not isinstance(pages,list) or not pages or
            any(type(p) is not int or not 1<=p<=books[ed]["pages"] for p in pages)):
            raise ValueError("V6_SOURCE_PAGE_OUT_OF_RANGE")
        if card.get("id") not in V6_LESSON_MODULES and card.get("id") not in V6_EXCLUDED_VISUAL:
            raise ValueError("V6_LESSON_MODULE_MAPPING_MISSING")
        if not card.get("principles") or not card.get("required_sequence") or not card.get("invalid_inferences"):
            raise ValueError("V6_LESSON_MISSING_DISTINCT_LEARNING_DIMENSION")
    if not V6_EXCLUDED_VISUAL.issubset({x["id"] for x in lessons}):
        raise ValueError("EXPECTED_VISUAL_ONLY_LESSON_TO_REMAIN_EXCLUDED")
    return {
        "schema":"WILLIAMS_V6_SOURCE_ONLY_ADMISSION_V1",
        "source_lessons_total":len(lessons),
        "source_lessons_eligible":len(lessons)-len(V6_EXCLUDED_VISUAL),
        "excluded_visual_lessons":sorted(V6_EXCLUDED_VISUAL),
        "canonical_books":sorted(books),
        "tc2_english_cross_reference_only":True,
        "tc2_source_conflicts_unresolved":len(conflicts["issues"]),
        "course_ocr_images_identical":True,
        "source_pages_spotchecked":True,
        "full_899_page_semantic_review_complete":False,
        "source_only_sft_permitted":True,
        "expert_market_gold_training_permitted":False,
        "independent_expert_reviews":0,
        "chart_market_gold_rows":0,
        "execution_allowed":False,
        "trading_qualification":False,
    }


def build_v6_source():
    """Build a non-duplicated source-only curriculum; no chart gold or exposed exam data."""
    admission=source_only_admission()
    root=Path(__file__).resolve().parent.parent
    candidate=json.loads((root/"knowledge"/"WILLIAMS_V7_SOURCE_CURRICULUM_CANDIDATES_20261115.json").read_text(encoding="utf-8"))
    conflicts=json.loads((root/"research"/"WILLIAMS_V7_SOURCE_CONFLICTS_20261117.json").read_text(encoding="utf-8"))
    pack=registry()
    books={b["id"]:b for b in pack["books"]}
    rows=[]
    for card in candidate["lesson_cards"]:
        if card["id"] in V6_EXCLUDED_VISUAL:
            continue
        modules=V6_LESSON_MODULES[card["id"]]
        ed=card["edition"]; pages=card["pages"]; source_sha=card["source_pdf_sha256"]
        refs=[{"edition":ed,"pages":pages,"sha256":source_sha,"rule_ids":card["rule_ids"]}]
        tasks=[
            ("THEORY",f"Объясни книжную концепцию «{card['concept']}» по указанным страницам {ed}, не превращая её в экспертную разметку графика.",
             {"status":"BOOK_SOURCE_PRINCIPLES","edition":ed,"pages":pages,"principles":card["principles"]}),
            ("CAUSAL_SEQUENCE",f"Восстанови причинно-следовательную последовательность для «{card['concept']}» ({ed}).",
             {"status":"BOOK_SOURCE_SEQUENCE","edition":ed,"pages":pages,"sequence":card["required_sequence"]}),
            ("COUNTEREXAMPLE",f"Назови недопустимые выводы при применении «{card['concept']}» ({ed}) и объясни, почему они не следуют из источника.",
             {"status":"BOOK_SOURCE_INVALID_INFERENCES","edition":ed,"pages":pages,"invalid_inferences":card["invalid_inferences"]}),
            ("BOUNDARY",f"Какие ограничения и неразрешённые вопросы остаются у концепции «{card['concept']}» ({ed})?",
             {"status":"SOURCE_BOUNDARY","edition":ed,"pages":pages,"unresolved":card["unresolved"],"do_not_force_conclusion":True}),
        ]
        if not card["unresolved"]:
            tasks=tasks[:3]
        for task_type,question,answer in tasks:
            cid="V6-"+digest({"lesson":card["id"],"task":task_type})[:18]
            rows.append(_v6_row(cid,modules,task_type,question,answer,
                edition=ed,pages=pages,source_sha256=source_sha,
                evidence_class="BOOK_SOURCE_PARAPHRASE",source_refs=refs))
    # Conflicts are taught only as unresolved source discrepancies, never as a
    # deterministic replacement formula or independent expert conclusion.
    for issue in conflicts["issues"]:
        ed=issue["source_edition"]; pages=issue["source_pages"]
        source_sha=issue["source_pdf_sha256"]
        question=f"В источнике обнаружено расхождение по теме «{issue['id']}». Как следует отразить его в аналитике без самостоятельного разрешения?"
        answer={"status":"UNRESOLVED_SOURCE_CONFLICT","issue_id":issue["id"],
                "topic":issue.get("unresolved_question"),
                "conflicting_evidence":issue["conflicting_source_evidence"],
                "resolution_status":issue.get("suggested_resolution_status","UNRESOLVED"),
                "do_not_choose_one_version":True,"independent_expert_approval":False}
        cid="V6-CONFLICT-"+digest({"issue":issue["id"]})[:18]
        rows.append(_v6_row(cid,["uncertainty"],"SOURCE_CONFLICT",question,answer,
            edition=ed,pages=pages,source_sha256=source_sha,
            evidence_class="SOURCE_CONFLICT",source_refs=[
                {"edition":ed,"pages":pages,"sha256":source_sha}]))
    for item in V6_PRACTICAL_TASKS:
        refs=item.get("source_refs",[])
        for ref in refs:
            if isinstance(ref,dict):
                ed=ref.get("edition")
                if ed not in books or ref.get("sha256",books[ed]["sha256"])!=books[ed]["sha256"]:
                    raise ValueError("V6_PRACTICAL_SOURCE_HASH_MISMATCH")
                pages=ref.get("pages",[])
                if not pages or any(type(p) is not int or not 1<=p<=books[ed]["pages"] for p in pages):
                    raise ValueError("V6_PRACTICAL_SOURCE_PAGE_OUT_OF_RANGE")
            elif ref not in {"USER_POLICY","SOURCE_CONFLICT_REGISTER"}:
                raise ValueError("V6_UNKNOWN_NON_BOOK_PROVENANCE")
        source_ed=refs[0]["edition"] if refs and isinstance(refs[0],dict) else "USER_POLICY"
        pages=refs[0]["pages"] if refs and isinstance(refs[0],dict) else []
        source_sha=books[source_ed]["sha256"] if source_ed in books else "USER_POLICY"
        cid="V6-PRACTICAL-"+item["module"]+"-"+digest(item)[:12]
        rows.append(_v6_row(cid,[item["module"]],"PRACTICAL",item["question"],item["answer"],
            edition=source_ed,pages=pages,source_sha256=source_sha,
            evidence_class=item["evidence_class"],source_refs=refs or ["USER_POLICY"]))
    ids=[x["case_id"] for x in rows]
    prompts=[digest({"prompt":x["prompt"],"edition":x["edition"]}) for x in rows]
    if len(ids)!=len(set(ids)) or len(prompts)!=len(set(prompts)):
        raise ValueError("V6_DUPLICATED_CASE_OR_PROMPT")
    coverage={module:sum(module in x["module_ids"] for x in rows) for module in V6_MODULES}
    if any(coverage[m]<=0 for m in V6_MODULES):
        raise ValueError("V6_ALL_FIFTEEN_COMPETENCIES_REQUIRED")
    if any(x["split"]!="train" or not x["eligible_for_sft"] or
           x["expert_market_gold"] or x["chart_expert_gold"] or x["execution_allowed"]
           for x in rows):
        raise ValueError("V6_SOURCE_ONLY_TRAINING_SCOPE_VIOLATION")
    manifest={
        **admission,
        "dataset_schema":V6_SCHEMA,
        "source_registry_sha256":digest(pack),
        "source_candidate_sha256":digest(candidate),
        "source_conflicts_sha256":digest(conflicts),
        "row_count":len(rows),
        "split_rows":{"train":len(rows),"validation":0,"test":0},
        "module_coverage":coverage,
        "task_type_counts":dict(Counter(x["task_type"] for x in rows)),
        "prompt_sha256":digest([x["prompt"] for x in rows]),
        "gold_sha256":digest([x["answer"] for x in rows]),
        "all_rows_train":True,
        "old_v5_432_case_database_preserved":True,
        "old_revealed_exams_for_regression_only":True,
        "future_data_used":False,
        "source_pdf_text_or_images_in_dataset":False,
    }
    return rows,manifest


def export_v6_source_database(path:Path):
    rows,manifest=build_v6_source()
    if path.exists() or path.is_symlink():
        raise FileExistsError("NEW_VERSIONED_V6_DATABASE_REQUIRED")
    path.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_RDWR,0o600)
    os.close(fd)
    try:
        with sqlite3.connect(path) as db:
            db.execute("""CREATE TABLE curriculum(
                case_id TEXT PRIMARY KEY,module_ids_json TEXT NOT NULL,
                edition TEXT NOT NULL,pdf_pages_json TEXT NOT NULL,
                source_sha256 TEXT NOT NULL,task_type TEXT NOT NULL,
                evidence_class TEXT NOT NULL,prompt_json TEXT NOT NULL,
                gold_answer TEXT NOT NULL,content_sha256 TEXT NOT NULL,
                eligible_for_sft INTEGER NOT NULL,expert_market_gold INTEGER NOT NULL,
                execution_allowed INTEGER NOT NULL)""")
            db.execute("""CREATE TABLE manifest(key TEXT PRIMARY KEY,value_json TEXT NOT NULL)""")
            for x in rows:
                db.execute("INSERT INTO curriculum VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",(
                    x["case_id"],json.dumps(x["module_ids"]),x["edition"],
                    json.dumps(x["pdf_pages"]),x["source_sha256"],x["task_type"],
                    x["evidence_class"],json.dumps(x["prompt"],ensure_ascii=False),
                    x["answer"],digest(x),int(x["eligible_for_sft"]),
                    int(x["expert_market_gold"]),int(x["execution_allowed"])))
            for key,value in manifest.items():
                db.execute("INSERT INTO manifest VALUES(?,?)",(
                    key,json.dumps(value,ensure_ascii=False,sort_keys=True)))
            db.commit()
        path.chmod(0o600)
        return manifest
    except Exception:
        path.unlink(missing_ok=True)
        raise


def export_database(path:Path):
    rows,manifest=build()
    if path.exists() or path.is_symlink():
        raise FileExistsError("NEW_VERSIONED_TRAINING_DATABASE_REQUIRED")
    path.parent.mkdir(parents=True,exist_ok=True)
    import os
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_RDWR,0o600)
    os.close(fd)
    try:
        with sqlite3.connect(path) as db:
            db.execute("""CREATE TABLE rules(
                edition TEXT NOT NULL,rule_id TEXT NOT NULL,
                split TEXT NOT NULL,source_sha256 TEXT NOT NULL,
                pdf_pages TEXT NOT NULL,
                review_status TEXT NOT NULL,
                PRIMARY KEY(edition,rule_id))""")
            db.execute("""CREATE TABLE curriculum(
                case_id TEXT PRIMARY KEY,edition TEXT NOT NULL,rule_id TEXT NOT NULL,
                split TEXT NOT NULL,task_type TEXT NOT NULL,
                variant INTEGER NOT NULL,prompt_json TEXT NOT NULL,
                gold_answer TEXT NOT NULL,content_sha256 TEXT NOT NULL,
                eligible_for_sft INTEGER NOT NULL,
                expert_market_gold INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY(edition,rule_id) REFERENCES rules(edition,rule_id))""")
            db.execute("""CREATE TABLE manifest(key TEXT PRIMARY KEY,value_json TEXT NOT NULL)""")
            db.execute("PRAGMA foreign_keys=ON")
            rule_cards={x["id"]:x for x in registry()["rule_cards"]}
            for split,rules in manifest["split_rule_ids"].items():
                for rule_id in rules:
                    c=rule_cards[rule_id]
                    row=next(x for x in rows if x["rule_id"]==rule_id)
                    db.execute("INSERT INTO rules VALUES(?,?,?,?,?,?)",
                        (c["source"],rule_id,split,row["source_sha256"],
                        json.dumps(c["pdf_pages"]),"PAGE_CHECKED_NOT_MARKET_GOLD"))
            for x in rows:
                db.execute("INSERT INTO curriculum VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (x["case_id"],x["edition"],x["rule_id"],x["split"],
                     x["task_type"],x["variant"],
                     json.dumps(x["prompt"],ensure_ascii=False),x["answer"],
                     digest(x),int(x["eligible_for_sft"]),0))
            for key,value in manifest.items():
                db.execute("INSERT INTO manifest VALUES(?,?)",
                    (key,json.dumps(value,ensure_ascii=False,sort_keys=True)))
            db.commit()
        # Guard private DB access even if user umask is lax.
        path.chmod(0o600)
        return manifest
    except Exception:
        path.unlink(missing_ok=True)
        raise


def main():
    parser=argparse.ArgumentParser(description="Build versioned Williams source-only training DB")
    parser.add_argument("--database",type=Path,required=True)
    parser.add_argument("--manifest",type=Path)
    parser.add_argument("--curriculum",choices=("v5","v6-source"),default="v5")
    a=parser.parse_args()
    manifest=export_database(a.database) if a.curriculum=="v5" else export_v6_source_database(a.database)
    if a.manifest:
        if a.manifest.exists():
            raise FileExistsError("NEW_MANIFEST_DESTINATION_REQUIRED")
        a.manifest.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),
                              encoding="utf-8")
    print("WILLIAMS_"+a.curriculum.upper().replace("-","_")+"_SFT_DATABASE",
          json.dumps({"path":str(a.database),"split_rows":manifest["split_rows"],
                      "market_gold":manifest.get("approved_market_labels",0),
                      "expert_market_gold":manifest.get("expert_market_gold",0),
                      "modules":manifest.get("module_coverage",{}),
                      "live_trading":False},ensure_ascii=False,sort_keys=True),
          flush=True)

if __name__=="__main__":
    main()
