"""Narrow four-book objective application pilot, not expert market labels.

Use explicit deterministic conditions derived from page-checked V5 TRAIN
rules. No V6 unapproved candidate labels, original PDFs, historical markets,
or V5 frozen validation/test rule IDs are admitted. Numerical development
partitions are different cases of the SAME skills, NOT independent expert
exams. No market entry is implied by a correct arithmetic answer.
"""
from collections import Counter
import json
import random

from .knowledge_curriculum_v5 import build, digest, EDITIONS
from .book_rules import registry

SCHEMA = "WILLIAMS_V6_OBJECTIVE_APPLICATION_DEVELOPMENT_V1"
SPLITS = ("train", "validation", "development")
# All selected rules are in frozen V5 train, not validation/test.
FAMILIES = {
    "fractal": ("TC1.FRACTAL", (68,69)),
    "ao_difference": ("NTD.AO", (55,)),
    "zone": ("NTD.ZONE", (91,)),
    "wm1_shape": ("TC2.WM1.BAR", (121,122)),
    "wm1_protection": ("TC2.WM1.STOP", (131,)),
    "wm2_direction": ("TC2.WM2", (137,139)),
    "wm3_activation": ("TC2.WM3", (147,148,149)),
    "course_momentum": ("COURSE.AO5345", (42,45)),
}
SYSTEM = ("Исследование методологии Билла Вильямса. Реши только заданную "
          "локальную проверку, не утверждай наличие сделки. Ответ строго "
          "одним JSON-объектом с единственным полем answer. "
          "Числа записывай целыми числами; для перечисленных вариантов "
          "используй точный код. Не добавляй пояснений в этом упражнении.")


def case(family, split, index):
    if family not in FAMILIES or split not in SPLITS or type(index) is not int or not 0<=index<16:
        raise ValueError("INVALID_OBJECTIVE_PRACTICE_CASE")
    # Different deterministic seeds and disjoint numeric anchors; no public
    # final exam claim, and no reuse of this exposed generator for that exam.
    slot=SPLITS.index(split)
    r=random.Random(281124+10000*slot+200*list(FAMILIES).index(family)+index)
    p=1000+1000*slot+40*index+r.randint(0,15)
    direction=1 if index%2==0 else -1
    valid=index%4 in (0,1)
    if family=="fractal":
        side="UP" if direction==1 else "DOWN"
        a=[p-direction*4,p-direction*2,p,p-direction*3,p-direction*1]
        closed=valid
        if not valid and index%8 in (2,3):
            closed=True
            a[-1]=p+direction*2
        q=(f"Пять {'максимумов' if direction==1 else 'минимумов'} последовательных баров: {a}. "
           f"Все пять закрыты: {'да' if closed else 'нет (пятый открыт)'}. "
           f"Проверяется только строгий фрактал {side} на третьем баре, без равных экстремумов. "
           "Ответ UP, DOWN или NONE.")
        gold=side if closed and all(direction*(a[2]-a[k])>0 for k in (0,1,3,4)) else "NONE"
    elif family=="ao_difference":
        d=direction*r.randint(2,17)
        q=f"Уже рассчитанные SMA5 и SMA34 медианных цен равны {p+d} и {p}. Чему равна их разность AO? Ответ целым числом."
        gold=d
    elif family=="zone":
        ao0=direction*(p//5)
        ac0=-ao0+7
        a=direction*r.randint(1,9)
        b=(direction if valid else -direction)*r.randint(1,9)
        q=(f"NTD: AO изменился с {ao0} до {ao0+a}, AC с {ac0} до {ac0+b}. "
           "Только цвет зоны: GREEN, RED или GRAY.")
        gold="GREEN" if a>0 and b>0 else "RED" if a<0 and b<0 else "GRAY"
    elif family=="wm1_shape":
        lo=p;hi=p+12;mid=(lo+hi)//2
        old=lo+2 if direction==1 else hi-2
        close=mid+direction*(3 if valid else -3)
        q=(f"TC2: предыдущий {'минимум' if direction==1 else 'максимум'} {old}. "
           f"Закрытый текущий бар L={lo}, H={hi}, C={close}. "
           "Проверяй только форму разворотного бара, не ангуляцию и не вход. "
           f"Форма {'бычья' if direction==1 else 'медвежья'}: YES или NO?")
        gold="YES" if (lo<old and close>mid if direction==1 else hi>old and close<mid) else "NO"
    elif family=="wm1_protection":
        lo=p;hi=p+12;tick=r.choice((1,2,3))
        q=(f"В учебной гипотезе WM1 {'LONG' if direction==1 else 'SHORT'} уже допустим. "
           f"Сигнальный бар L={lo}, H={hi}; заданный отступ {tick} целых ценовых единиц. "
           "Рассчитай первоначальный защитный уровень за противоположным экстремумом. Ответ целым числом.")
        gold=lo-tick if direction==1 else hi+tick
    elif family=="wm2_direction":
        first=-direction*(p//5)
        a=[first,first+direction,first+2*direction,first+(3 if valid else 1)*direction]
        q=(f"Четыре закрытых значения AO: {a}. Проверяется только наличие трех последовательных "
           f"{'повышений' if direction==1 else 'снижений'}, не полноценный WM2. Ответ YES или NO.")
        gold="YES" if all(direction*(b-a0)>0 for a0,b in zip(a,a[1:])) else "NO"
    elif family=="wm3_activation":
        price=p+direction
        formed=p-direction*3
        teeth=price-direction*(2 if valid else -2)
        q=(f"TC2: подтвержденный фрактал {'покупки' if direction==1 else 'продажи'} имел "
           f"Teeth={formed} при формировании. В момент предполагаемого пробоя цена={price}, "
           f"актуальные Teeth={teeth}. Остальные условия не оцениваются. "
           "Выполняется ли именно условие отношения цены к Teeth при срабатывании? YES или NO.")
        gold="YES" if direction*(price-teeth)>0 else "NO"
    else:
        signal=direction*(p//5)
        osc=signal+direction*r.randint(1,11)
        q=(f"Profitunity Course: значение 5/34 = {osc}, его сигнальная SMA5 = {signal}. "
           "Определи направление относительного моментума этого профиля, не торговое решение. UP или DOWN.")
        gold="UP" if osc>signal else "DOWN"
    rule,pages=FAMILIES[family]
    return {"case_id":f"OBJ-{split}-{family}-{index:02d}","split":split,
            "family":family,"rule_id":rule,"edition":rule.split(".")[0],
            "pdf_page":pages[0],"pdf_pages":list(pages),"question":q,
            "answer":json.dumps({"answer":gold},ensure_ascii=False,separators=(",",":")),
            "market_expert_gold":False,"historical_chart":False,
            "execution_allowed":False,"numeric_anchor":p}


def curriculum():
    old,manifest=build()
    frozen=set(manifest["split_rule_ids"]["validation"])|set(manifest["split_rule_ids"]["test"])
    train_rules=set(manifest["split_rule_ids"]["train"])
    cards={r["id"]:r for r in registry()["rule_cards"]}
    for rule,pages in FAMILIES.values():
        if rule not in train_rules or rule in frozen or cards[rule]["status"]!="REVIEWED_PAGES" or not set(pages)<=set(cards[rule]["pdf_pages"]):
            raise ValueError("PILOT_RULE_NOT_PAGE_CHECKED_FROZEN_TRAIN")
    rows=[case(family,split,i) for split,count in zip(SPLITS,(8,4,4))
          for family in FAMILIES for i in range(count)]
    if len({r["question"] for r in rows})!=len(rows) or len({r["case_id"] for r in rows})!=len(rows):
        raise ValueError("DUPLICATE_OBJECTIVE_PILOT_CASE")
    if set(r["edition"] for r in rows)!=set(EDITIONS):
        raise ValueError("FOUR_BOOK_PILOT_REQUIRED")
    return rows,{"schema":SCHEMA,"rows":len(rows),"split_rows":dict(Counter(r["split"] for r in rows)),
        "families":list(FAMILIES),"frozen_registry_sha256":manifest["source_registry_sha256"],
        "dataset_sha256":digest(rows),"frozen_v5_split_rule_ids_preserved":True,
        "independent_expert_review":False,"independent_final_exam":False,
        "same_skill_development_partitions":True,"candidate_labels_used":0,
        "historical_chart_rows":0,"execution_allowed":False}


def audit_v5():
    """Quantify task shortcuts without changing the frozen curriculum."""
    rows,manifest=build()
    results={}
    for part in ("train","validation","test"):
        subset=[r for r in rows if r["split"]==part]
        mcq=[r for r in subset if r["task_type"]=="MCQ"]
        boundary=[r for r in subset if r["task_type"]=="BOUNDARY"]
        longest=generic=0
        for row in mcq:
            choices=row["prompt"]["choices"];gold=json.loads(row["answer"])["choice"]
            lengths=[k for k,v in choices.items() if len(v)==max(map(len,choices.values()))]
            generic_choices=[k for k,v in choices.items() if not v.startswith(
                ("Достаточно заметить термин","Любой сигнал из"))]
            longest+=lengths==[gold]
            generic+=generic_choices==[gold]
        results[part]={"rows":len(subset),"unique_rules":len({r["rule_id"] for r in subset}),
            "unique_exact_answers":len({r["answer"] for r in subset}),
            "mcq_rows":len(mcq),"longest_choice_correct":longest,
            "generic_distractor_shortcut_correct":generic,
            "boundary_rows":len(boundary),
            "boundary_answer_evidence_in_prompt":sum(
                r["prompt"]["reviewed_source_context"]["paraphrased_rule"]==
                json.loads(r["answer"])["evidence"] for r in boundary)}
    return {"source_registry_sha256":manifest["source_registry_sha256"],"splits":results,
            "historical_v5_frozen_unchanged":True,"shortcut_is_dataset_defect_not_causal_proof":True,
            "execution_allowed":False}


if __name__=="__main__":
    print(json.dumps({"v5_audit":audit_v5(),"objective_pilot":curriculum()[1]},ensure_ascii=False,sort_keys=True))
