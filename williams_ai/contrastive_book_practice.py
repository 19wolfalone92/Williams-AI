"""Paired authored rule lessons, value-focused loss and strict output grading.

These are calculated synthetic exercises, not expert-labeled market charts.
The old24 questions remain a development diagnostic after their first exam.
"""
from __future__ import annotations
from collections import Counter
from decimal import Decimal,ROUND_FLOOR
from hashlib import sha256
import json
import math
import random
import re

SYSTEM="Ты решаешь проверочное задание. Следуй требуемому JSON-формату, не придумывай отсутствующие факты."
RULES=("TC1.PSY.PLAN","TC2.WM1.TRIGGER","COURSE.HUMP","COURSE.W4.ZERO","COURSE.AO5345","NTD.AC","TC2.WM3")


def paired_cases(split:str)->list[dict]:
    ranges={"train":range(1,33),"validation":range(1001,1009),"test":range(3001,3005)}
    if split not in ranges:raise ValueError("DECLARED_CONTRASTIVE_SPLIT_REQUIRED")
    rows=[]
    def add(n,variant,edition,rule,page,text,expected):
        expected={**expected,"execution_allowed":False}
        types=", ".join(k+": "+("true/false" if type(v) is bool else "число" if isinstance(v,(int,float)) else "строка") for k,v in expected.items())
        rows.append({"case_id":f"paired-{split}-{n}-{rule}-{variant}","pair_id":f"{split}-{n}-{rule}",
            "edition":edition,"rule_id":rule,"pdf_page":page,
            "question":text+" Исследование без ордеров. Один JSON-объект: "+types+". execution_allowed=false. Без других полей и пояснений.",
            "answer":json.dumps(expected,ensure_ascii=False,separators=(",",":"))})
    for n in ranges[split]:
        rng=random.Random(817331+n)
        equity=Decimal(rng.randrange(500,20001));fraction=Decimal(rng.choice(("0.0025","0.004","0.005","0.006","0.007","0.008","0.01")))
        distance=Decimal(rng.randrange(5,101))/10;step=Decimal(rng.choice(("0.01","0.02","0.03","0.05","0.1")))
        tick=Decimal(rng.choice(("0.01","0.05","0.1","0.25","0.5")))
        low=Decimal(rng.randrange(1000,20001))*tick;high=low+Decimal(rng.randrange(5,200))*tick
        end=100+n*17;gap=rng.randrange(1,5)
        hist=Decimal(rng.randrange(-200,201))/100+Decimal(n)/100000
        delta=Decimal(rng.randrange(1,21))/100
        wave_sign=rng.choice((-1,1));before=Decimal(wave_sign)*(Decimal(rng.randrange(5,31))/10+Decimal(n)/100000)
        ac_size=Decimal(rng.randrange(1,101))/100+Decimal(n)/100000
        green=rng.choice((1,2,2,2,3,4));right=1
        for v in (0,1):
            d=distance*(v+1);budget=equity*fraction
            qty=(budget/d/step).to_integral_value(rounding=ROUND_FLOOR)*step
            add(n,v,"TC1","TC1.PSY.PLAN",83,
                f"Equity={equity} USDT, доля риска={fraction}, расстояние entry-stop={d}, шаг количества={step}. "
                "Пункт стоит 1 USDT, издержки исключены для этой арифметики. Вычисли risk_budget и quantity, округляя количество вниз к шагу.",
                {"risk_budget":float(budget),"quantity":float(qty)})
            long=v==0
            add(n,v,"TC2","TC2.WM1.TRIGGER",124,
                f"TC2: закрытый {'бычий' if long else 'медвежий'} WM1 и ангуляция подтверждены. Low={low}, high={high}, tick={tick}. "
                "Назови entry на один tick за экстремумом по направлению и stop на tick за противоположным. Заявка не исполнялась.",
                {"entry":float(high+tick if long else low-tick),"stop":float(low-tick if long else high+tick)})
            begin=end+gap if v else end-(0 if n%4==0 else gap)
            add(n,v,"COURSE","COURSE.HUMP",72,
                f"Course PDF72: крайняя первая формация включает бары до {end}, вторая — с {begin}; обе границы включены. "
                "Оба противоположных фрактала подтверждены, прочие условия выполнены. Проверь только hump_valid по неперекрытию.",
                {"hump_valid":begin>end})
            after=Decimal(wave_sign if not v else -wave_sign)*Decimal(rng.randrange(1,10))/10
            add(n,v,"COURSE","COURSE.W4.ZERO",44,
                f"Course PDF44, масштаб 120 баров, после предполагаемой {'бычьей' if wave_sign>0 else 'медвежьей'} W3 "
                f"закрытые значения 5/34 изменились с {before} на {after}. Завершение младшей c/e не подтверждено. "
                "Укажи minimum_w4_met и w4_finished: минимум и конец коррекции различаются.",
                {"minimum_w4_met":before*after<0,"w4_finished":False})
            signal=hist+(delta if v else -delta)
            add(n,v,"COURSE","COURSE.AO5345",45,
                f"Профиль Course PDF45: гистограмма 5/34={hist}, её собственная SMA(5)={signal}. "
                "Укажи momentum LONG/SHORT по сравнению этих двух величин. Структура входа не дана.",
                {"momentum":"LONG" if hist>signal else "SHORT"})
            ac=ac_size if v else -ac_size;required=2 if ac>0 else 3
            add(n,v,"NTD","NTD.AC",75,
                f"NTD: покупка по AC, значение AC={ac}, подряд зелёных столбцов={green}, перед ними красный. AO=0.8. "
                "Укажи required_green и signal_ready, используя знак AC.",
                {"required_green":required,"signal_ready":green>=required})
            right=1+v
            add(n,v,"TC2","TC2.WM3",147,
                f"Строгий пятибарный фрактальный HIGH={100+n} выше всех допустимых соседних HIGH. Два левых бара закрыты; "
                f"справа закрыто {right} подходящих баров. Неподтверждённые правые данные запрещены. Укажи confirmed сейчас.",
                {"confirmed":right==2})
    if len({r["question"] for r in rows})!=len(rows):raise ValueError("CONTRASTIVE_QUESTION_DUPLICATE")
    return rows


def quick_development_cases()->list[dict]:
    rows=paired_cases("validation")
    return [r for rule in RULES for r in [x for x in rows if x["rule_id"]==rule][:2]]


def strict_answer(text:str):
    def unique(items):
        result={}
        for k,v in items:
            if k in result:raise ValueError("DUPLICATE_JSON_FIELD")
            result[k]=v
        return result
    try:
        obj=json.loads(text,object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
        return obj if isinstance(obj,dict) else None
    except (ValueError,TypeError):return None


def correct_answer(response:str,answer:str)->bool:
    from .structured_transfer_exam import _matches
    expected=json.loads(answer);actual=strict_answer(response)
    return actual is not None and set(actual)==set(expected) and all(_matches(actual[k],v) for k,v in expected.items())


def score_responses(cases:list[dict],responses:list[dict])->dict:
    by_id={r["case_id"]:r for r in cases}
    if (len(by_id)!=len(cases) or len(responses)!=len(cases)
        or len({r["case_id"] for r in responses})!=len(responses)
        or set(by_id)!={r["case_id"] for r in responses}):raise ValueError("EXACT_CONTRASTIVE_COHORT_REQUIRED")
    by_rule={rule:[] for rule in RULES};rows=[];observed={rule:[] for rule in RULES};expected_values={rule:[] for rule in RULES}
    for result in responses:
        case=by_id[result["case_id"]];ok=correct_answer(result["response"],case["answer"])
        rows.append({"case_id":case["case_id"],"rule_id":case["rule_id"],"correct":ok,
            "valid_json":strict_answer(result["response"]) is not None})
        by_rule[case["rule_id"]].append(ok)
        observed[case["rule_id"]].append(strict_answer(result["response"]))
        expected_values[case["rule_id"]].append(json.loads(case["answer"]))
    counts=Counter(" ".join(r["response"].split()) for r in responses)
    constant_fields={}
    for rule,values in expected_values.items():
        if not values:continue
        for key in values[0]:
            expected={json.dumps(v[key],sort_keys=True) for v in values}
            actual={json.dumps(v.get(key),sort_keys=True) for v in observed[rule] if v is not None}
            if len(expected)>1 and len(actual)==1:constant_fields.setdefault(rule,[]).append(key)
    return {"correct":sum(r["correct"] for r in rows),"total":len(rows),
        "accuracy":sum(r["correct"] for r in rows)/len(rows),
        "per_rule":{rule:{"correct":sum(v),"total":len(v)} for rule,v in by_rule.items() if v},
        "distinct_responses":len(counts),"largest_identical_response_count":max(counts.values()),
        "constant_fields_despite_varied_expected_values":constant_fields,
        "cases":rows,"expert_market_chart_exam":False,"execution_allowed":False}


def question_rows(cases:list[dict])->list[dict]:
    return [{"case_id":r["case_id"],"domain":r["rule_id"],"prompt":r["question"]} for r in cases]


def questions_digest(questions:list[dict])->str:
    return sha256(json.dumps(questions,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def value_spans(answer:str)->list[tuple[int,int,float]]:
    obj=strict_answer(answer)
    if obj is None or not obj:raise ValueError("FLAT_TYPED_ANSWER_REQUIRED")
    pattern=r'"([^"\\]+)"\s*:\s*("(?:[^"\\]|\\.)*"|true|false|null|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)'
    found=list(re.finditer(pattern,answer))
    if {m.group(1) for m in found}!=set(obj):raise ValueError("FLAT_JSON_VALUES_REQUIRED")
    return [(m.start(2),m.end(2),2.0 if m.group(1)=="execution_allowed" else 8.0) for m in found]


def tokenize_typed(rows,tokenizer,max_tokens=768):
    result=[]
    for row in rows:
        messages=[{"role":"system","content":SYSTEM},{"role":"user","content":row["question"]}]
        prefix=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
        full_messages=messages+[{"role":"assistant","content":row["answer"]}]
        rendered=tokenizer.apply_chat_template(full_messages,tokenize=False,add_generation_prompt=False,enable_thinking=False)
        encoded=tokenizer(rendered,add_special_tokens=False,return_offsets_mapping=True)
        ids=encoded["input_ids"];offsets=encoded["offset_mapping"]
        if ids[:len(prefix)]!=prefix or len(ids)>max_tokens:raise ValueError("EXACT_TYPED_TEMPLATE_NO_TRUNCATION_REQUIRED")
        start=rendered.rfind(row["answer"])
        if start<0:raise ValueError("COMPLETE_TYPED_ANSWER_IN_TEMPLATE_REQUIRED")
        spans=[(a+start,b+start,w) for a,b,w in value_spans(row["answer"])]
        weights=[0.0]*len(prefix)
        for a,b in offsets[len(prefix):]:
            weights.append(max([1.0]+[w for lo,hi,w in spans if a<hi and b>lo]))
        result.append({"input_ids":ids,"attention_mask":[1]*len(ids),
            "labels":[-100]*len(prefix)+ids[len(prefix):],"loss_weights":weights})
    return result


def weighted_causal_loss(logits,labels,weights):
    """Equal example contribution; result-value tokens get explicit weight."""
    import torch
    import torch.nn.functional as F
    shifted=labels[...,1:].contiguous();w=weights[...,1:].float()*(shifted!=-100)
    if bool((w<0).any()) or not bool(torch.isfinite(w).all()) or bool((w.sum(-1)<=0).any()):
        raise ValueError("FINITE_POSITIVE_SUPERVISION_WEIGHTS_REQUIRED")
    losses=F.cross_entropy(logits[...,:-1,:].float().reshape(-1,logits.shape[-1]),shifted.reshape(-1),
        reduction="none",ignore_index=-100).reshape(shifted.shape)
    return ((losses*w).sum(-1)/w.sum(-1)).mean()
