"""Authored structured-output practice; no real-chart expert-label claim."""
from decimal import Decimal, ROUND_FLOOR
import json


def structured_rule_cases():
    result={s:[] for s in ("train","validation")}
    def add(split,edition,rule,page,question,answer):
        answer={**answer,"execution_allowed":False}
        result[split].append({"edition":edition,"rule_id":rule,"pdf_page":page,
            "question":question+" Исследование. Верни ровно один JSON-объект с полями "+", ".join(answer)+
                "; числовые значения — JSON numbers, логические — true/false. execution_allowed=false. Без пояснений и дополнительных полей.",
            "answer":json.dumps(answer,ensure_ascii=False,separators=(",",":"))})
    for split,values in (("train",range(1,17)),("validation",range(91,99))):
        for n in values:
            equity=Decimal(1100+n*23);fraction=Decimal(("0.004","0.005","0.006")[n%3])
            distance=Decimal("0.9")+Decimal(n%7)*Decimal("0.3")
            step=Decimal(("0.01","0.02","0.05")[n%3]);budget=equity*fraction
            qty=(budget/distance/step).to_integral_value(rounding=ROUND_FLOOR)*step
            add(split,"TC1","TC1.PSY.PLAN",83,
                f"Учебная арифметика риска: equity={equity}, доля риска={fraction}, расстояние entry-stop={distance}, "
                f"шаг количества={step}, стоимость пункта=1. Издержки исключены только для вычисления. quantity округли вниз к шагу.",
                {"risk_budget":float(budget),"quantity":float(qty)})
            low=Decimal(81+n*3);high=low+Decimal("4.2");tick=Decimal("0.1");long=n%2==1
            add(split,"TC2","TC2.WM1.TRIGGER",124,
                f"Закрытый {'бычий' if long else 'медвежий'} WM1 и ангуляция подтверждены отдельно. Low={low}, high={high}, tick={tick}. "
                "Нужны stop-entry за экстремумом по направлению и защитный stop за противоположным. Заявка ещё не исполнена.",
                {"entry":float(high+tick if long else low-tick),"stop":float(low-tick if long else high+tick)})
            end=n*10+6;begin=end-1 if n%3==0 else end+1
            add(split,"COURSE","COURSE.HUMP",72,
                f"Оба противоположных фрактала подтверждены. Крайние формации: первая заканчивается на баре {end}, "
                f"вторая начинается с бара {begin}, границы включены. Остальное считается выполненным. Проверь только Hump-неперекрытие.",
                {"hump_valid":begin>end})
            crossed=n%2==0
            add(split,"COURSE","COURSE.W4.ZERO",44,
                f"После предполагаемой W3 масштаб=120 баров, значения 5/34 изменились с {1+Decimal(n)/100} на {(-0.1 if crossed else 0.1)}. "
                "Конец младшей c/e не подтверждён. Различи минимум W4 и её завершение.",
                {"minimum_w4_met":crossed,"w4_finished":False})
            hist=Decimal(n)/100-Decimal("0.5");signal=hist+(Decimal("0.2") if n%2 else Decimal("-0.2"))
            add(split,"COURSE","COURSE.AO5345",45,
                f"Профиль Course: 5/34={hist}, собственная SMA(5)={signal}. momentum — LONG либо SHORT; данных входа нет.",
                {"momentum":"LONG" if hist>signal else "SHORT"})
            ac=(Decimal("0.1")+Decimal(n)/1000)*(-1 if n%2 else 1);green=2 if n%3 else 3
            required=3 if ac<0 else 2
            add(split,"NTD","NTD.AC",75,
                f"NTD AC покупка. AC={ac}, подряд зелёных столбцов={green}, перед ними красный. AO=0.8. "
                "Определи required_green и signal_ready по знаку AC.",
                {"required_green":required,"signal_ready":green>=required})
            right=n%3
            add(split,"TC2","TC2.WM3",147,
                f"Строгий фрактальный HIGH={100+n} превышает HIGH всех допустимых соседей. Два левых бара закрыты, справа закрыто {right} подходящих баров. "
                "Неподтверждённых правых данных нет. Определи confirmed сейчас.",{"confirmed":right==2})
    # Repeated rule forms intentionally teach output contracts, not independent
    # market observations; numeric parameters and full questions stay separate.
    if {r["question"] for r in result["train"]}&{r["question"] for r in result["validation"]}:
        raise ValueError("STRUCTURED_TRAIN_VALIDATION_QUESTION_OVERLAP")
    return result


def balanced_recovery_curriculum():
    from .deep_book_research import practice_curriculum,corrective_course_practice
    train,val,audit=practice_curriculum()
    kept=[];numeric_counts={"risk":0,"wm1":0}
    for row in train:
        kind=("risk" if row["question"].startswith("Расчёт учебной позиции") else
              "wm1" if row["question"].startswith("Учебный WM1 ") else None)
        if kind:
            if numeric_counts[kind]>=12:continue
            numeric_counts[kind]+=1
        kept.append(row)
    corrections=corrective_course_practice();typed=structured_rule_cases()
    kept.extend(corrections["train"][:48]);kept.extend(typed["train"])
    val["COURSE"].extend(corrections["validation"])
    for row in typed["validation"]:val[row["edition"]].append(row)
    audit.update({"mode":"BALANCED_AUTHORED_EXPLANATIONS_CORRECTIONS_AND_TYPED_RULE_PRACTICE",
        "train_examples":len(kept),"validation_examples":sum(map(len,val.values())),
        "structured_train_examples":len(typed["train"]),"structured_validation_examples":len(typed["validation"]),
        "corrective_course_examples":48,"synthetic_arithmetic_examples_kept":numeric_counts,
        "same_rule_can_appear_in_train_and_validation":True,"independent_expert_chart_labels":False,
        "structured_examples_are_not_independent_market_observations":True})
    return kept,val,audit
