"""Fresh objective four-book diagnostic, not independent expert chart gold.

Reuses the existing GGUF examiner and autonomous retrieval. No training, no
orders, no final qualification. All individual responses and the generated
cohort are sealed; only aggregate scores are published.
"""
from collections import Counter
from hashlib import sha256
import argparse
import gc
import json
import math
import os
from pathlib import Path
import random
import secrets

from .knowledge_curriculum_v5 import digest
from .v5_postfit_exam import ExamLlama, BASE_SHA
from .real_gguf_acceptance import actual_gguf_sha256
from .unprompted_source_application import self_retrieve_source

# The first answer in each authored diagnostic triple is the reference.
# The actual model-facing letters are balanced and independently shuffled.
ITEMS = (
("structure","TC1","Закрытые H: 110, 114, 119; L: 100, 103, 108. Какой минимальный вывод о структуре?","Последовательность более высоких максимумов и минимумов","Последовательность более низких минимумов","Установлена завершённая пятая волна"),
("structure","TC1","Закрытые H: 119, 114, 110; L: 108, 103, 100. Какой минимальный вывод?","Более низкие максимумы и минимумы","Более высокие максимумы и минимумы","Подтверждён разворот вверх"),
("alligator","NTD","Jaw рассчитана из закрытых баров и отображается на 8 баров вправо. Какой исходный индекс соответствует отображению t=230?","222","238","230"),
("alligator","NTD","Teeth отображается со сдвигом на 5 баров вправо. Какой исходный индекс соответствует позиции t=91?","86","96","91"),
("angulation","TC2","Наклоны ценовой огибающей и пасти вверх: 4 и 1 единица за бар. Они расходятся. Каков вывод?","Геометрия поддерживает гипотезу раскрытия угла","Параллельные линии с постоянным расстоянием","Раскрытие угла доказано только знаком AO"),
("angulation","TC2","Цена и линия пасти движутся параллельно с наклоном +2, расстояние 30. Что известно об ангуляции?","Большое постоянное расстояние само по себе не доказывает раскрытие угла","Расстояние подтверждает раскрытие угла","Ангуляция доказана направлением цены"),
("ao","NTD","SMA5 медианной цены=127, SMA34=131. Чему равен AO?","-4","4","258"),
("ao","NTD","SMA5 медианной цены=92, SMA34=85. Чему равен AO?","7","-7","177"),
("ac","NTD","AO=-3, SMA5(AO)=-5. Чему равен AC?","2","-2","-8"),
("ac","NTD","AO=8, SMA5(AO)=11. Чему равен AC?","-3","3","19"),
("fractal","TC1","Максимумы пяти закрытых баров: 18,21,29,25,24. Какой верхний фрактал подтверждён?","29","25","21"),
("fractal","TC1","Минимумы: 15,13,8,12,14; пятый бар ещё открыт. Что известно о нижнем фрактале 8?","Не подтверждён вторым закрытым правым баром","Подтверждён первым правым баром","Подтверждён ценой 14"),
("waves","COURSE","При предполагаемой W4 осциллятор 5/34: 4,0,2. Был ли переход за ноль?","Нет, лишь касание","Да, отрицательная область достигнута","Да, W4 завершена"),
("waves","COURSE","При предполагаемой W4 осциллятор 5/34: 4,1,-2. Что доказывает этот факт?","Переход за ноль, но не сам по себе окончание W4","Окончание W4 и готовый вход","Касание без перехода"),
("wm1","TC2","Предыдущий минимум 90, текущий закрытый бар L=88 H=104 C=100. Какая геометрия?","Новый нижний минимум и закрытие выше середины","Новый минимум и закрытие ниже середины","Закрытие ровно на середине"),
("wm1","TC2","Предыдущий максимум 105, текущий закрытый бар L=90 H=108 C=94. Какая геометрия?","Новый верхний максимум и закрытие ниже середины","Новый максимум и закрытие выше середины","Не обновлён максимум"),
("wm2","TC2","AO закрытых баров: -8,-6,-4,-1. Сколько последовательных повышений наблюдается?","Три","Два","Четыре"),
("wm2","TC2","AO закрытых баров: 9,6,3,4. Сколько последовательных понижений непосредственно перед последним закрытием?","Ноль, последний столбец вырос","Два, последняя смена направления не важна","Три"),
("wm3","TC2","Подтверждённый верхний фрактал 200; тик 0.5. При касании 200.5 Teeth=201. Условие пробоя выше Teeth?","Не выполнено","Выполнено","Выполнено при одном подтверждающем баре"),
("wm3","TC2","Подтверждённый нижний фрактал 80; тик 0.5. При касании 79.5 Teeth=81. Условие пробоя ниже Teeth?","Выполнено; биржевое исполнение из этого не следует","Не выполнено, Teeth выше цены","Выполнено только при Teeth=79.5"),
("wise_interaction","TC2","Может ли WM2 быть первым допустимым сигналом TC2 при выполнении его условий?","Да","Нет, всегда только после WM1","Только после NTD внешнего фрактала"),
("wise_interaction","TC2","Появился противоположный Мудрец, но его STOP-уровень не достигнут. Что известно?","Сигнал наблюдается, срабатывание не доказано","Разворот уже исполнен","Первоначальная позиция обязательно закрыта"),
("multitimeframe","TC2","По заданной политике D1/H4 — контекст, H1 — решения, M15/M5 — наблюдение. H1 сигнал подтверждён, M15 не подтвердил. Как применять именно это правило?","Отсутствие подтверждения M15 само по себе не блокирует","M15 всегда запрещает вход","Решение принимает только M5"),
("multitimeframe","COURSE","Старший моментум вниз, текущий вверх. Как минимально зафиксировать контекст?","Раздельно записать противоречащие направления","Оба направления вверх","Оба направления вниз"),
("management","TC2","Для LONG выбрано сопровождение по 3 завершённым барам: минимумы 98,101,99. Тик 0.5. Уровень на тик ниже их минимального экстремума?","97.5","100.5","98.5"),
("management","TC2","Для SHORT выбрано сопровождение по 5 завершённым барам: максимумы 105,103,107,102,101. Тик 0.5. Уровень на тик выше максимума?","107.5","102.5","106.5"),
("risk","TC2","Пользовательский лимит дневного убытка 3% от заданной базы 2000 USDT. Какова величина лимита?","60 USDT","100 USDT","30 USDT"),
("risk","TC2","Пользовательская суммарная маржа ограничена 50% от заданного баланса 2400 USDT. Каков лимит?","1200 USDT","2400 USDT","240 USDT"),
("uncertainty","TC1","Есть одиночная дивергенция, степень волн неизвестна. Какой вывод научно допустим?","Гипотеза, требующая проверки структуры и степени","Доказанная старшая W5","Безусловный сигнал разворота"),
("uncertainty","TC2","Есть подтверждённая свеча, но отсутствуют данные о положении Teeth на момент пробоя WM3. Что утверждать?","Условие активации нельзя подтвердить по этим данным","Условие подтверждено по Teeth при формировании","Условие всегда выполнено"),
)


def build(seed):
    rng=random.Random(seed)
    letters=list("ABC")*(len(ITEMS)//3)
    rng.shuffle(letters)
    rows=[]
    for i,(skill,edition,question,correct,wrong1,wrong2) in enumerate(ITEMS):
        if len({correct,wrong1,wrong2})!=3:
            raise ValueError("DUPLICATED_OPTIONS")
        letter=letters[i]
        wrong=[wrong1,wrong2]
        rng.shuffle(wrong)
        choices={x:correct if x==letter else wrong.pop() for x in "ABC"}
        rows.append({"competency":skill,"expected":letter,
                     "public":{"id":f"DIAG-{i:03d}","edition":edition,
                               "question":question,"choices":choices}})
    rng.shuffle(rows)
    return rows


def grade_choice(choice, expected, finish_reason):
    if finish_reason=="length":
        return "TECHNICAL_TRUNCATION"
    if choice not in ("A","B","C"):
        return "TECHNICAL_OR_UNPARSED"
    return "CORRECT" if choice==expected else "INCORRECT"


class CaptureExam(ExamLlama):
    def _learning_completion(self, *, messages, schema, max_tokens, retry_exam_only=False):
        messages=[dict(m) for m in messages]
        messages[0]["content"]+=" /no_think"
        result=self._model.create_chat_completion(
            messages=messages, response_format={"type":"json_object","schema":schema},
            temperature=0.0,seed=1942,max_tokens=min(max_tokens,128),stream=False)
        choices=result.get("choices",[])
        first=choices[0] if choices else {}
        self.raw_capture={"finish_reason":first.get("finish_reason"),
                          "content":first.get("message",{}).get("content")}
        return result


def run(base,adapter,receipt_path,output):
    from scripts.ai_submit_v5_qlora_kaggle import _seal_data
    from scripts.v5_recovery_integrity import EXPECTED_ADAPTER_SHA256
    secret=os.environ.get("KAGGLE_API_TOKEN","")
    if len(secret)<20 or output.exists():
        raise ValueError("PRIVATE_NEW_EXAM_DESTINATION_REQUIRED")
    receipt=json.loads(receipt_path.read_text())
    if receipt["adapter_sha256"]!=EXPECTED_ADAPTER_SHA256:
        raise ValueError("FROZEN_ADAPTER_REQUIRED")
    actual_gguf_sha256(base,BASE_SHA)
    seed=secrets.randbits(64)
    rows=build(seed)
    frozen=digest(rows)
    output.mkdir(mode=0o700,parents=True,exist_ok=False)
    (output/"frozen-cohort.json.sealed").write_bytes(
        _seal_data(json.dumps({"seed":seed,"rows":rows},ensure_ascii=False).encode(),secret))
    print("OBJECTIVE_DIAGNOSTIC_FROZEN",frozen,flush=True)
    public={}
    private=[]
    for model_id,lora in (("v5",adapter),("base",None)):
        model=CaptureExam(base=base,lora=lora,thinking=False)
        for mode in ("closed_book","open_book"):
            stats=Counter()
            groups={}
            for row in rows:
                case=row["public"]
                model.raw_capture={}
                retrieved=[]
                if mode=="open_book":
                    retrieved=self_retrieve_source(case["question"],case["edition"],
                        retriever="weighted_v2")["retrieved"]
                lessons=[{"source":x["source"],"excerpt":x["excerpt"]} for x in retrieved]
                error=None
                try:
                    answer=model.answer_exam_case(dict(case),prior_lessons=lessons or None)
                    grade=grade_choice(answer.get("choice"),row["expected"],model.raw_capture.get("finish_reason"))
                except Exception as exc:
                    error=type(exc).__name__
                    grade="TECHNICAL_EXCEPTION"
                    answer={}
                stats[grade]+=1
                for group in ("edition:"+case["edition"],"skill:"+row["competency"]):
                    groups.setdefault(group,Counter())[grade]+=1
                private.append({"model":model_id,"mode":mode,"id":case["id"],
                    "parsed":answer,"raw":model.raw_capture,"error_type":error,"grade":grade,
                    "retrieved_rule_ids":[x["rule_id"] for x in retrieved]})
            key=model_id+":"+mode
            public[key]={"total":len(rows),"counts":dict(stats),
                        "by_group":{k:dict(v) for k,v in groups.items()}}
            print("OBJECTIVE_DIAGNOSTIC_RESULT",key,json.dumps(dict(stats),sort_keys=True),flush=True)
            # Preserve completed cohorts if later inference fails.
            (output/(key.replace(":","-")+".json.sealed")).write_bytes(
                _seal_data(json.dumps([p for p in private if p["model"]==model_id and p["mode"]==mode],
                                      ensure_ascii=False).encode(),secret))
        del model
        gc.collect()
    report={"schema":"WILLIAMS_OBJECTIVE_DIAGNOSTIC_V1",
            "frozen_cohort_sha256":frozen,"adapter_sha256":receipt["adapter_sha256"],
            "gguf_lora_sha256":sha256(adapter.read_bytes()).hexdigest(),
            "base_gguf_sha256":BASE_SHA,"results":public,
            "developer_authored_diagnostic":True,"independent_expert_gold":False,
            "historical_chart_exam":False,"new_optimizer_steps":0,
            "execution_allowed":False,"qualified":False}
    (output/"aggregate.json").write_text(json.dumps(report,indent=2,sort_keys=True))
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ("base","adapter","receipt","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    run(args.base,args.adapter,args.receipt,args.output)
