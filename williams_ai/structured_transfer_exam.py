"""Sealed generative rule/arithmetic transfer test, not trading qualification.

The GPU receives questions only. Expected JSON values stay on the CPU grader.
The same prompts compare base Qwen, the first private adapter, and its corrected
continuation. This checks actual generated text rather than only text loss or
candidate-token scores; independently expert-labeled market waves are absent.
"""
from __future__ import annotations
from collections import Counter
from decimal import Decimal, ROUND_FLOOR
from hashlib import sha256
import json
import math
from pathlib import Path


def sealed_cases() -> tuple[list[dict], list[dict]]:
    questions, keys = [], []
    def add(ident, domain, text, expected):
        expected = {**expected, "execution_allowed": False}
        schema = ", ".join(expected)
        questions.append({"case_id": ident, "domain": domain,
            "prompt": text + f" Исследование без ордеров. Ответ — один JSON-объект с полями {schema}; execution_allowed=false. Без других полей и пояснений."})
        keys.append({"case_id": ident, "domain": domain, "expected": expected})
    for i, (equity, pct, distance, step) in enumerate(((10000, "0.007", "2.5", "0.02"), (7123, "0.006", "3.7", "0.05"),
                                                      (1800, "0.004", "0.8", "0.03"), (9500, "0.008", "7.3", "0.1"))):
        budget = Decimal(equity) * Decimal(pct)
        qty = (budget / Decimal(distance) / Decimal(step)).to_integral_value(rounding=ROUND_FLOOR) * Decimal(step)
        add(f"risk-{i}", "TC1_PLAN_ARITHMETIC", f"Учебный {'LONG' if i%2 else 'SHORT'}. Equity={equity} USDT; доля риска={pct}; "
            f"расстояние entry-stop={distance}; шаг количества={step}. Пункт стоит 1 USDT. Издержки исключены только для этой арифметики. "
            "Вычисли risk_budget и quantity; количество округли вниз к шагу.", {"risk_budget": float(budget), "quantity": float(qty)})
    for i in range(4):
        low = Decimal(307 + i*13); high = low + Decimal("4.5"); tick = Decimal("0.25")
        long = i%2 == 0
        add(f"wm1-{i}", "TC2_WM1_LEVELS", f"В TC2 подтверждены закрытый {'бычий' if long else 'медвежий'} WM1 и ангуляция. "
            f"Low={low}, high={high}, tick={tick}. Назови entry за экстремумом по направлению и stop за противоположным. "
            "Это план, заявка не исполнялась.", {"entry": float(high+tick if long else low-tick), "stop": float(low-tick if long else high+tick)})
    for i, (end, begin) in enumerate(((206,205),(318,319),(402,402),(507,510))):
        add(f"hump-{i}", "COURSE_HUMP_NONOVERLAP", f"Course PDF72. Первая крайняя фрактальная формация включает бары до {end} включительно, "
            f"вторая — начиная с {begin} включительно. Оба противоположных фрактала подтверждены; прочие условия здесь считаются выполненными. "
            "Проверь только неперекрытие формаций. Поле hump_valid.", {"hump_valid": begin > end})
    for i, (hist, signal) in enumerate((("0.6","0.8"),("-0.7","-0.9"),("-0.6","-0.4"),("0.8","0.5"))):
        add(f"momentum-{i}", "COURSE_SIGNAL_LINE", f"Профиль Course 5/34/5, PDF45: значение гистограммы={hist}, "
            f"собственная SMA(5) гистограммы={signal}. Назови только фильтр momentum как LONG или SHORT; структура входа отсутствует.",
            {"momentum": "LONG" if Decimal(hist) > Decimal(signal) else "SHORT"})
    for i, (a,b) in enumerate((("1.7","0.2"),("2.4","-0.2"),("0.9","-0.1"),("1.1","0.1"))):
        add(f"w4-{i}", "COURSE_W4_MINIMUM", f"Course PDF44. Масштаб последовательности — 120 баров. После предполагаемой W3 "
            f"закрытые значения 5/34 изменились с {a} на {b}. Завершение младшей c/e не установлено. "
            "Укажи minimum_w4_met и w4_finished.", {"minimum_w4_met": Decimal(b)<0, "w4_finished": False})
    for n in (1,2):
        add(f"fractal-{n}", "TC2_FRACTAL_CAUSALITY", f"Строгий пятибарный фрактал: центральный HIGH больше всех соседних HIGH, "
            f"две свечи слева закрыты, справа закрыто {n} подходящих свечей. Правых неподтверждённых данных не использовать. "
            "Можно ли сейчас назвать фрактал confirmed?", {"confirmed": n == 2})
    for i,(ac,green) in enumerate((("-0.3",2),("0.3",2))):
        add(f"ac-{i}", "NTD_AC_SIGN", f"NTD AC, покупка. Последние {green} столбца зелёные, до них красный. "
            f"Текущее значение именно AC={ac}; AO положительный. Требуются required_green и signal_ready. "
            "Не заменяй знак AC знаком AO.", {"required_green": 3 if Decimal(ac)<0 else 2, "signal_ready": Decimal(ac)>0})
    if len(questions) != 24 or len({q["case_id"] for q in questions}) != 24:
        raise ValueError("FIXED_24_CASE_TRANSFER_EXAM_REQUIRED")
    return questions, keys


def _answer(text):
    try:
        obj = json.loads(text.strip())
    except (TypeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _matches(actual, expected):
    if type(expected) is bool:
        return type(actual) is bool and actual == expected
    if isinstance(expected, (int,float)):
        return type(actual) in (int,float) and math.isfinite(actual) and abs(actual-expected) <= 1e-7*max(1,abs(expected))
    return type(actual) is type(expected) and actual == expected


def grade(questions: list[dict], keys: list[dict], precommit: dict) -> dict:
    digest = sha256(json.dumps(questions,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    if precommit.get("questions_sha256") != digest or set(precommit.get("stages",{})) != {"base","first_private","corrected"}:
        raise ValueError("SEALED_SAME_PROMPTS_THREE_MODELS_REQUIRED")
    expected = {k["case_id"]: k for k in keys}
    result = {}
    for stage, rows in precommit["stages"].items():
        if len(rows)!=24 or {r["case_id"] for r in rows} != set(expected) or len({r["case_id"] for r in rows}) != 24:
            raise ValueError("EXACT_TRANSFER_COHORT_NO_REPLAY_OR_MISSING_ANSWERS")
        scores = []
        for row in rows:
            key = expected[row["case_id"]]
            parsed = _answer(row["response"])
            correct = (parsed is not None and set(parsed)==set(key["expected"])
                and all(_matches(parsed[k],v) for k,v in key["expected"].items()))
            scores.append({"case_id": row["case_id"], "domain": key["domain"], "correct": correct,
                           "valid_json_object": parsed is not None})
        normalized = [" ".join(r["response"].split()) for r in rows]
        counts = Counter(normalized)
        result[stage] = {"correct": sum(x["correct"] for x in scores), "total": 24,
            "valid_json": sum(x["valid_json_object"] for x in scores),
            "distinct_responses": len(counts), "largest_identical_response_count": max(counts.values()),
            "case_results": scores}
    return {"mode": "GENERATED_JSON_24_CASE_SOURCE_RULE_ARITHMETIC_TRANSFER_RESEARCH",
        "stages": result, "generated_text_not_candidate_logit_only": True,
        "independent_expert_market_wave_exam": False, "profitability_measured": False,
        "trading_qualification": False,"execution_allowed":False}


def predict(questions_path: Path, adapter_dirs: dict[str,Path], new_output: Path):
    """GPU only; this entry receives NO expected answer-key argument."""
    from contextlib import nullcontext
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    from .deep_book_research import MODEL, REVISION
    questions = json.loads(questions_path.read_bytes())
    if (len(questions)!=24 or any(set(q)!={"case_id","domain","prompt"} for q in questions)
            or new_output.exists() or torch.cuda.device_count()!=1):
        raise ValueError("ONLY_SEALED_QUESTION_PROMPTS_ON_ONE_GPU_REQUIRED")
    tokenizer = AutoTokenizer.from_pretrained(MODEL,revision=REVISION,trust_remote_code=False)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,trust_remote_code=False,use_safetensors=True,
        device_map={"":0},dtype=dtype,quantization_config=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype))
    model = PeftModel.from_pretrained(base,adapter_dirs["first_private"],adapter_name="first_private",is_trainable=False,local_files_only=True)
    model.load_adapter(adapter_dirs["corrected"],adapter_name="corrected",is_trainable=False,local_files_only=True)
    model.eval(); model.config.use_cache=True
    stages = {}
    with torch.inference_mode():
        for stage in ("base","first_private","corrected"):
            stages[stage]=[]
            if stage!="base":model.set_adapter(stage)
            for row in questions:
                messages=[{"role":"system","content":"Ты решаешь проверочное задание. Следуй требуемому JSON-формату, не придумывай отсутствующие факты."},
                    {"role":"user","content":row["prompt"]}]
                prefix=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
                ids=torch.tensor([prefix],device=next(model.parameters()).device)
                with (model.disable_adapter() if stage=="base" else nullcontext()):
                    generated=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),do_sample=False,
                        max_new_tokens=128,pad_token_id=tokenizer.eos_token_id)
                response=tokenizer.decode(generated[0,len(prefix):],skip_special_tokens=True).strip()
                stages[stage].append({"case_id":row["case_id"],"response":response})
            # Persist each whole stage in case an external quota interrupts later.
            new_output.write_text(json.dumps({"questions_sha256":sha256(json.dumps(questions,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
                "stages":stages,"answer_keys_loaded_on_gpu":False,"execution_allowed":False},ensure_ascii=False,indent=2))
    return {"cases":24,"model_conditions":3,"execution_allowed":False}


def predict_loaded(model,tokenizer,questions_path:Path,initial_adapter:Path,new_output:Path):
    """Reuse the trained CUDA base: base / own416 checkpoint / final adapter.

    This receives prompts only, after final weights were saved. No extra GPU
    instance or test-answer keys are needed for the comparison.
    """
    from contextlib import nullcontext
    import torch
    questions=json.loads(questions_path.read_bytes())
    if (not isinstance(questions,list) or len(questions)!=24
        or len({q.get("case_id") for q in questions})!=24
        or any(set(q)!={"case_id","domain","prompt"} or not isinstance(q["prompt"],str) for q in questions)
        or new_output.exists()):raise ValueError("SEALED_PROMPTS_ONLY_TRANSFER_COHORT_REQUIRED")
    model.load_adapter(initial_adapter,adapter_name="own_checkpoint_reference",is_trainable=False,local_files_only=True)
    model.gradient_checkpointing_disable();model.eval();model.config.use_cache=True
    stages={}
    with torch.inference_mode():
        for stage,adapter in (("base",None),("first_private","own_checkpoint_reference"),("corrected","default")):
            if adapter:model.set_adapter(adapter)
            stages[stage]=[]
            for row in questions:
                messages=[{"role":"system","content":"Ты решаешь проверочное задание. Следуй требуемому JSON-формату, не придумывай отсутствующие факты."},
                    {"role":"user","content":row["prompt"]}]
                prefix=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
                ids=torch.tensor([prefix],device=next(model.parameters()).device)
                with (model.disable_adapter() if adapter is None else nullcontext()):
                    tokens=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),do_sample=False,
                        max_new_tokens=128,pad_token_id=tokenizer.eos_token_id)
                response=tokenizer.decode(tokens[0,len(prefix):],skip_special_tokens=True).strip()
                stages[stage].append({"case_id":row["case_id"],"response":response})
            new_output.write_text(json.dumps({"questions_sha256":sha256(json.dumps(questions,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
                "stages":stages,"answer_keys_loaded_on_gpu":False,"first_private_is_interrupted_source_checkpoint":True,
                "execution_allowed":False},ensure_ascii=False,indent=2))
    model.set_adapter("default")
    return {"cases":24,"model_conditions":3,"in_memory_gpu_reuse":True,"expected_keys_in_model_prompt":False,
        "first_private_is_interrupted_source_checkpoint":True,"execution_allowed":False}
