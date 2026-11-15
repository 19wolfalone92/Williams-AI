"""Audit frozen historical 100 Qwen Base vs v5 LoRA *without* gold or further inference.

Read verified CPU GGUF and GPU HF/NF4 artifacts, never derive trading
correctness from decoder validity. Generates JSON and markdown evidence
and explicitly unapproved review/training-needs queue, not SFT labels.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

BOOKS=("TC1","NTD","TC2","COURSE")
ROLES=("matched_base","candidate_v5")
HEX=set("0123456789abcdef")
SCHEMA="WILLIAMS_V6_POSTFLIGHT_CPU_GPU_100_DEVELOPMENT_DIAGNOSTIC_NO_EXPERT_GOLD_V1"

def require(ok, msg):
    if not ok: raise ValueError("HIST100_POSTFLIGHT_"+msg)

def ishash(x): return isinstance(x,str) and len(x)==64 and set(x)<=HEX

def check(source, data):
    prefix="real_" if source=="cpu" else "real_"
    require(data.get("complete") is True and data.get("accuracy") is None and
            data.get("gold_labels") is None, "NOT_COMPLETE_OR_HAS_GOLD")
    if source=="cpu":
        require(data.get("real_chart_count")==100 and
                data.get("real_base_inference_rows")==100 and
                data.get("real_v5_inference_rows")==100 and
                data.get("matched_inputs") is True and
                data.get("trading_qualification") is False and
                data.get("model_train_overlap_independently_audited") is False,
                "CPU_NOT_VERIFIED")
        require(ishash(data.get("base_model_sha256")) and
                ishash(data.get("trained_lora_sha256")) and
                ishash(data.get("protocol_sha256")), "CPU_WEIGHT_OR_PROTOCOL_NOT_PINNED")
        select=data["market_selection_plan_sha256"]
    else:
        require(data.get("historical_charts")==100 and
                data.get("real_base_answers")==100 and
                data.get("real_trained_v5_answers")==100 and
                data.get("verified_same_prompts_per_paired_model") is True and
                data.get("no_live_trading") is True and
                data.get("not_homogeneous_input_length_policy") is True and
                data.get("fallback_case_ids")==["HIST100-25-3","HIST100-25-4"] and
                data.get("gpu_primary_protocol_98_case_max_tokens")==3200 and
                data.get("gpu_last_two_fallback_max_tokens")==4096,
                "GPU_INCOMPLETE_OR_NOT_PINNED")
        require(ishash(data.get("true_trained_adapter_sha256")), "GPU_ADAPTER_NOT_PINNED")
        select=data["source_selection_sha256"]
    require(ishash(select),"SOURCE_SELECTION_HASH")
    out={}
    raw=data.get("observed")
    require(isinstance(raw,dict) and set(raw)==set(ROLES),"BAD_MODEL_ROLES")
    for role in ROLES:
        rows=raw[role]
        require(isinstance(rows,list) and len(rows)==100,"NOT_100_MODEL_ROWS")
        ids={r.get("id") for r in rows}
        require(len(ids)==100 and
                ids=={f"HIST100-{n:02d}-{d}" for n in range(1,26) for d in range(1,5)},
                "MISSING_OR_DUPLICATED_CASES")
        require(all(r.get("edition") in BOOKS and
                    r.get("finish_reason") in ("stop","error","length") and
                    ishash(r.get("chart_sha256")) and ishash(r.get("prompt_sha256"))
                    for r in rows),"INVALID_INFERENCE_RECEIPT_ROW")
        require(all(sum(r["edition"]==book for r in rows)==25 for book in BOOKS),
                "BOOK_SELECTION_IS_NOT_25_X_4")
        require(all(
            r.get("decision")=="ABSTAIN" if r["finish_reason"]!="stop"
            else isinstance(r.get("model_reason"),str) and
                 1<=len(r.get("model_reason"))<=240
            for r in rows),"FAILED_REASON_MUST_NOT_BE_TREATED_AS_VALID")
        out[role]={r["id"]:r for r in rows}
    for cid in out["matched_base"]:
        a,b=out["matched_base"][cid],out["candidate_v5"][cid]
        require(all(a[k]==b[k] for k in
             ("id","edition","family","symbol","cutoff_utc","chart_sha256",
              "prompt_sha256","retrieved_rule_ids")),"MODEL_INPUT_MISMATCH")
        if source=="gpu":
            require(a["original_cpu_full_prompt_sha256"]==b["original_cpu_full_prompt_sha256"] and
                    ishash(a["original_cpu_full_prompt_sha256"]),"GPU_ORIGINAL_FULL_PROMPT_MISSING")
    return select,out

def status(row):
    if row["finish_reason"]!="stop":
        return "TECHNICAL_"+row["finish_reason"].upper()
    if row["decision"]=="ABSTAIN":
        return "MODEL_ABSTAIN_NOT_EXPERT_REVIEWED"
    if row["decision"]=="INSUFFICIENT_DATA":
        return "MODEL_CLAIMS_INSUFFICIENT_DATA"
    return "MODEL_DECISION_NOT_EXPERT_REVIEWED"

def summarize(rows):
    ordered=list(rows.values())
    reasons=[x.get("model_reason") for x in ordered if x.get("model_reason")]
    return {
        "model_answers":len(ordered),
        "finish":dict(Counter(x["finish_reason"] for x in ordered)),
        "decision":dict(Counter(x["decision"] for x in ordered)),
        "technical_abstain_count":sum(x["decision"]=="ABSTAIN" and x["finish_reason"]!="stop" for x in ordered),
        "model_abstain_count_not_expert_verified":sum(x["decision"]=="ABSTAIN" and x["finish_reason"]=="stop" for x in ordered),
        "exactly_240_char_reasons":sum(len(s)==240 for s in reasons),
        "exact_insufficient_reason_non_insufficient_label":sum(
            x.get("model_reason","").strip().upper()=="INSUFFICIENT_DATA"
            and x["decision"] not in ("INSUFFICIENT_DATA","ABSTAIN")
            for x in ordered),
        "unique_nonempty_reasons":len(set(reasons)),
        "by_book":{book:{
            "finish":dict(Counter(x["finish_reason"] for x in ordered if x["edition"]==book)),
            "decisions":dict(Counter(x["decision"] for x in ordered if x["edition"]==book)),
            "technical_abstain":sum(x["decision"]=="ABSTAIN" and x["finish_reason"]!="stop" and x["edition"]==book for x in ordered)
        } for book in BOOKS}
    }

def analyze(cpu,gpu):
    csha,c=check("cpu",cpu);gsha,g=check("gpu",gpu)
    require(csha==gsha,"CPU_GPU_SELECTION_PLAN_DIFFERENT")
    # CPU's exact longer full prompt SHA is also recorded in GPU output.
    for role in ROLES:
        for cid,row in c[role].items():
            other=g[role][cid]
            require(row["chart_sha256"]==other["chart_sha256"] and
                    row["edition"]==other["edition"] and
                    row["prompt_sha256"]==other["original_cpu_full_prompt_sha256"],
                    "CROSS_BACKEND_ORIGINAL_INPUT_MISMATCH")
    summaries={src:{r:summarize(rows[r]) for r in ROLES}
               for src,rows in (("cpu",c),("gpu",g))}
    deltas={src:sum(rows["matched_base"][k]["decision"]!=rows["candidate_v5"][k]["decision"]
                    for k in rows["matched_base"])
            for src,rows in (("cpu",c),("gpu",g))}
    same={r:sum(c[r][k]["decision"]==g[r][k]["decision"] for k in c[r]) for r in ROLES}
    queue=[]
    for book in BOOKS:
        group=[cid for cid in c["matched_base"] if c["matched_base"][cid]["edition"]==book]
        queue.append({
            "edition":book,"case_count":25,
            "cpu_errors_base":sum(c["matched_base"][i]["finish_reason"]!="stop" for i in group),
            "cpu_errors_lora":sum(c["candidate_v5"][i]["finish_reason"]!="stop" for i in group),
            "gpu_errors_base":sum(g["matched_base"][i]["finish_reason"]!="stop" for i in group),
            "gpu_errors_lora":sum(g["candidate_v5"][i]["finish_reason"]!="stop" for i in group),
            "cpu_disagreements":sum(c["matched_base"][i]["decision"]!=c["candidate_v5"][i]["decision"] for i in group),
            "gpu_disagreements":sum(g["matched_base"][i]["decision"]!=g["candidate_v5"][i]["decision"] for i in group),
            "needs_original_page_and_diagram_review":True,
            "independent_expert_labels":0,"training_eligible":False,
        })
    return {
        "schema":SCHEMA,"complete_200_real_model_rows_per_backend":True,
        "research_only":True,"same_original_100_chart_selection_sha256":csha,
        "original_cpu_prompt_hashes_cross_checked_gpu":True,
        "backends_are_not_identical":"CPU_GGUF_96H1_vs_GPU_HF_NF4_20_48H1_PLUS_2_FALLBACK",
        "gpu_fallback_not_homogeneous":True,
        "model_training_modified":False,"new_model_inference_executed":False,
        "expert_gold_labels":None,"book_accuracy":None,"financial_profitability":None,
        "prospective_exam":False,"ready_for_live_trading":False,
        "source_books_count":4,"summaries":summaries,
        "model_disagreement_count_NOT_ACCURACY":deltas,
        "same_prediction_across_backends_NOT_CORRECTNESS":same,
        "expert_review_priority_queue_NOT_TRAINING_DATA":queue,
    }

def report(x):
    lines=["# Williams AI v6 — технический аудит после 100 исторических H1 (НЕ экзамен знаний)",
           "","Два полностью завершённых реальных прогона: CPU GGUF и GPU HF/NF4, в каждом 100 Base + 100 genuine v5 LoRA.",
           "**Это не точность торговли:** нет независимых экспертных gold-меток; различия backend и длины контекста.",
           "","## Генерация и отказы","",
           "| Режим/модель | Stop | Error | Length | Технические ABSTAIN | Модельные ABSTAIN |",
           "|---|---:|---:|---:|---:|---:|"]
    for src in ("cpu","gpu"):
        for role in ROLES:
            s=x["summaries"][src][role]
            lines.append(f"| {src.upper()} {role} | {s['finish'].get('stop',0)} | {s['finish'].get('error',0)} | {s['finish'].get('length',0)} | {s['technical_abstain_count']} | {s['model_abstain_count_not_expert_verified']} |")
    lines+=["","## Критические проблемы / проверка книг","",
            "| Книга | CPU LoRA errors/length | GPU LoRA errors/length | CPU Base-v5 разные | GPU Base-v5 разные |",
            "|---|---:|---:|---:|---:|"]
    for row in x["expert_review_priority_queue_NOT_TRAINING_DATA"]:
        lines.append(f"| {row['edition']} | {row['cpu_errors_lora']} | {row['gpu_errors_lora']} | {row['cpu_disagreements']} | {row['gpu_disagreements']} |")
    lines+=["",f"CPU Base/LoRA: {x['model_disagreement_count_NOT_ACCURACY']['cpu']} разных решений из 100.",
           f"GPU Base/LoRA: {x['model_disagreement_count_NOT_ACCURACY']['gpu']} разных решений из 100.",
           f"Между GPU и CPU совпали категории: Base {x['same_prediction_across_backends_NOT_CORRECTNESS']['matched_base']}/100, LoRA {x['same_prediction_across_backends_NOT_CORRECTNESS']['candidate_v5']}/100. Это НЕ экспертная оценка.",
           "","### Дальше, строго по воротам качества",
           "1. Исправить техническую обработку ошибок JSON/length, не выдавать ошибку за осмысленный ABSTAIN. Заморозить новые протоколы, старые ответы не переписывать.",
           "2. По четырём источникам проверить оригинальные страницы и рисунки, независимые отзывы двух экспертов и конфликты переводов. Кандидатная очередь ниже — **не gold и не тренировочный датасет**.",
           "3. Затем заморозить отдельный train/dev/test, без утечки исторических 100 и будущих H1, и только потом учить следующую LoRA.",
           "4. Перспективный H1-набор не использовать для подгонки ответов. Не подключать AI к исполнению ордеров до независимой проверки.",""]
    return "\n".join(lines)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cpu",type=Path,required=True)
    p.add_argument("--gpu",type=Path,required=True)
    p.add_argument("--out-json",type=Path,required=True)
    p.add_argument("--out-markdown",type=Path,required=True)
    args=p.parse_args()
    require(not args.out_json.exists() and not args.out_markdown.exists(),"OUTPUT_ALREADY_EXISTS")
    result=analyze(json.loads(args.cpu.read_bytes()),json.loads(args.gpu.read_bytes()))
    args.out_json.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+"\n")
    args.out_markdown.write_text(report(result))
    print(json.dumps({k:v for k,v in result.items() if k in (
        "model_disagreement_count_NOT_ACCURACY",
        "same_prediction_across_backends_NOT_CORRECTNESS",
        "expert_review_priority_queue_NOT_TRAINING_DATA",
    )},ensure_ascii=False,sort_keys=True,indent=2))

if __name__=="__main__": main()
