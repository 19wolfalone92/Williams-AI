"""Independent post-fit actual GGUF LoRA exam (read-only).

Old 16-case comparison is DIAGNOSTIC and already inspected in past research.
The 28 new test MCQs are rule-ID-disjoint from SFT train and validation,
but are synthetic developer-derived paraphrases, NOT expert wave-chart gold.
All test answers remain on the examiner side; model only sees task prompts.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import gc
import json
import os
from pathlib import Path

from .knowledge_curriculum_v5 import build,digest
from .four_book_blind_exam import grade_four_book_model
from .local_model import LocalWilliamsModel
from .real_gguf_acceptance import actual_gguf_sha256

BASE_SHA="f6f851777709861056efcdad3af01da38b31223a3ba26e61a4f8bf3a2195813a"
OLD_PROTOCOL_SHA="75e96a62894e29a1cc3e5e71623d7d1a78a93e0ce620555c7da0bb29b8f5ce5b"


class ExamLlama:
    answer_exam_case=LocalWilliamsModel.answer_exam_case
    _learning_completion=LocalWilliamsModel._learning_completion
    def __init__(self,*,base:Path,lora:Path|None=None,thinking=True):
        from llama_cpp import Llama
        kwargs={"model_path":str(base.resolve()),"n_ctx":4096,
                "n_gpu_layers":0,"verbose":False}
        if lora is not None:
            if not lora.is_file() or lora.read_bytes()[:4]!=b"GGUF":
                raise ValueError("REAL_TRAINED_LORA_GGUF_REQUIRED")
            kwargs.update(lora_path=str(lora.resolve()),lora_scale=1.0)
        self._model=Llama(**kwargs)
        self._learning_reasoning=thinking
        self._reasoning_budget=512
        self.latest_learning_call=None


def _heldout_exam(model,rows,*,mode):
    test=[x for x in rows if x["split"]=="test" and x["task_type"]=="MCQ"]
    if len(test)!=28:
        raise ValueError("EXPECTED_28_FROZEN_RULE_DISJOINT_MCQS")
    answers=[]
    for row in test:
        if row.get("eligible_for_sft") is not False:
            raise ValueError("MODEL_TEST_ANSWER_LEAKAGE_OR_SPLIT_COLLISION")
        p=row["prompt"]
        safe={"id":row["case_id"],"edition":row["edition"],
              "question":p["question"],"choices":p["choices"]}
        expected=json.loads(row["answer"])["choice"]
        # No correct letter or gold rule_id passed into inference.
        pred=model.answer_exam_case(safe)
        choice=pred.get("choice") if isinstance(pred,dict) else ""
        choice=choice if choice in ("A","B","C") else ""
        trace=getattr(model,"latest_learning_call",None)
        trace=trace if isinstance(trace,dict) else {}
        answers.append({"id":row["case_id"],"edition":row["edition"],
                        "model_decoding_mode":trace.get("mode"),
                        "initial_finish_reason":trace.get("finish_reason"),
                        "json_fallback_used":trace.get("fallback_used",False),
                        "json_fallback_reason":trace.get("fallback_reason"),
                        "json_fallback_finish_reason":trace.get("fallback_finish_reason"),
                        "chosen":choice,"correct":choice==expected,
                        "abstained":not bool(choice),
                        "answer_key_hidden_from_model":True})
    correct=sum(x["correct"] for x in answers)
    by_ed={ed:{"correct":sum(x["correct"] for x in answers if x["edition"]==ed),
               "total":sum(x["edition"]==ed for x in answers),
               "abstained":sum(x["abstained"] for x in answers if x["edition"]==ed)}
           for ed in ("TC1","NTD","TC2","COURSE")}
    return {"mode":mode,"total":len(answers),"correct":correct,
            "by_edition":by_ed,
            "case_pass_fail":answers,
            "source_card_rule_disjoint_test_only":True,
            "independent_expert_wave_market_labels":False}


def evaluate(base:Path,gguf_lora:Path,receipt:dict,*,out:Path):
    if (not isinstance(receipt,dict) or
        receipt.get("schema")!="WILLIAMS_V5_GENUINE_QWEN3_4B_SOURCE_ONLY_LORA" or
        receipt.get("weights_changed") is not True or
        receipt.get("test_cases_provided_to_trainer")!=0 or
        len(receipt.get("adapter_sha256",""))!=64 or
        receipt.get("trading_qualification") is not False):
        raise ValueError("AUTHENTICATED_TRAINED_WILLIAMS_V5_RECEIPT_REQUIRED")
    actual=actual_gguf_sha256(base,BASE_SHA)
    rows,manifest=build()
    if (receipt["rule_registry_sha256"]!=manifest["source_registry_sha256"] or
        receipt["dataset_split_gold_sha256"]!=manifest["split_gold_sha256"]):
        raise ValueError("EXAM_DATA_NOT_IDENTICAL_TO_FROZEN_TRAINING_MANIFEST")
    observed={}
    for name,adapter in (("candidate_v5",gguf_lora),("public_base_control",None)):
        model=ExamLlama(base=base,lora=adapter,thinking=True)
        a=grade_four_book_model(model,per_edition=4,seed=20261024,
                                autonomous_source_retrieval=False)
        b=grade_four_book_model(model,per_edition=4,seed=20261024,
                                autonomous_source_retrieval=True)
        if a["protocol_sha256"]!=OLD_PROTOCOL_SHA or b["protocol_sha256"]!=OLD_PROTOCOL_SHA:
            raise ValueError("UNEXPECTED_CHANGED_OLD_FOUR_BOOK_EXAM")
        frozen=_heldout_exam(model,rows,mode=name)
        observed[name]={
            "prior_16_case_closed_book":{"correct":a["correct"],"total":a["total"],
                "by_edition":a["by_edition"],
                "abstentions":sum(x["abstained"] for x in a["results"])},
            "prior_16_case_with_source":{"correct":b["correct"],"total":b["total"],
                "by_edition":b["by_edition"],
                "abstentions":sum(x["abstained"] for x in b["results"])},
            "new_rule_disjoint_28_case_exam":frozen,
            "weights_trained_in_this_run":adapter is not None,
        }
        del model
        gc.collect()
    score=observed["candidate_v5"]["new_rule_disjoint_28_case_exam"]
    baseline=observed["public_base_control"]["new_rule_disjoint_28_case_exam"]
    same_counts=score["total"]==baseline["total"]==28
    if not same_counts:
        raise ValueError("FROZEN_HOLDOUT_CASE_COUNT_MISMATCH")
    # Require improvement on the new questions, but never claim that this
    # synthetic benchmark can qualify the model for live futures.
    report={
        "schema":"WILLIAMS_V5_ACTUAL_TRAINED_MODEL_BLIND_EXAM",
        "model_commit":receipt["model_commit"],
        "adapter_sha256":receipt["adapter_sha256"],
        "gguf_lora_sha256":sha256(gguf_lora.read_bytes()).hexdigest(),
        "fixed_protocol_sha256":OLD_PROTOCOL_SHA,
        "source_curriculum_manifest_sha256":digest(manifest),
        "candidate_vs_base":observed,
        "rule_disjoint_delta_correct":score["correct"]-baseline["correct"],
        "source_RAG_16_case_delta":(
            observed["candidate_v5"]["prior_16_case_with_source"]["correct"]-
            observed["candidate_v5"]["prior_16_case_closed_book"]["correct"]),
        "old_model_serial50_16case_reference":{"without_sources":9,"with_sources":8,
            "exam_run_id":43085091992},
        "genuine_gpu_weights_updated":True,
        "old_16_case_exam_previously_observed_not_new_untouched_test":True,
        "new_28_case_is_developer_synthetic_not_expert_chart_gold":True,
        "expert_wave_chart_mastery_measured":False,
        "user_original_books_exported":False,
        "user_model_automatically_promoted":False,
        "execution_allowed":False,"trading_qualification":False,
    }
    if out.exists():
        raise FileExistsError("DO_NOT_OVERWRITE_INDEPENDENT_EXAM_RECEIPT")
    out.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(out,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,"w",encoding="utf-8") as stream:
        json.dump(report,stream,ensure_ascii=False,sort_keys=True,indent=2)
    print("ACTUAL_NEW_WILLIAMS_V5_EXAM_RESULTS",
          json.dumps({"candidate":score["correct"],"base":baseline["correct"],
                      "delta":report["rule_disjoint_delta_correct"],
                      "candidate_sources":observed["candidate_v5"]["prior_16_case_with_source"]["correct"],
                      "candidate_without_sources":observed["candidate_v5"]["prior_16_case_closed_book"]["correct"],
                      "trading_qualification":False},sort_keys=True),flush=True)
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base-gguf",type=Path,required=True)
    p.add_argument("--trained-lora-gguf",type=Path,required=True)
    p.add_argument("--training-receipt",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    receipt=json.loads(args.training_receipt.read_text(encoding="utf-8"))
    evaluate(args.base_gguf,args.trained_lora_gguf,receipt,out=args.output)


if __name__=="__main__":
    main()
