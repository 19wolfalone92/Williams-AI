"""Actual Qwen3-4B base-vs-QLoRA BLIND prediction, then separate grading.

No future price labels are even read in 'predict'. Scores are normalized
candidate next-token log probabilities (not empirically calibrated odds).
This does not evaluate expert wave annotation or prove trading profit.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
from hashlib import sha256
import json
import math
import os
from pathlib import Path

from .chart_supervision import chart_prompt_observations, checked_chart_context
from .independent_chart_holdout import grade_blind_predictions
from .lora_integrity import local_hf_identity, regular_sha256, strict_json
from .private_lora import _private_file, checked_local_qwen_config

CLASSES = ("UP", "DOWN", "RANGE")
INFERENCE_QUESTION_FIELDS = frozenset({
    "case_id", "market_context", "task", "status",
    "future_data_in_prompt", "trade_action", "execution_allowed",
})
SOURCE_MODE = "PRIVATE_FOUR_BOOK_TEXT_ONLY_QWEN3_4B_NF4_QLORA"


def _read_jsonl(path: Path, *, max_bytes: int = 12_000_000) -> list[dict]:
    path = _private_file(path)
    if path.stat().st_size > max_bytes:
        raise ValueError("BOUNDED_PRIVATE_RESEARCH_JSONL_REQUIRED")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = strict_json(line)
            if not isinstance(item, dict):
                raise ValueError("EXPECTED_JSONL_OBJECT")
            rows.append(item)
    return rows


def verified_blind_questions(path: Path) -> tuple[list[dict], str]:
    rows = _read_jsonl(path)
    if len(rows) != 30 or len({r.get("case_id") for r in rows}) != 30:
        raise ValueError("THIRTY_UNIQUE_ASOF_QUESTIONS_REQUIRED")
    for row in rows:
        if (set(row) != INFERENCE_QUESTION_FIELDS
                or row["future_data_in_prompt"] is not False
                or row["trade_action"] is not None
                or row["execution_allowed"] is not False
                or row["task"] != ("Predict UP/DOWN/RANGE over next 24 H1 CLOSED bars, "
                                   "with probabilities summing to 1. No order; may abstain.")
                or row["status"] !=
                   "UNLABELED_CANDIDATE_NEEDS_INDEPENDENT_WAVE_WM1_WM2_WM3_EXPERT_REVIEW"):
            raise ValueError("FUTURE_BLIND_MARKET_QUESTION_CONTRACT_REQUIRED")
        context = row["market_context"]
        checked = checked_chart_context(context)
        if (checked["end"].isoformat().replace("+00:00","Z")
                != context["cutoff"].replace(".000Z","Z").replace(".999Z","Z")
                and not context["cutoff"].startswith(checked["end"].strftime("%Y-%m-%dT%H:%M:%S"))):
            raise ValueError("CUTOFF_CHANGED_AFTER_MARKET_SEAL")
        if (context["timeframe"] != "H1" or context["venue"] != "FUTURES"
                or len(context["bars"]) != 240
                or row["case_id"] != f"{context['symbol']}-H1-asof-{int(checked['end'].timestamp()*1000)}"):
            raise ValueError("UNEXPECTED_REAL_H1_CASE_ID_OR_HISTORY")
    digest = sha256(path.read_bytes()).hexdigest()
    return rows, digest


def source_adapter_identity(base_dir: Path, adapter_dir: Path) -> dict:
    """Admit only a four-book source adapter that passed TRAINING validation.

    This does not imply a market qualification.
    """
    base = _private_file(base_dir, must_exist=False)
    adapter = _private_file(adapter_dir, must_exist=False)
    if not base.is_dir() or not adapter.is_dir():
        raise ValueError("LOCAL_BASE_AND_SOURCE_ADAPTER_REQUIRED")
    report = strict_json((adapter / "source_research_report.json").read_bytes())
    if (report.get("mode") != SOURCE_MODE
            or report.get("weight_training_completed") is not True
            or report.get("source_only") is not True
            or report.get("validation_gate",{}).get("accepted") is not True
            or not isinstance(report.get("adapter_sha256"),str)
            or report.get("blind_market_holdout_passed") is not False
            or report.get("execution_allowed") is not False):
        raise ValueError("QLORA_SOURCE_ADAPTER_NOT_ACCEPTED_OR_WRONG_PROVENANCE")
    if checked_local_qwen_config(base)["model_type"] != "qwen3":
        raise ValueError("QWEN3_4B_BASE_REQUIRED")
    import json as _json
    config = _json.loads((base / "config.json").read_text(encoding="utf-8"))
    if config.get("hidden_size") != 2560:
        raise ValueError("EXPECTED_QWEN3_4B_ONLY")
    if (local_hf_identity(base) != report.get("base_identity")
            or regular_sha256(adapter / "adapter_model.safetensors")
               != report["adapter_sha256"]):
        raise ValueError("TRAINED_BASE_OR_ADAPTER_WEIGHTS_CHANGED")
    return {"base_sha256": report["base_identity"]["sha256"],
            "adapter_sha256": report["adapter_sha256"],
            "fit_validation_passed": True, "source_only_not_market_training": True}


def market_prompt(row: dict) -> str:
    """Give Qwen ONLY as-of numeric observations, no future and no labels."""
    raw = chart_prompt_observations(row["market_context"])
    compact = {
        "symbol": raw["symbol"], "timeframe":raw["timeframe"],
        "as_of":raw["as_of"], "market_sha256":raw["market_sha256"],
        "columns":raw["tail_columns"], "closed_tail":raw["tail"][-12:],
        "confirmed_pivots_columns":raw["strict_fractal_columns"],
        "confirmed_pivots":raw["last_confirmed_strict_fractals"][-8:],
        "visible_ranges":raw["prior_ranges"],
    }
    return ("Только исследование рынка без ордеров. Видны ТОЛЬКО закрытые "
            "свечи и причинные индикаторы. Волновая интерпретация неопределённа; "
            "не выдумывай будущие бары, гарантированный разворот или сделку. "
            "Выбери один класс направления движения через следующие 24 бара H1: "
            "UP, DOWN или RANGE; RANGE если движение менее одного ATR H1 на "
            "текущем масштабе. Ответ — ровно одно слово класса.\n"
            + json.dumps(compact,ensure_ascii=False,sort_keys=True,separators=(",",":")))


def candidate_first_token_ids(tokenizer, messages: list[dict]) -> tuple[list[int],list[int]]:
    """Require three DISTINCT first answer tokens, avoiding tokenization leaks.

    Compares next-token probabilities conditional on the exact SAME prefix.
    """
    prefix = tokenizer.apply_chat_template(messages, tokenize=True,
        add_generation_prompt=True,enable_thinking=False)
    if not prefix or not isinstance(prefix,list):
        raise ValueError("QWEN_NONEMPTY_CHAT_PREFIX_REQUIRED")
    targets = []
    for label in CLASSES:
        complete = tokenizer.apply_chat_template(
            messages + [{"role":"assistant","content":label}],
            tokenize=True, add_generation_prompt=False,enable_thinking=False)
        if (not isinstance(complete,list)
                or complete[:len(prefix)] != prefix
                or len(complete) <= len(prefix)):
            raise ValueError("QWEN_CLASS_TOKENIZATION_PREFIX_MISMATCH")
        targets.append(complete[len(prefix)])
    if len(set(targets)) != len(CLASSES):
        raise ValueError("QWEN_CLASS_TOKENS_NOT_UNIQUELY_IDENTIFIABLE")
    return prefix, targets


def predict_pair(*, questions: Path, hf_base: Path, source_adapter: Path,
                 new_output: Path) -> dict:
    """CUDA-only, real pretrained Qwen3-4B with adapter enable/disable."""
    if new_output.exists() or new_output.is_symlink():
        raise ValueError("PREDICTION_PRECOMMIT_MUST_BE_NEW")
    cases, question_sha = verified_blind_questions(questions)
    provenance = source_adapter_identity(hf_base, source_adapter)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    if torch.cuda.device_count() != 1 or not torch.cuda.is_available():
        raise RuntimeError("ACTUAL_SINGLE_GPU_QWEN3_4B_REQUIRED")
    model_root = _private_file(hf_base,must_exist=False)
    tokenizer = AutoTokenizer.from_pretrained(model_root,
        local_files_only=True, trust_remote_code=False)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype)
    base = AutoModelForCausalLM.from_pretrained(model_root,
        trust_remote_code=False,local_files_only=True,quantization_config=quant,
        torch_dtype=dtype,device_map={"":0},use_safetensors=True)
    model = PeftModel.from_pretrained(base,source_adapter,is_trainable=False,
                                     local_files_only=True)
    model.eval()
    device = next(model.parameters()).device
    # Prevent labels/keys from entering this process: no outcome-key file arg.
    # Explicit no-truncation and one deterministic next-token scoring per case.
    prepared = []
    for case in cases:
        messages=[
            {"role":"system","content":"Ты — осторожный исследователь рыночных сценариев, не торговый исполнитель."},
            {"role":"user","content":market_prompt(case)},
        ]
        prefix, token_ids = candidate_first_token_ids(tokenizer,messages)
        if len(prefix)>3072:
            raise ValueError("PROMPT_EXCEEDS_FIXED_QWEN4B_EVALUATION_CONTEXT")
        prepared.append((case["case_id"],prefix,token_ids))
    scores={"baseline":[],"adapted":[]}
    with torch.inference_mode():
        for stage in ("baseline","adapted"):
            ctx=model.disable_adapter() if stage=="baseline" else nullcontext()
            with ctx:
                for case_id,prefix,target_ids in prepared:
                    x=torch.tensor([prefix],dtype=torch.long,device=device)
                    logits=model(input_ids=x,attention_mask=torch.ones_like(x),
                                 use_cache=False).logits[0,-1]
                    chosen=logits[target_ids].float()
                    if not torch.isfinite(chosen).all().item():
                        raise ValueError("NONFINITE_QWEN_BLIND_SCORE")
                    probs=torch.softmax(chosen,dim=-1).detach().cpu().tolist()
                    scores[stage].append({"case_id":case_id,"probabilities":
                        dict(zip(CLASSES,map(float,probs)))})
    if len(scores["baseline"]) != 30 or len(scores["adapted"]) != 30:
        raise ValueError("INCOMPLETE_30_CASE_BOTH_MODEL_COHORTS")
    receipt={"mode":"PRECOMMITTED_QWEN3_4B_REAL_H1_NEXT_TOKEN_COMPARISON",
             "market_questions_file_sha256":question_sha,
             "model_provenance":provenance,
             "cases_each_model":30,
             "scoring":"SOFTMAX_ON_FIRST_ANSWER_CLASS_TOKENS_NOT_CALIBRATED_POSTERIORS",
             "true_future_outcomes_loaded_in_prediction":False,
             "trading_orders_created":False,
             "independent_expert_wave_labels":False,
             "research_only":True,"execution_allowed":False,
             "source_revision":os.environ.get("GITHUB_SHA")}
    final={"receipt":receipt,**scores}
    target=_private_file(new_output,must_exist=False)
    target.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    with target.open("x",encoding="utf-8") as f:
        json.dump(final,f,ensure_ascii=False,indent=2,allow_nan=False)
    target.chmod(0o600)
    return receipt


def grade_pair(*, predictions: Path, withheld_outcomes: Path,
               new_report: Path) -> dict:
    """SECOND stage only, after immutable prediction precommit exists."""
    data = strict_json(_private_file(predictions).read_bytes())
    receipt=data.get("receipt",{})
    if (receipt.get("mode")!="PRECOMMITTED_QWEN3_4B_REAL_H1_NEXT_TOKEN_COMPARISON"
            or receipt.get("true_future_outcomes_loaded_in_prediction") is not False
            or len(data.get("baseline",[]))!=30
            or len(data.get("adapted",[]))!=30):
        raise ValueError("PAIRED_MODEL_PRECOMMIT_REQUIRED_BEFORE_OUTCOME_ACCESS")
    keys = _read_jsonl(withheld_outcomes, max_bytes=2_000_000)
    if len(keys)!=30:
        raise ValueError("THIRTY_WITHHELD_OUTCOME_KEYS_REQUIRED")
    before=grade_blind_predictions(keys,data["baseline"])
    after=grade_blind_predictions(keys,data["adapted"])
    delta_brier=before["multiclass_brier"]-after["multiclass_brier"]
    report={
        "mode":"REAL_H1_FIXED_HORIZON_QWEN3_4B_BEFORE_AFTER_HOLDOUT_REPORT",
        "before":before,"after":after,
        "accuracy_delta":after["top1_accuracy"]-before["top1_accuracy"],
        "brier_improvement":delta_brier,
        "simple_holdout_improved_brier_only":delta_brier>0,
        "30_small_observations_not_statistically_reliable_for_trading":True,
        "no_expert_wave_wm1_wm2_wm3_labels":True,
        "no_transaction_cost_backtest":True,
        "no_calibration_on_external_market_cohort":True,
        "private_book_text_not_in_this_report":True,
        "trading_qualification":False,"execution_allowed":False,
    }
    new_report=_private_file(new_report,must_exist=False)
    if new_report.exists() or new_report.is_symlink():
        raise ValueError("GRADE_REPORT_OUTPUT_MUST_BE_NEW")
    new_report.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    with new_report.open("x",encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    new_report.chmod(0o600)
    return report


def main():
    p=argparse.ArgumentParser(description="Actual Qwen3-4B text-only LoRA -> blind REAL H1 chart skill evaluation")
    commands=p.add_subparsers(dest="command",required=True)
    pred=commands.add_parser("predict",help="actual one-GPU base/adapted prediction, NO FUTURE KEYS")
    pred.add_argument("--questions",type=Path,required=True)
    pred.add_argument("--hf-base",type=Path,required=True)
    pred.add_argument("--source-adapter",type=Path,required=True)
    pred.add_argument("--new-output",type=Path,required=True)
    grade=commands.add_parser("grade",help="read future keys only after predictions sealed")
    grade.add_argument("--predictions",type=Path,required=True)
    grade.add_argument("--withheld-outcomes",type=Path,required=True)
    grade.add_argument("--new-report",type=Path,required=True)
    a=p.parse_args()
    result=(predict_pair(questions=a.questions,hf_base=a.hf_base,
        source_adapter=a.source_adapter,new_output=a.new_output)
        if a.command=="predict" else
        grade_pair(predictions=a.predictions,
             withheld_outcomes=a.withheld_outcomes,new_report=a.new_report))
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
