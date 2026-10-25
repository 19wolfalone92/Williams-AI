"""PRIVATE, reviewed-only Qwen LoRA adaptation; no automatic promotion.

GGUF (quantized inference weights) is NOT a trainable Hugging Face base model.
This module requires a locally downloaded, unquantized HF causal LM and GPU.
Only explicitly independently-reviewed SOURCE + CHART teaching examples may
update adapter weights. Raw Qwen guesses, paper PnL and source summaries are
not approved training labels. No data upload, networking, secrets or orders.

This is an optional controlled path to ACTUAL language-model weight updates,
not a claim that these updates have been run or will increase profits.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from typing import Any

from .book_index import ROOT, _private_database
from .book_rules import registry
from .lora_integrity import EDITIONS, local_hf_identity, regular_sha256, strict_json
from .lora_training import LoraPlan, fit_lora
from .chart_supervision import (
    CHART_FIELDS, checked_chart_lesson, check_market_partitions,
    chart_prompt_observations, require_market_supervision,
)

_REQUIRED=frozenset({
    "case_id","edition","rule_id","pdf_page","source_sha256",
    "question","answer","review_status","reviewer_id","split",
})
_SPLITS=("train","validation","test")
_ID=re.compile(r"^[A-Za-z0-9_.:-]{6,100}$")
_HASH=re.compile(r"^[0-9a-f]{64}$")


def _private_file(path:str|Path,*,must_exist:bool=True)->Path:
    original=Path(path).expanduser()
    if original.is_symlink():
        raise ValueError("PRIVATE_REGULAR_PATH_REQUIRED")
    p=original.resolve()
    if p==ROOT or ROOT in p.parents:
        raise ValueError("LICENSED_DATA_AND_MODEL_ADAPTERS_MUST_STAY_OUTSIDE_GIT")
    if must_exist and (not p.is_file() or Path(path).is_symlink()):
        raise ValueError("PRIVATE_REVIEWED_DATASET_FILE_REQUIRED")
    return p


def checked_private_training_data(path:str|Path, *,
                                  min_train:int=64,min_validation:int=16,
                                  min_test:int=16)->dict[str,Any]:
    """Validate provenance, explicit human approval and withheld test cohort.

    The review flag is an operator declaration. This program cannot verify
    human expertise or prove label quality independently.
    """
    file=_private_file(path)
    if file.suffix.lower()!=".jsonl" or file.stat().st_size>40_000_000:
        raise ValueError("DATASET_MUST_BE_SMALL_PRIVATE_JSONL")
    for n in (min_train,min_validation,min_test):
        if not isinstance(n,int) or isinstance(n,bool) or n<1:
            raise ValueError("INVALID_MINIMUM_BY_SPLIT")
    books={x["id"]:x for x in registry()["books"]}
    rules={x["id"]:x for x in registry()["rule_cards"]}
    groups={k:[] for k in _SPLITS}
    ids=set()
    case_hashes=set()
    rule_split_owners={}
    per_edition=Counter()
    per_split_editions={k:Counter() for k in _SPLITS}
    chart_rows=[]
    # Fit exactly the bytes whose digest is recorded, even if the operator
    # replaces the input file while the audit/tokenization is running.
    with file.open("rb") as source:
        payload=source.read(40_000_001)
    if len(payload)>40_000_000:
        raise ValueError("DATASET_MUST_BE_SMALL_PRIVATE_JSONL")
    try:
        text=payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("PRIVATE_DATASET_MUST_BE_UTF8") from exc
    with io.StringIO(text) as reader:
        for lineno,line in enumerate(reader,1):
            if not line.strip():
                continue
            if lineno>100000:
                raise ValueError("PRIVATE_REVIEWED_SET_TOO_LARGE")
            try: obj=strict_json(line)
            except (ValueError,TypeError) as e:
                raise ValueError("INVALID_JSONL_RECORD") from e
            if not isinstance(obj,dict) or set(obj) not in (
                    _REQUIRED, _REQUIRED | CHART_FIELDS):
                raise ValueError("UNRECOGNIZED_SUPERVISION_FIELDS")
            if any(not isinstance(obj[k],str) for k in
                   ("case_id","edition","rule_id","source_sha256","split")):
                raise ValueError("INVALID_SUPERVISION_FIELD_TYPES")
            case_id=obj["case_id"]
            ed=obj["edition"]
            rule=rules.get(obj["rule_id"])
            page=obj["pdf_page"]
            if (not isinstance(case_id,str) or not _ID.fullmatch(case_id)
                    or case_id in ids or ed not in books or rule is None
                    or rule["source"]!=ed or page not in rule["pdf_pages"]
                    or isinstance(page,bool) or not isinstance(page,int)
                    or obj["source_sha256"]!=books[ed]["sha256"]):
                raise ValueError("DUPLICATE_OR_UNVERIFIED_BOOK_RULE_REFERENCE")
            if (rule.get("status")!="REVIEWED_PAGES"
                    or obj["review_status"]!="INDEPENDENT_EXPERT_REVIEWED"
                    or not isinstance(obj["reviewer_id"],str)
                    or not _ID.fullmatch(obj["reviewer_id"])):
                raise ValueError("UNREVIEWED_SOURCE_LABEL_MUST_NOT_TRAIN_MODEL")
            question,answer=obj["question"],obj["answer"]
            if (not isinstance(question,str) or not 20<=len(question)<=2400
                    or not isinstance(answer,str) or not 20<=len(answer)<=2500):
                raise ValueError("EMPTY_OR_OVERSIZED_REVIEWED_LESSON")
            normalized=hashlib.sha256(
                re.sub(r"\s+"," ",question).strip().casefold().encode()).hexdigest()
            if normalized in case_hashes:
                raise ValueError("DUPLICATE_QUESTION_ACROSS_HOLDOUT_PARTITIONS")
            case_hashes.add(normalized)
            if obj["split"] not in groups:
                raise ValueError("UNREGISTERED_TRAINING_SPLIT")
            source_rule=(ed,obj["rule_id"])
            owner=rule_split_owners.setdefault(source_rule,obj["split"])
            if owner!=obj["split"]:
                raise ValueError("SAME_BOOK_RULE_LEAKED_ACROSS_FROZEN_SPLITS")
            if CHART_FIELDS <= set(obj):
                chart_rows.append(checked_chart_lesson(obj))
            ids.add(case_id)
            per_edition[ed]+=1
            per_split_editions[obj["split"]][ed]+=1
            groups[obj["split"]].append(obj)
    if (len(groups["train"])<min_train
            or len(groups["validation"])<min_validation
            or len(groups["test"])<min_test
            or set(per_edition)!=set(books)):
        raise ValueError("INSUFFICIENT_REVIEWED_INDEPENDENT_SOURCE_EXAMPLES")
    if any(set(counts)!=set(books) for counts in per_split_editions.values()):
        raise ValueError("FOUR_BOOKS_IN_EACH_SPLIT_REQUIRED")
    digest=hashlib.sha256(payload).hexdigest()
    market_audit=check_market_partitions(chart_rows)
    return {
        "mode":"LOCAL_INDEPENDENT_EXPERT_DECLARED_SOURCE_TRAINING_SET",
        "source_dataset_sha256":digest,
        "counts":{k:len(v) for k,v in groups.items()},
        "source_editions":dict(sorted(per_edition.items())),
        "source_editions_by_split":{k:dict(sorted(v.items())) for k,v in per_split_editions.items()},
        "test_holdout_loaded_by_training_loop":False,
        "reviewer_declaration_not_independently_verified":True,
        "individual_book_rules_never_cross_splits":True,
        **market_audit,
        "items_for_local_fit":{k:groups[k] for k in ("train","validation")},
        "no_execution":True,
    }


def _tokenized(cases:list[dict[str,Any]],tokenizer:Any,*,max_tokens:int)->list[dict[str,list[int]]]:
    """Mask prompt tokens: train ONLY on reviewed answer, never copy the task."""
    if type(max_tokens) is not int or not 128<=max_tokens<=4096:
        raise ValueError("INVALID_LANGUAGE_MODEL_CONTEXT")
    examples=[]
    for item in cases:
        system="Проверяемая система Билла Вильямса. Исследование, не торговый ордер. Приводи первоисточник и учитывай альтернативные волновые гипотезы."
        question=(f"Источник {item['edition']} PDF {item['pdf_page']}; "
                  f"правило {item['rule_id']}.\n{item['question']}")
        if "market_context" in item:
            # Revalidate approvals at the inference handoff, never accept
            # an answer-derived feature or a model's precomputed wave label.
            checked_chart_lesson(item)
            observations=chart_prompt_observations(item["market_context"])
            question += "\nЗакрытый рыночный контекст:\n" + json.dumps(
                observations, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        messages=[{"role":"system","content":system},{"role":"user","content":question}]
        prefix=tokenizer.apply_chat_template(messages,
            tokenize=True,add_generation_prompt=True,enable_thinking=False)
        # Render the COMPLETE training conversation with the tokenizer's
        # real assistant/end-of-turn tokens. Concatenating an encoded answer
        # and eos by hand can corrupt Qwen's thinking/turn boundaries.
        complete=tokenizer.apply_chat_template(
            messages+[{"role":"assistant","content":item["answer"]}],
            tokenize=True,add_generation_prompt=False,enable_thinking=False)
        if (not isinstance(prefix,list) or not isinstance(complete,list)
                or not prefix or complete[:len(prefix)]!=prefix):
            raise ValueError("QWEN_TRAINING_TEMPLATE_MUST_PRESERVE_PROMPT_PREFIX")
        response=complete[len(prefix):]
        if not response or len(complete)>max_tokens:
            raise ValueError("REVIEWED_CASE_TOO_LONG_FOR_CONTEXT_NO_TRUNCATION")
        examples.append({
            "input_ids":complete,
            "attention_mask":[1]*len(complete),
            "labels":[-100]*len(prefix)+response,
        })
    return examples


def checked_local_qwen_config(base: Path) -> dict[str,str]:
    """Reject another model family before loading optional CUDA libraries.

    The local digest is an integrity receipt, not independent proof that
    the weights were published by Qwen or came from a particular revision.
    """
    path=base/"config.json"
    if path.is_symlink() or not path.is_file() or path.stat().st_size>1_000_000:
        raise ValueError("LOCAL_QWEN_CONFIG_REQUIRED")
    payload=path.read_bytes()
    try:
        config=strict_json(payload)
    except (ValueError,UnicodeDecodeError) as exc:
        raise ValueError("LOCAL_QWEN_CONFIG_REQUIRED") from exc
    if (not isinstance(config,dict) or not isinstance(config.get("model_type"),str)
            or config["model_type"] not in
            {"qwen2","qwen2_moe","qwen3","qwen3_moe"}):
        raise ValueError("LOCAL_CAUSAL_QWEN_MODEL_REQUIRED")
    if config.get("quantization_config") is not None or config.get("auto_map"):
        raise ValueError("UNQUANTIZED_BUILTIN_QWEN_BASE_REQUIRED")
    return {"model_type":config["model_type"],
            "config_sha256":hashlib.sha256(payload).hexdigest()}


def train_private_lora(
    *, data:Path, model_dir:Path, output_dir:Path,
    max_steps:int=100,max_tokens:int=4096,
    source_only_research:bool=False,
    plan:LoraPlan|None=None,
)->dict[str,Any]:
    """Execute actual LoRA updates ONLY on private GPU with reviewed cases."""
    audited=checked_private_training_data(data)
    if not isinstance(source_only_research,bool):
        raise ValueError("SOURCE_ONLY_RESEARCH_MUST_BE_EXPLICIT_BOOLEAN")
    if not source_only_research:
        require_market_supervision(audited)
    base=_private_file(model_dir,must_exist=False)
    target=_private_file(output_dir,must_exist=False)
    if (not base.is_dir() or base.is_symlink() or target==base
            or base in target.parents or target in base.parents
            or target.exists() or Path(model_dir).is_symlink()):
        raise ValueError("LOCAL_BASE_MODEL_AND_NEW_PRIVATE_OUTPUT_REQUIRED")
    if not isinstance(max_steps,int) or isinstance(max_steps,bool) or not 1<=max_steps<=10000:
        raise ValueError("INVALID_TRAINING_STEP_LIMIT")
    if type(max_tokens) is not int or not 128<=max_tokens<=4096:
        raise ValueError("INVALID_LANGUAGE_MODEL_CONTEXT")
    plan=LoraPlan() if plan is None else plan
    if not isinstance(plan,LoraPlan):
        raise ValueError("INVALID_BOUNDED_LORA_PLAN")
    base_receipt=checked_local_qwen_config(base)
    base_identity=local_hf_identity(base)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("PRIVATE_LORA_TRAINING_REQUIRES_LOCAL_CUDA_GPU")
    from transformers import AutoModelForCausalLM,AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(
        str(base),local_files_only=True,trust_remote_code=False)
    if tokenizer.eos_token is None:
        raise ValueError("EOS_TOKEN_REQUIRED")
    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token
    trainrows=_tokenized(audited["items_for_local_fit"]["train"],tokenizer,
                        max_tokens=max_tokens)
    validation={ed:_tokenized([row for row in audited["items_for_local_fit"]["validation"]
                              if row["edition"]==ed],tokenizer,max_tokens=max_tokens)
                for ed in EDITIONS}
    use_bf16=torch.cuda.is_bf16_supported()
    base_model=AutoModelForCausalLM.from_pretrained(
        str(base),local_files_only=True,trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16)
    if local_hf_identity(base)!=base_identity:
        raise ValueError("LOCAL_HF_BASE_CHANGED_DURING_LOAD")
    target.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    # Checkpoints/optimizer diagnostics stay in a private temporary directory.
    # No rejected or interrupted run leaves a deployable adapter behind.
    with TemporaryDirectory(prefix=".williams-lora-",dir=target.parent) as scratch:
        model,fit=fit_lora(base_model,trainrows,validation,workdir=Path(scratch)/"fit",
            pad_token_id=tokenizer.pad_token_id,max_steps=max_steps,plan=plan)
        if local_hf_identity(base)!=base_identity:
            raise ValueError("LOCAL_HF_BASE_CHANGED_DURING_TRAINING")
        improved=fit["validation_gate"]["accepted"]
        baseline=fit["baseline_by_book"]["ALL"]
        trained=fit["adapted_by_book"]["ALL"]
        if improved:
            accepted=Path(scratch)/"accepted"
            accepted.mkdir(mode=0o700)
            model.save_pretrained(str(accepted),safe_serialization=True)
            tokenizer.save_pretrained(str(accepted))
            adapter_weights_sha256=regular_sha256(accepted/"adapter_model.safetensors")
            adapter_config_sha256=regular_sha256(accepted/"adapter_config.json")
            manifest={
                "mode":"LOCAL_RESEARCH_LORA_NOT_TRADING_READY",
                "training_receipt_version":2,
                "source_dataset_sha256":audited["source_dataset_sha256"],
                "base_local_model":base.name,
                "local_base_config":base_receipt,
                "local_base_identity":base_identity,
                **fit,
                "adapter_weights_sha256":adapter_weights_sha256,
                "adapter_config_sha256":adapter_config_sha256,
                "source_only_research":source_only_research,
                "chart_examples_by_split":audited["chart_examples_by_split"],
                "market_supervision_is_numeric_not_image_vision":True,
                "baseline_validation_loss":baseline,
                "trained_validation_loss":trained,
                "test_labels_not_supplied_to_trainer":True,
                "no_holdout_market_profitability_verified":True,
                "execution_allowed":False,"trade_action":None,
            }
            from importlib.metadata import version
            manifest["training_packages"]={name:version(name) for name in
                ("torch","transformers","peft","accelerate","safetensors")}
            (accepted/"research_manifest.json").write_text(
                json.dumps(manifest,sort_keys=True,indent=2,allow_nan=False),encoding="utf-8")
            for path in accepted.iterdir():
                if not path.is_file() or path.is_symlink():
                    raise ValueError("REGULAR_ADAPTER_OUTPUT_FILES_REQUIRED")
                path.chmod(0o600)
            # Exclusive creation rejects an output path appearing mid-fit.
            target.mkdir(mode=0o700,exist_ok=False)
            try:
                for path in sorted(accepted.iterdir(),key=lambda p:p.name=="research_manifest.json"):
                    os.replace(path,target/path.name)
            except BaseException:
                shutil.rmtree(target)
                raise
    return {
        "adapter_saved":improved,
        "research_only":True,
        "source_dataset_sha256":audited["source_dataset_sha256"],
        "source_only_research":source_only_research,
        "chart_examples_by_split":audited["chart_examples_by_split"],
        "local_base_config":base_receipt,
        "baseline_validation_loss":baseline,
        "adapted_validation_loss":trained,
        **fit,
        "test_set_never_used_for_training_or_selection":True,
        "new_market_trading_skill_proven":False,
        "execution_allowed":False,"trade_action":None,
    }


def main()->None:
    p=argparse.ArgumentParser(description="Private independent-reviewed-source GPU LoRA research only")
    p.add_argument("--reviewed-jsonl",type=Path,required=True)
    p.add_argument("--local-hf-model",type=Path,required=True)
    p.add_argument("--private-adapter-output",type=Path,required=True)
    p.add_argument("--steps",type=int,default=100)
    p.add_argument("--max-tokens",type=int,default=4096)
    p.add_argument("--rank",type=int,choices=(4,8,16,32,64),default=16)
    p.add_argument("--learning-rate",type=float,default=5e-5)
    p.add_argument("--target-profile",choices=("attention","attention-mlp"),default="attention")
    p.add_argument("--gradient-accumulation",type=int,default=8)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--audit-only",action="store_true")
    p.add_argument("--source-only-research",action="store_true",
                   help="Explicit text-only experiment; cannot claim market training")
    a=p.parse_args()
    if a.audit_only:
        result=checked_private_training_data(a.reviewed_jsonl)
        result.pop("items_for_local_fit",None)
    else:
        result=train_private_lora(data=a.reviewed_jsonl,
            model_dir=a.local_hf_model,output_dir=a.private_adapter_output,
            max_steps=a.steps,max_tokens=a.max_tokens,
            source_only_research=a.source_only_research,
            plan=LoraPlan(rank=a.rank,learning_rate=a.learning_rate,
                target_profile=a.target_profile,gradient_accumulation=a.gradient_accumulation,
                seed=a.seed))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
