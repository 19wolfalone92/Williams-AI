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
import math
from pathlib import Path
import re
from typing import Any

from .book_index import ROOT, _private_database
from .book_rules import registry
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
    p=Path(path).expanduser().resolve()
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
            try: obj=json.loads(line)
            except (ValueError,TypeError) as e:
                raise ValueError("INVALID_JSONL_RECORD") from e
            if not isinstance(obj,dict) or set(obj) not in (
                    _REQUIRED, _REQUIRED | CHART_FIELDS):
                raise ValueError("UNRECOGNIZED_SUPERVISION_FIELDS")
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
    if not 128<=max_tokens<=4096:
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
    if not path.is_file() or path.stat().st_size>1_000_000:
        raise ValueError("LOCAL_QWEN_CONFIG_REQUIRED")
    payload=path.read_bytes()
    try:
        config=json.loads(payload)
    except (ValueError,UnicodeDecodeError) as exc:
        raise ValueError("LOCAL_QWEN_CONFIG_REQUIRED") from exc
    if (not isinstance(config,dict) or not isinstance(config.get("model_type"),str)
            or config["model_type"] not in
            {"qwen2","qwen2_moe","qwen3","qwen3_moe"}):
        raise ValueError("LOCAL_CAUSAL_QWEN_MODEL_REQUIRED")
    return {"model_type":config["model_type"],
            "config_sha256":hashlib.sha256(payload).hexdigest()}


def train_private_lora(
    *, data:Path, model_dir:Path, output_dir:Path,
    max_steps:int=100,max_tokens:int=1024,
    source_only_research:bool=False,
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
    base_receipt=checked_local_qwen_config(base)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("PRIVATE_LORA_TRAINING_REQUIRES_LOCAL_CUDA_GPU")
    from transformers import AutoModelForCausalLM,AutoTokenizer,Trainer,TrainingArguments
    from peft import LoraConfig,TaskType,get_peft_model
    tokenizer=AutoTokenizer.from_pretrained(
        str(base),local_files_only=True,trust_remote_code=False)
    if tokenizer.eos_token is None:
        raise ValueError("EOS_TOKEN_REQUIRED")
    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token
    trainrows=_tokenized(audited["items_for_local_fit"]["train"],tokenizer,
                        max_tokens=max_tokens)
    validation=_tokenized(audited["items_for_local_fit"]["validation"],tokenizer,
                          max_tokens=max_tokens)
    use_bf16=torch.cuda.is_bf16_supported()
    base_model=AutoModelForCausalLM.from_pretrained(
        str(base),local_files_only=True,trust_remote_code=False,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,device_map="auto")
    model=get_peft_model(base_model,LoraConfig(
        r=8,lora_alpha=16,lora_dropout=0.05,
        target_modules=["q_proj","v_proj","o_proj"],
        bias="none",task_type=TaskType.CAUSAL_LM))
    model.config.use_cache=False
    class Dataset:
        def __init__(self,items):self.items=items
        def __len__(self):return len(self.items)
        def __getitem__(self,i):return self.items[i]
    def collate(batch):
        size=max(len(x["input_ids"]) for x in batch)
        def pad(key,filler):
            return torch.tensor([x[key]+[filler]*(size-len(x[key]))
                                 for x in batch],dtype=torch.long)
        return {"input_ids":pad("input_ids",tokenizer.pad_token_id),
                "attention_mask":pad("attention_mask",0),
                "labels":pad("labels",-100)}
    # TrainingArguments/Trainer implementation may change across versions.
    # Dependencies are imported here, never during the read-only test suite.
    target.mkdir(parents=True,exist_ok=False,mode=0o700)
    target.chmod(0o700)
    args=TrainingArguments(
        output_dir=str(target),
        max_steps=max_steps,per_device_train_batch_size=1,
        per_device_eval_batch_size=1,gradient_accumulation_steps=8,
        learning_rate=1e-4,logging_steps=20,save_strategy="no",
        report_to="none",remove_unused_columns=False,seed=42,
        bf16=use_bf16,fp16=not use_bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant":False},
    )
    trainer=Trainer(model=model,args=args,train_dataset=Dataset(trainrows),
                    eval_dataset=Dataset(validation),data_collator=collate)
    baseline=float(trainer.evaluate()["eval_loss"])
    trainer.train()
    trained=float(trainer.evaluate()["eval_loss"])
    if not all(math.isfinite(x) for x in (baseline,trained)):
        raise ValueError("NONFINITE_VALIDATION_LOSS_REJECT_ADAPTER")
    # A validation-only improvement is not evidence of new-market profit.
    improved=trained<baseline
    if improved:
        # The private output directory was created before Trainer logging.
        target.mkdir(parents=True,exist_ok=True,mode=0o700)
        target.chmod(0o700)
        model.save_pretrained(str(target),safe_serialization=True)
        tokenizer.save_pretrained(str(target))
        (target/"research_manifest.json").write_text(
            json.dumps({
                "mode":"LOCAL_RESEARCH_LORA_NOT_TRADING_READY",
                "source_dataset_sha256":audited["source_dataset_sha256"],
                "base_local_model":base.name,
                "local_base_config":base_receipt,
                "optimizer_steps":int(trainer.state.global_step),
                "source_only_research":source_only_research,
                "chart_examples_by_split":audited["chart_examples_by_split"],
                "market_supervision_is_numeric_not_image_vision":True,
                "baseline_validation_loss":baseline,
                "trained_validation_loss":trained,
                "test_labels_not_supplied_to_trainer":True,
                "no_holdout_market_profitability_verified":True,
                "execution_allowed":False,"trade_action":None,
            },sort_keys=True,indent=2),encoding="utf-8")
    return {
        "adapter_saved":improved,
        "research_only":True,
        "source_dataset_sha256":audited["source_dataset_sha256"],
        "source_only_research":source_only_research,
        "chart_examples_by_split":audited["chart_examples_by_split"],
        "local_base_config":base_receipt,
        "baseline_validation_loss":baseline,
        "adapted_validation_loss":trained,
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
    p.add_argument("--max-tokens",type=int,default=1024)
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
            source_only_research=a.source_only_research)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
