"""Independent genuine Qwen3-4B Base + authentic trained v5 LoRA historical100 GPU inference.

Kaggle GPU + CPU data preparation. Independent DIFFERENT backend/protocol
from GGUF llama.cpp CPU; never mix outputs as same protocol. No gold, train,
books, live accounts or trading. All input market data is official SHA ZIP.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import nullcontext
from hashlib import sha256
import json
import os
from pathlib import Path
import time

from scripts.ai_v6_historical100_models import gather
from scripts.ai_v6_historical100_real_binance import run as collect_official_data
from williams_ai.independent_exam_v6 import canonical, FAMILIES
from williams_ai.local_model import _strict_chart_json
from williams_ai.v6_expert_model_runner import prompt_for_case, protocol_sha256
from williams_ai.v6_v5_lora_binding import MODEL_COMMIT, RAW_ADAPTER_SHA, verified_receipt

SCHEMA = "WILLIAMS_V6_HIST100_HF_NF4_GPU_BASE_V5_LORA_NO_GOLD_V1"
MAX_CONTEXT_TOKENS = 8192
MAX_NEW_TOKENS = 256
# Bound T4 attention activation memory; GPU prompt policy differs from CPU GGUF.
GPU_MAX_INPUT_TOKENS = 3200
GPU_VISIBLE_CLOSED_H1_MAX = 48
GPU_VISIBLE_CLOSED_H1_MIN = 20
SEED = 1942


def need(ok, reason):
    if not ok:
        raise ValueError("HIST100_REAL_GPU_" + reason)


def find_trained_lora():
    """Only authentic training-kernel private Kaggle input, never public upload."""
    root = Path("/kaggle/input")
    # Kaggle can nest previous private kernel output under extra directories.
    # Require original adapter + authentic training receipt SHA, not a guess.
    paths = sorted(p for p in root.rglob("adapter_model.safetensors")
                   if "williams_v5_model" in p.parts and
                   p.parent.name == "adapter")
    print("HIST100_V5_PRIVATE_MOUNT_CANDIDATES",len(paths),flush=True)
    need(len(paths) == 1, "NO_UNIQUE_ATTACHED_PRIVATE_V5_TRAINING_OUTPUT")
    adapter_path = paths[0]
    trained_root = adapter_path.parent.parent
    receipt = trained_root / "training_receipt.json"
    config = adapter_path.parent / "adapter_config.json"
    need(config.is_file() and config.stat().st_size < 40000,
         "TRAINED_LORA_CONFIG_ABSENT")
    verified_receipt(receipt, adapter_path)
    params = json.loads(config.read_text(encoding="utf-8"))
    need(params.get("peft_type") == "LORA" and
         params.get("r") == 8 and
         "Qwen3-4B" in str(params.get("base_model_name_or_path", "")),
         "WRONG_ADAPTER_CONFIGURATION")
    return adapter_path.parent


def compact_gpu_prompt(tokenizer, original_messages):
    """Causally compact ONLY the HF GPU prompt; never change stored 1024 H1 input.

    A 96-candle HF prompt caused real torch.OutOfMemoryError on Tesla T4
    (6.96GiB transient attention allocation). Freeze a smaller, explicitly
    different GPU prompt policy, preserving unchanged book rules and
    full-history derived causal H1/H4/D1 technical features. Both model roles
    receive EXACTLY identical compact bytes. Refuse a case rather than
    silently throw away all price context.
    """
    need(len(original_messages) == 2, "MALFORMED_TWO_MESSAGE_GPU_PROMPT")
    base=json.loads(original_messages[1]["content"])
    original_bars=base["closed_bars"]
    need(isinstance(original_bars,list) and
         len(original_bars) >= GPU_VISIBLE_CLOSED_H1_MAX,
         "INSUFFICIENT_ORIGINAL_CANDLE_CONTEXT")
    base["closed_bars"]=original_bars[-GPU_VISIBLE_CLOSED_H1_MAX:]
    while True:
        messages=[original_messages[0],
                  {"role":"user","content":canonical(base).decode("utf-8")}]
        tokens=tokenizer.apply_chat_template(
            messages,tokenize=True,add_generation_prompt=True,
            enable_thinking=False)
        count=len(tokens)
        if count <= GPU_MAX_INPUT_TOKENS:
            return messages, sha256(canonical(messages)).hexdigest(), count, len(base["closed_bars"])
        need(len(base["closed_bars"]) > GPU_VISIBLE_CLOSED_H1_MIN,
             "GPU_PROMPT_TOO_LARGE_WITH_20_CLOSED_BARS")
        base["closed_bars"]=base["closed_bars"][2:]


def decode_hf(model, tokenizer, messages, decision_choices, *, use_adapter, torch):
    from contextlib import nullcontext
    import time
    tokenized = tokenizer.apply_chat_template(messages, tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
    input_len = tokenized.shape[-1]
    need(input_len + MAX_NEW_TOKENS <= MAX_CONTEXT_TOKENS,
         "GPU_PROMPT_EXCEEDS_PINNED_CONTEXT")
    tokenized = tokenized.to("cuda:0")
    context = nullcontext() if use_adapter else model.disable_adapter()
    start = time.monotonic()
    with context, torch.inference_mode():
        generation = model.generate(input_ids=tokenized,
            attention_mask=torch.ones_like(tokenized), do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS, use_cache=True,
            pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    out = generation[0][input_len:]
    tokens = len(out)
    text = tokenizer.decode(out, skip_special_tokens=True).strip()
    if tokens >= MAX_NEW_TOKENS and not out[-1].item() == tokenizer.eos_token_id:
        return "ABSTAIN", "length", None, round(time.monotonic()-start, 3), input_len, tokens
    try:
        answer = _strict_chart_json(text)
    except (TypeError, ValueError):
        answer = None
    if (not isinstance(answer, dict) or set(answer) != {"decision", "reason"} or
        answer.get("decision") not in decision_choices or
        not isinstance(answer.get("reason"), str) or
        not 1 <= len(answer["reason"]) <= 240):
        return "ABSTAIN", "error", None, round(time.monotonic()-start, 3), input_len, tokens
    return answer["decision"], "stop", answer["reason"], round(time.monotonic()-start, 3), input_len, tokens


def write_json_atomic(dest, data):
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_bytes(canonical(data))
    os.replace(tmp, dest)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--plan", type=Path, required=True)
    a.add_argument("--out-dir", type=Path, required=True)
    args = a.parse_args()
    need(not args.out_dir.exists(), "OUTPUT_ALREADY_EXISTS")
    args.out_dir.mkdir(parents=True)
    import torch
    need(torch.cuda.is_available(), "KAGGLE_GPU_NOT_ALLOCATED")
    need(torch.cuda.device_count() >= 1, "GPU_DEVICE_NOT_FOUND")
    # CPU prepares and validates 1024+24 bars and calculates source features.
    chart_root = args.out_dir / "official_hist100"
    data = collect_official_data(args.plan, chart_root)
    need(data["distinct_chart_windows"] == 100, "MARKET_DATA_INCOMPLETE")
    selected, dataset_sha = gather(args.plan, chart_root)
    adapter = find_trained_lora()
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-4B", revision=MODEL_COMMIT, trust_remote_code=False)
    quant = BitsAndBytesConfig(load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16)
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B",
        revision=MODEL_COMMIT, trust_remote_code=False, device_map={"":0},
        quantization_config=quant, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
    model.eval()
    need(next(model.parameters()).is_cuda, "MODEL_NOT_ON_GPU")
    print("REAL_KAGGLE_CUDA_MODEL_READY", torch.cuda.get_device_name(0), flush=True)
    rows = {"matched_base": [], "candidate_v5": []}
    receipt = {
        "schema": SCHEMA, "complete": False,
        "cases_total": 100, "completed_paired_cases": 0,
        "source": data["source"], "source_zip_sha256_verified": True,
        "market_selection_plan_sha256": dataset_sha,
        "original_cpu_prompt_protocol_sha256": protocol_sha256(),
        "gpu_decoding_backend": "HuggingFace Transformers v4.57 + PEFT + bitsandbytes NF4",
        "gpu_prompt_max_input_tokens": GPU_MAX_INPUT_TOKENS,
        "gpu_prompt_causal_closed_bars_initial": GPU_VISIBLE_CLOSED_H1_MAX,
        "gpu_prompt_causal_closed_bars_min": GPU_VISIBLE_CLOSED_H1_MIN,
        "gpu_prompt_is_compacted_vs_cpu_protocol": True,
        "gpu_protocol_not_bit_identical_to_llama_cpp_gguf": True,
        "qwen_base_revision": MODEL_COMMIT,
        "verified_genuine_trained_adapter_sha256": RAW_ADAPTER_SHA,
        "same_input_prompt_hashes_for_both_models": True,
        "cuda_is_available": True, "gpu_name": torch.cuda.get_device_name(0),
        "cpu_preparation_and_gpu_inference": True,
        "real_model_inference_rows": 0,
        "actual_book_accuracy": None, "expert_gold": None,
        "source_train_overlap_independently_audited": False,
        "prospective_exam": False, "no_trading": True,
        "observed": rows,
    }
    out = args.out_dir / "ACTUAL_GPU_HIST100_QWEN_BASE_V5_NO_GOLD.json"
    write_json_atomic(out, receipt)
    for i, (case, chart, evidence, chart_sha) in enumerate(selected, start=1):
        messages, meta = prompt_for_case(case, chart)
        need(meta == evidence, "BOOK_SOURCE_CHANGED_DURING_INFERENCE")
        # This step freezes an explicit GPU-only prompt. The same 1024-bar
        # chart remains checksummed; full H1/H4/D1 context is unchanged.
        gpu_messages,gpu_hash,gpu_tokens,visible_bars=compact_gpu_prompt(
            tokenizer,messages)
        available = list(FAMILIES[case["family"]]) + ["ABSTAIN"]
        for role, active in (("matched_base", False), ("candidate_v5", True)):
            if meta["source_missing"]:
                answer = ("ABSTAIN", "stop", None, 0.0, 0, 0)
            else:
                answer = decode_hf(model, tokenizer, gpu_messages, available,
                                   use_adapter=active, torch=torch)
            decision,finish,reason,elapsed,ntokens,generated=answer
            rows[role].append({
                "id": case["id"], "edition": case["edition"],
                "family": case["family"], "symbol": chart["symbol"],
                "cutoff_utc": case["cutoff_utc"], "decision": decision,
                "finish_reason": finish, "model_reason": reason,
                "chart_sha256": chart_sha,
                "prompt_sha256": gpu_hash,
                "original_cpu_full_prompt_sha256": evidence["prompt_sha256"],
                "gpu_prompt_tokens": gpu_tokens,
                "gpu_visible_closed_h1_bars": visible_bars,
                "retrieved_rule_ids": evidence["retrieved_rule_ids"],
                "source_missing": evidence["source_missing"],
                "duration_sec": elapsed,
                "tokens_in": ntokens, "tokens_out": generated,
            })
        torch.cuda.empty_cache()
        receipt["completed_paired_cases"] = i
        receipt["real_model_inference_rows"] = i*2
        write_json_atomic(out, receipt)
        if i % 5 == 0:
            print("REAL_KAGGLE_GPU_BASE_V5_HIST100_PROGRESS",i,100,flush=True)
    need(all(len(rows[role])==100 for role in rows),"NOT_EXACTLY_100_PAIRED")
    for left,right in zip(rows["matched_base"], rows["candidate_v5"], strict=True):
        need(all(left[k]==right[k] for k in ("id","edition","family","symbol","cutoff_utc",
                                           "chart_sha256","prompt_sha256","retrieved_rule_ids")),
             "MISMATCHED_MODEL_INPUTS")
    receipt["complete"] = True
    receipt["summary"] = {
        role: {
            "decisions": dict(Counter(x["decision"] for x in rs)),
            "finish_reasons": dict(Counter(x["finish_reason"] for x in rs)),
            "reasons_240_chars": sum(len(x.get("model_reason") or "")==240 for x in rs),
        } for role,rs in rows.items()
    }
    receipt["decision_disagreements"] = sum(
        x["decision"]!=y["decision"] for x,y in
        zip(rows["matched_base"],rows["candidate_v5"],strict=True))
    write_json_atomic(out,receipt)
    # Output one technical receipt only, not weights/books or future prices.
    print("REAL_KAGGLE_GPU_100_BASE_100_V5_COMPLETED_NOT_EXPERT_GOLD",
          json.dumps({"pairs":100,"disagreements":receipt["decision_disagreements"],
                      "summary":receipt["summary"]},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
