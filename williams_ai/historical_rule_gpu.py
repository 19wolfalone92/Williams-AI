"""Actual two-T4 book + historical-rule pilot; no discretionary chart gold."""
from __future__ import annotations
from contextlib import nullcontext
from datetime import timedelta
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import shutil
import threading
import time
from urllib.request import Request, urlopen

from .historical_rule_curriculum import (SCHEMA, FAMILIES, SYSTEM, audit_cases, canonical,
    digest, prompts_only, score)
from .deep_book_research import MODEL, REVISION, EDITIONS, checked_payload
from .restorative_book_practice import INITIAL_SHA, PREVIOUS40_SHA, tokenize_reasoned
from .contrastive_book_practice import tokenize_typed, weighted_causal_loss
from .lora_integrity import validation_admission

RUN_SLUG = "williams-ai-historical-books-20261029"
CONTROL_URL = "https://raw.githubusercontent.com/19wolfalone92/Williams-AI/research/private-four-book-deep-training-20261027/experiments/historical-rule-training-control.json"
STAGES = ("base", "previous40", "focused69", "candidate")
PROTOCOL = {"mode": "direct", "max_new_tokens": 256, "seconds": 25,
    "do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0}


def selection(grade, source, practice, references):
    if set(grade["per_rule"]) != set(FAMILIES):
        raise ValueError("ALL_NINE_DEVELOPMENT_RULES_REQUIRED")
    regression = max(after[e] / before[e] - 1 for after, before in (
        (source, references["focused69_source"]), (source, references["previous40_source"]),
        (practice, references["focused69_practice"]), (practice, references["previous40_practice"])) for e in EDITIONS)
    if not math.isfinite(regression): raise ValueError("FINITE_SOURCE_PRESERVATION_REQUIRED")
    minimum = min(v["correct"] / v["total"] for v in grade["per_rule"].values())
    collapsed = len(grade["constant_fields_despite_varied_expected_values"])
    metric = minimum + .1 * grade["accuracy"] - .1 * collapsed
    if regression > .02: metric = -3 - regression
    return {"score": metric, "max_any_reference_book_regression": regression,
        "minimum_rule_accuracy": minimum, "collapsed_rules": collapsed,
        "preserved_within_two_percent": regression <= .02}


class StopWatcher:
    def __init__(self):
        self.requested = threading.Event(); self.finished = threading.Event()

    def start(self):
        def watch():
            failed_since = None
            while not self.finished.is_set():
                try:
                    with urlopen(Request(CONTROL_URL + "?t=" + str(int(time.time())//20), headers={"Cache-Control": "no-cache"}), timeout=8) as response:
                        obj = json.load(response)
                    if set(obj) != {"run_slug", "stop"} or obj["run_slug"] != RUN_SLUG or type(obj["stop"]) is not bool:
                        raise ValueError("EXPLICIT_HISTORICAL_STOP_CONTROL_REQUIRED")
                    if obj["stop"]: self.requested.set(); return
                    failed_since = None
                except Exception:
                    failed_since = failed_since or time.monotonic()
                    if time.monotonic() - failed_since > 300: self.requested.set(); return
                self.finished.wait(20)
        threading.Thread(target=watch, daemon=True).start()


def train(source_path: Path, curriculum_path: Path, questions_path: Path,
          initial_path: Path, previous_path: Path, reference_path: Path, output: Path):
    import torch
    import torch.distributed as dist
    from huggingface_hub import HfApi, snapshot_download
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
        Trainer, TrainerCallback, TrainingArguments, StoppingCriteria, StoppingCriteriaList, set_seed)
    from .private_book_source import tokenize_source_pages
    from .private_lora import _tokenized
    from .structured_rule_practice import balanced_recovery_curriculum

    started = time.monotonic(); rank = int(os.environ["LOCAL_RANK"]); world = int(os.environ["WORLD_SIZE"])
    if world != 2 or torch.cuda.device_count() != 2: raise ValueError("TWO_REAL_CUDA_DEVICES_REQUIRED")
    torch.cuda.set_device(rank); device = torch.device("cuda", rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=20))
    watcher = StopWatcher(); watcher.start(); set_seed(20261029)
    source = checked_payload(json.loads(source_path.read_bytes()))
    payload = json.loads(curriculum_path.read_bytes()); reference = json.loads(reference_path.read_bytes())
    if set(payload) != {"schema", "train", "validation", "audit"} or payload["schema"] != SCHEMA:
        raise ValueError("NO_TEST_ROWS_OR_HUMAN_APPROVALS_IN_TRAINING_PAYLOAD")
    if any(audit_cases(payload[s]) != payload["audit"][s] for s in ("train", "validation")):
        raise ValueError("FROZEN_HISTORICAL_CURRICULUM_CHANGED")
    if (len(payload["train"]) != 1440 or len(payload["validation"]) != 72
            or {r["case_id"] for r in payload["train"]} & {r["case_id"] for r in payload["validation"]}):
        raise ValueError("DISTINCT_FROZEN_TRAIN_AND_DEVELOPMENT_CASES_REQUIRED")
    for path, expected in ((initial_path, INITIAL_SHA), (previous_path, PREVIOUS40_SHA)):
        if sha256((path/"adapter_model.safetensors").read_bytes()).hexdigest() != expected:
            raise ValueError("VERIFIED_INITIAL_AND_REFERENCE_ADAPTERS_REQUIRED")
    if reference["source_audit"] != source["audit"] or reference["candidate_adapter_sha256"] != INITIAL_SHA:
        raise ValueError("SAME_ORIGINAL_SOURCE_SPLITS_REQUIRED")
    if rank == 0:
        output.mkdir(mode=0o700); shutil.copytree(initial_path, output/"source_adapter")
    info = HfApi().model_info(MODEL, revision=REVISION, files_metadata=True)
    if info.sha != REVISION: raise ValueError("PINNED_QWEN3_4B_REQUIRED")
    official = {f.rfilename: f.lfs.sha256 for f in info.siblings if f.rfilename.endswith(".safetensors") and f.lfs}
    if rank == 0:
        base = Path(snapshot_download(MODEL, revision=REVISION, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"]))
        for name, expected in official.items():
            if sha256((base/name).read_bytes()).hexdigest() != expected: raise ValueError("OFFICIAL_BASE_SHA_FAILED")
    dist.barrier(); base = Path(snapshot_download(MODEL, revision=REVISION, local_files_only=True))
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    legacy, validation, _ = balanced_recovery_curriculum()
    source_val = {e: tokenize_source_pages([r for r in source["validation"] if r["edition"] == e], tokenizer, max_tokens=384) for e in EDITIONS}
    practice_val = {e: _tokenized(validation[e], tokenizer, max_tokens=768) for e in EDITIONS}
    def ordinary(rows): return [{**r, "loss_weights": [float(t != -100) for t in r["labels"]]} for r in rows]
    market_tokens = tokenize_typed(payload["train"], tokenizer, max_tokens=1536)
    reasoned = tokenize_reasoned([r for r in payload["train"] if r["rule_id"] in {"TC1.PSY.PLAN", "NTD.AC"}], tokenizer, max_tokens=1536)
    source_tokens = ordinary(tokenize_source_pages(source["train"], tokenizer, max_tokens=384))
    legacy_tokens = ordinary(_tokenized(legacy, tokenizer, max_tokens=768))
    training = market_tokens + reasoned + source_tokens + legacy_tokens
    random.Random(20261029).shuffle(training)
    dev_tokens = tokenize_typed(payload["validation"], tokenizer, max_tokens=1536)
    model = AutoModelForCausalLM.from_pretrained(base, trust_remote_code=False, local_files_only=True, use_safetensors=True,
        dtype=torch.float16, device_map={"": rank}, quantization_config=BitsAndBytesConfig(load_in_4bit=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16))
    if (model.config.model_type != "qwen3" or model.config.hidden_size != 2560 or model.config.num_hidden_layers != 36):
        raise ValueError("EXACT_4B_ARCHITECTURE_REQUIRED")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    model = PeftModel.from_pretrained(model, initial_path, is_trainable=True, local_files_only=True)
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if sum(p.numel() for p in parameters.values()) != 33030144 or any(".lora_" not in n for n in parameters):
        raise ValueError("ONLY_KNOWN_RANK16_LORA_MATRICES_TRAINABLE")
    initial = {n: p.detach().cpu().clone() for n, p in parameters.items()}
    references = {"focused69_source": reference["source_final_loss"], "focused69_practice": reference["practice_final_loss"],
        "previous40_source": reference["source_baseline_loss"], "previous40_practice": reference["practice_baseline_loss"]}
    receipt = {"mode": SCHEMA, "source_commit": os.environ["TRAINING_SOURCE_SHA"], "model_repo": MODEL, "model_revision": REVISION,
        "official_weights_sha256": official, "source_audit": source["audit"], "initial_adapter_sha256": INITIAL_SHA,
        "previous40_adapter_sha256": PREVIOUS40_SHA, "world_size": world,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(2)], "trainable_parameters": 33030144,
        "training_rows": len(training), "historical_unique_train_cases": len(market_tokens), "authored_reasoned_rows": len(reasoned),
        "original_source_train_windows": len(source_tokens), "legacy_train_rows": len(legacy_tokens),
        "historical_audit": payload["audit"], "maximum_optimizer_attempts": 768, "maximum_training_seconds": 3600,
        "learning_rate": 8e-6, "development_generation_protocol": PROTOCOL, "checkpoints": [],
        "references": references, "expert_approved_market_training": False, "execution_allowed": False,
        "trading_qualification": False, "complete_training": False, "complete_test": False}
    def write(name, value):
        if rank == 0: (output/(name+".json")).write_bytes(canonical(value))
    def save_receipt(): write("research_receipt", receipt)
    def gather(local):
        parts = [None]*world; dist.all_gather_object(parts, local)
        return [r for part in parts for r in part]
    def stopped():
        flag = torch.tensor(int(watcher.requested.is_set()), device=device); dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())
    class Deadline(StoppingCriteria):
        def __init__(self, when): self.when = when
        def __call__(self, input_ids, scores, **kwargs): return watcher.requested.is_set() or time.monotonic() >= self.when
    def predict(questions, *, mode="direct", stage="candidate"):
        was_training = model.training; cache = model.config.use_cache; model.eval(); model.config.use_cache = True
        local = []
        with torch.inference_mode():
            for i, row in enumerate(questions):
                if i % world != rank: continue
                ids = tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": row["prompt"]}],
                    tokenize=True, add_generation_prompt=True, enable_thinking=mode == "thinking")
                tensor = torch.tensor([ids], device=device)
                options = {k: v for k, v in PROTOCOL.items() if k not in {"mode", "seconds"}}
                seconds = PROTOCOL["seconds"]
                if mode == "thinking": options.update(max_new_tokens=1536, temperature=.6, top_p=.95); seconds = 60
                torch.manual_seed(int(sha256((row["case_id"]+mode).encode()).hexdigest()[:8], 16))
                until = time.monotonic()+seconds
                with (model.disable_adapter() if stage == "base" else nullcontext()):
                    generated = model.generate(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                        pad_token_id=tokenizer.eos_token_id, stopping_criteria=StoppingCriteriaList([Deadline(until)]), **options)[0, len(ids):].tolist()
                timed_out = time.monotonic() >= until
                complete = bool(generated and generated[-1] == tokenizer.eos_token_id)
                if mode == "thinking":
                    end_think = tokenizer.convert_tokens_to_ids("</think>")
                    if end_think not in generated: complete = False; generated = []
                    else: generated = generated[len(generated)-generated[::-1].index(end_think):]
                local.append({"case_id": row["case_id"], "response": tokenizer.decode(generated, skip_special_tokens=True).strip(),
                    "thinking_complete": complete, "response_time_limit_reached": timed_out})
        results = {r["case_id"]: r for r in gather(local)}
        if len(results) != len(questions): raise ValueError("NO_MISSING_MODEL_RESPONSES")
        model.config.use_cache = cache; model.train(was_training)
        return [results[q["case_id"]] for q in questions]
    def losses(groups, capture=None):
        was_training = model.training; model.eval(); local = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for edition, rows in groups.items():
                for i, row in enumerate(rows):
                    if i % world != rank: continue
                    batch = {k: torch.tensor([row[k]], device=device) for k in ("input_ids", "attention_mask", "labels")}
                    value = float(model(**batch, use_cache=False).loss)
                    if not math.isfinite(value): raise ValueError("FINITE_BOOK_WINDOW_LOSS_REQUIRED")
                    local.append([edition, i, value])
        values = gather(local); result = {e: sum(v for ed, _, v in values if ed == e)/len(groups[e]) for e in EDITIONS}
        result["ALL"] = sum(v[2] for v in values)/len(values)
        if capture: write(capture, {"window_counts": {e: len(groups[e]) for e in EDITIONS}, "window_losses": values})
        model.train(was_training); return result
    save_receipt()
    receipt["initial_historical_development"] = score(payload["validation"], predict(prompts_only(payload["validation"])))
    save_receipt()
    class Rows:
        def __init__(self, rows): self.rows = rows
        def __len__(self): return len(self.rows)
        def __getitem__(self, i): return self.rows[i]
    def collate(rows):
        width = max(len(r["input_ids"]) for r in rows)
        return {k: torch.tensor([r[k]+[fill]*(width-len(r[k])) for r in rows], dtype=torch.float32 if k == "loss_weights" else torch.long)
            for k, fill in (("input_ids", tokenizer.pad_token_id), ("attention_mask", 0), ("labels", -100), ("loss_weights", 0.))}
    class Updates(TrainerCallback):
        def __init__(self): self.actual = 0; self.skipped = 0; self.nonfinite = False; self.saved = {}
        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            gradients = [p.grad for p in parameters.values() if p.grad is not None]
            if not gradients: raise ValueError("REAL_LORA_GRADIENTS_REQUIRED")
            self.nonfinite = not bool(torch.stack([torch.isfinite(g).all() for g in gradients]).all())
        def on_optimizer_step(self, args, state, control, **kwargs):
            skipped = bool(trainer.accelerator.optimizer_step_was_skipped)
            flags = torch.tensor([int(skipped), int(skipped)], device=device)
            dist.all_reduce(flags[:1], op=dist.ReduceOp.MIN); dist.all_reduce(flags[1:], op=dist.ReduceOp.MAX)
            if flags[0] != flags[1] or (self.nonfinite and not skipped): raise ValueError("BOTH_RANKS_MUST_SKIP_NONFINITE_UPDATES")
            if skipped: self.skipped += 1
            else:
                self.actual += 1
                if not bool(torch.stack([torch.isfinite(p).all() for p in parameters.values()]).all()): raise ValueError("NONFINITE_TRAINED_ADAPTER")
        def on_step_end(self, args, state, control, **kwargs):
            flag = torch.tensor([int(time.monotonic()-started >= 3600), int(watcher.requested.is_set())], device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            if bool(flag.any()):
                control.should_training_stop = True; control.should_save = True; control.should_evaluate = not bool(flag[1])
                if flag[1]: args.load_best_model_at_end = False
        def on_save(self, args, state, control, **kwargs): self.saved[state.global_step] = self.actual
    class HistoricalTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            inputs = dict(inputs); labels = inputs.pop("labels"); weights = inputs.pop("loss_weights")
            result = model(**inputs, use_cache=False); loss = weighted_causal_loss(result.logits, labels, weights)
            if not bool(torch.isfinite(loss)): raise ValueError("FINITE_WEIGHTED_LOSS_REQUIRED")
            return (loss, result) if return_outputs else loss
        def evaluation_loop(self, *args, **kwargs):
            result = super().evaluation_loop(*args, **kwargs)
            if stopped(): return result._replace(metrics={**result.metrics, "eval_historical_score": -9.})
            answers = predict(prompts_only(payload["validation"])); grade = score(payload["validation"], answers)
            source_loss = losses(source_val, "source_windows_"+str(self.state.global_step))
            practice_loss = losses(practice_val, "practice_windows_"+str(self.state.global_step))
            chosen = selection(grade, source_loss, practice_loss, references)
            row = {"optimizer_step": self.state.global_step, "grade": grade, "source_loss": source_loss,
                "practice_loss": practice_loss, "selection": chosen}
            receipt["checkpoints"].append(row); save_receipt()
            write("development_"+str(self.state.global_step), answers)
            if rank == 0: print("HISTORICAL_DEVELOPMENT", json.dumps({"step": self.state.global_step, "correct": grade["correct"], **chosen}), flush=True)
            return result._replace(metrics={**result.metrics, "eval_historical_score": chosen["score"]})
    args = TrainingArguments(output_dir=str(output.parent/"private_historical_checkpoints"), max_steps=768,
        per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=1,
        learning_rate=8e-6, warmup_ratio=.03, weight_decay=.01, lr_scheduler_type="cosine", max_grad_norm=1.,
        optim="adamw_torch", eval_strategy="steps", eval_steps=256, save_strategy="steps", save_steps=256,
        save_total_limit=2, save_safetensors=True, load_best_model_at_end=True, metric_for_best_model="historical_score",
        greater_is_better=True, prediction_loss_only=True, logging_steps=32, logging_nan_inf_filter=False,
        report_to="none", disable_tqdm=True, remove_unused_columns=False, label_names=["labels"],
        fp16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_backend="nccl", ddp_find_unused_parameters=False, ddp_broadcast_buffers=False, ddp_timeout=1200,
        seed=20261029, data_seed=20261029)
    updates = Updates()
    trainer = HistoricalTrainer(model=model, args=args, train_dataset=Rows(training), eval_dataset=Rows(dev_tokens), data_collator=collate, callbacks=[updates])
    trainer.model_accepts_loss_kwargs = False
    if trainer.accelerator.scaler is None: raise ValueError("REAL_FP16_SCALER_REQUIRED")
    trainer.accelerator.scaler = torch.amp.GradScaler("cuda", init_scale=1024., growth_interval=2000)
    if not stopped(): trainer.train()
    receipt.update(actual_updates=updates.actual, optimizer_attempted_steps=trainer.state.global_step, amp_skipped_steps=updates.skipped,
        actual_epoch=trainer.state.epoch, selected_checkpoint=Path(trainer.state.best_model_checkpoint).name if trainer.state.best_model_checkpoint else None,
        selected_metric=trainer.state.best_metric, selected_actual_updates=updates.saved.get(int(Path(trainer.state.best_model_checkpoint).name.split("-")[-1])) if trainer.state.best_model_checkpoint else None)
    receipt["stopped_by_user_or_control_unavailable"] = stopped()
    receipt["adapter_weights_changed"] = any(not torch.equal(p.detach().cpu(), initial[n]) for n, p in parameters.items())
    if rank == 0:
        model.save_pretrained(output/"candidate_adapter", safe_serialization=True); tokenizer.save_pretrained(output/"candidate_adapter")
    dist.barrier(); receipt["candidate_adapter_sha256"] = sha256((output/"candidate_adapter/adapter_model.safetensors").read_bytes()).hexdigest()
    receipt["complete_training"] = receipt["adapter_weights_changed"] and not receipt["stopped_by_user_or_control_unavailable"]
    save_receipt()
    if not receipt["complete_training"]:
        watcher.finished.set(); dist.destroy_process_group(); return receipt
    receipt["source_final_loss"] = losses(source_val, "source_validation_window_losses")
    receipt["practice_final_loss"] = losses(practice_val, "practice_validation_window_losses")
    receipt["text_gates"] = {name: validation_admission(before, receipt["source_final_loss"] if name.endswith("source") else receipt["practice_final_loss"]) for name, before in references.items()}
    receipt["candidate_frozen_before_test_file_read"] = True; save_receipt()
    model.gradient_checkpointing_disable(); model.eval()
    model.load_adapter(previous_path, adapter_name="previous40", is_trainable=False, local_files_only=True)
    model.load_adapter(initial_path, adapter_name="focused69", is_trainable=False, local_files_only=True)
    questions = json.loads(questions_path.read_bytes())
    if set(questions) != {"historical", "numeric", "sha256"} or any(digest(questions[k]) != questions["sha256"][k] for k in ("historical", "numeric")):
        raise ValueError("FROZEN_PROMPTS_ONLY_FINAL_EXAM_REQUIRED")
    results = {"receipt": {"source_commit": receipt["source_commit"], "candidate_adapter_sha256": receipt["candidate_adapter_sha256"],
        "protocol": PROTOCOL, "test_questions_sha256": questions["sha256"], "expected_keys_loaded_on_gpu": False},
        "historical": {}, "numeric": {}, "thinking_diagnostic": {}}
    for stage in STAGES:
        model.set_adapter("default" if stage in {"base", "candidate"} else stage)
        for cohort in ("historical", "numeric"):
            results[cohort][stage] = predict(questions[cohort], stage=stage); write("final_predictions", results)
        if stage in {"base", "focused69", "candidate"}:
            # Fixed diagnostic only: never select weights or rewrite final answers.
            small = [q for family in ("TC1.PSY.PLAN", "NTD.AC") for q in [q for q in questions["historical"] if q["domain"] == family][:4]]
            results["thinking_diagnostic"][stage] = predict(small, stage=stage, mode="thinking"); write("final_predictions", results)
    receipt["complete_test"] = True; receipt["training_seconds_through_final_exam"] = time.monotonic()-started
    save_receipt(); watcher.finished.set(); dist.barrier(); dist.destroy_process_group()
    return receipt
