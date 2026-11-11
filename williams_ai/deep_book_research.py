"""Two-stage private-original Qwen3-4B research; no trading activation.

Source language adaptation is followed by authored rule/numeric practice.
Source OCR is not expert chart supervision. A fixed market cohort is scored
only after prediction sealing, outside the GPU process. Original test pages
are deliberately absent. Keep all plaintext and trained weights outside Git.
"""
from __future__ import annotations

import base64
from contextlib import nullcontext
from decimal import Decimal, ROUND_FLOOR
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import time

EDITIONS = ("TC1", "NTD", "TC2", "COURSE")
MODEL = "Qwen/Qwen3-4B"
REVISION = "3fb356690bbdae7353c94efb3eecdce276e6ed1c"


def checked_payload(payload: dict) -> dict:
    from .book_rules import registry
    from .private_book_source import FIELDS
    if (set(payload) != {"format", "manifest", "files", "withheld_test_page_ids", "audit",
                         "test_texts_in_cloud_payload", "execution_allowed"}
            or payload["format"] != "PRIVATE_SOURCE_TRAIN_VALIDATION_ONLY_V1"
            or payload["test_texts_in_cloud_payload"] is not False
            or payload["execution_allowed"] is not False
            or set(payload["files"]) != {"train", "validation"}
            or set(payload["withheld_test_page_ids"]) != set(EDITIONS)):
        raise ValueError("SOURCE_ONLY_NO_TEST_TEXT_PAYLOAD_REQUIRED")
    manifest = payload["manifest"]
    books = {b["id"]: b for b in registry()["books"]}
    if (manifest.get("status") != "SOURCE_ONLY_NOT_EXPERT_APPROVED"
            or manifest.get("page_boundary_embargo") != 2
            or manifest.get("all_figures_understood") is not False
            or manifest.get("market_training_or_profitability_verified") is not False):
        raise ValueError("UNVERIFIED_EXTRACTION_MUST_REMAIN_EXPLICIT")
    result = {}
    ids = {s: {e: set() for e in EDITIONS} for s in ("train", "validation", "test")}
    for ed, pages in payload["withheld_test_page_ids"].items():
        if (not isinstance(pages, list) or any(type(p) is not int or not 1 <= p <= books[ed]["pages"] for p in pages)
                or len(set(pages)) != len(pages)
                or len(pages) != manifest["splits"]["test"]["books"][ed]):
            raise ValueError("FROZEN_TEST_PAGE_IDENTITIES_REQUIRED")
        ids["test"][ed] = set(pages)
    for split in ("train", "validation"):
        raw = base64.b64decode(payload["files"][split], validate=True)
        if len(raw) > 5_000_000 or sha256(raw).hexdigest() != manifest["splits"][split]["sha256"]:
            raise ValueError("FROZEN_SOURCE_SPLIT_SHA_MISMATCH")
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        for row in rows:
            if set(row) != FIELDS or row["edition"] not in books:
                raise ValueError("ORIGINAL_SOURCE_PAGE_FIELDS_REQUIRED")
            ed, page = row["edition"], row["pdf_page"]
            if (type(page) is not int or not 1 <= page <= books[ed]["pages"]
                    or page in ids[split][ed] or not isinstance(row["text"], str)
                    or not 30 <= len(row["text"]) <= 250000
                    or row["source_sha256"] != books[ed]["sha256"]
                    or manifest["book_registry"][ed] != {"sha256": books[ed]["sha256"], "pages": books[ed]["pages"]}
                    or row["extraction_status"] != ("UNVERIFIED_OCR" if ed == "COURSE" else "PDF_TEXT_NOT_EXPERT_REVIEWED")):
                raise ValueError("WRONG_OR_DUPLICATE_ORIGINAL_SOURCE_PAGE")
            ids[split][ed].add(page)
        if (len(rows) != manifest["splits"][split]["rows"]
                or {ed: len(p) for ed, p in ids[split].items()} != manifest["splits"][split]["books"]):
            raise ValueError("FROZEN_SOURCE_COUNTS_MISMATCH")
        result[split] = rows
    for ed in EDITIONS:
        for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
            if any(abs(a-b) <= 2 for a in ids[first][ed] for b in ids[second][ed]):
                raise ValueError("TWO_PAGE_SOURCE_HOLDOUT_EMBARGO_BROKEN")
    return {**result, "audit": {"source_page_counts": {s: {e: len(p) for e, p in eds.items()} for s, eds in ids.items()},
        "manifest_sha256": sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "original_pdf_sha256": {ed: books[ed]["sha256"] for ed in EDITIONS},
        "course_partial_and_ocr_unreviewed": True, "original_test_texts_sent_to_gpu": False,
        "two_page_embargo_verified": True, "expert_market_annotations": False}}


def practice_curriculum() -> tuple[list[dict], dict[str, list[dict]], dict]:
    from scripts.ai_real_qwen_source_lora_start import build_public_exercises
    from .book_rules import registry
    train, validation, old = build_public_exercises()
    held = {r["rule_id"] for group in validation.values() for r in group}
    train = list(train)
    for card in registry()["rule_cards"]:
        if card["status"] != "REVIEWED_PAGES" or card["id"] in held:
            continue
        for question in (f"Объясни правило «{card['title']}» и пределы его применения.",
                         f"Я хочу применить «{card['title']}» автоматически к любой похожей картинке. Что проверить?"):
            train.append({"edition": card["source"], "rule_id": card["id"], "pdf_page": card["pdf_pages"][0],
                          "question": question,
                          "answer": card["summary"] + " Это исследовательское объяснение; исполнение не разрешено."})
    # Independently calculated synthetic values; no claim of expert chart labels.
    # Numeric holdouts use disjoint values and a different question wording.
    for split, numbers in (("train", range(1, 97)), ("validation", range(151, 163))):
        for n in numbers:
            equity = Decimal(900 + n * 17)
            risk = equity * Decimal("0.005")
            entry = Decimal(100 + n) + Decimal("0.1")
            distance = Decimal(2 + n % 7) + Decimal("0.2")
            qty = (risk / distance / Decimal("0.01")).to_integral_value(rounding=ROUND_FLOOR) * Decimal("0.01")
            side = "LONG" if n % 2 else "SHORT"
            stop = entry - distance if side == "LONG" else entry + distance
            question = (f"Расчёт учебной позиции {side}: equity={equity} USDT, риск 0.5%, "
                        f"entry={entry}, stop={stop}, шаг количества=0.01. Стоимость пункта=1 USDT. "
                        "Укажи бюджет риска и количество с округлением вниз. Комиссии/проскальзывание пока не заданы.")
            if split == "validation":
                question = "Проверь арифметику без отправки ордера. " + question.replace("Расчёт учебной позиции", "Данные")
            row = {"edition": "TC1", "rule_id": "TC1.PSY.PLAN", "pdf_page": 83, "question": question,
                   "answer": f"Бюджет риска={risk} USDT. Расстояние до стопа={distance}. Количество={qty}. "
                   "Это верхняя граница до издержек; комиссии и проскальзывание уменьшают допустимый размер. "
                   "Фильтры Binance и подтверждённая защита обязательны. execution_allowed=false."}
            (train if split == "train" else validation["TC1"]).append(row)
            low = Decimal(40 + n); high = low + Decimal(5)
            bullish = bool(n % 2)
            entry = high + Decimal("0.1") if bullish else low - Decimal("0.1")
            stop = low - Decimal("0.1") if bullish else high + Decimal("0.1")
            row = {"edition": "TC2", "rule_id": "TC2.WM1.TRIGGER", "pdf_page": 124,
                   "question": f"Учебный WM1 {'бычий' if bullish else 'медвежий'}: закрытый бар low={low}, high={high}. "
                   "Направление закрытия и ангуляция проверены отдельно. tick=0.1. Назови стоп-вход и начальный защитный уровень; заявка ещё не исполнена.",
                   "answer": f"Стоп-вход={entry}; начальная защита за противоположной границей={stop}. "
                   "Вход появится только при прохождении триггера и соблюдении риск-ограничений; "
                   "наличие сигнала не является заполнением заявки. Защиту подтверждает биржа. Это числовой пример, не ордер."}
            (train if split == "train" else validation["TC2"]).append(row)
    rng = random.Random(20261027)
    rng.shuffle(train)
    questions = [r["question"] for r in train]
    val_questions = [r["question"] for rows in validation.values() for r in rows]
    if len(set(questions)) != len(questions) or set(questions) & set(val_questions):
        raise ValueError("PRACTICE_DUPLICATES_OR_SPLIT_LEAKAGE")
    return train, validation, {"train_examples": len(train), "validation_examples": len(val_questions),
        "original_public_rule_holdouts": old, "synthetic_arithmetic_not_human_chart_labels": True,
        "source_training_can_expose_same_concepts": True,
        "numeric_parameter_ranges_disjoint": True, "expert_market_understanding_proven": False}


def corrective_course_practice():
    """Authored causal examples after direct PDF-image review; not expert charts."""
    groups = {s: [] for s in ("train", "validation")}
    for split, values in (("train", range(1, 33)), ("validation", range(101, 109))):
        for n in values:
            start_end = 20 + n * 3
            overlap = n % 3 == 0
            signal_begin = start_end - 1 if overlap else start_end + 2
            up = n % 2 == 0
            extreme = 60 + n
            level = Decimal(extreme) + (Decimal("0.1") if up else Decimal("-0.1"))
            groups[split].append({"edition": "COURSE", "rule_id": "COURSE.HUMP", "pdf_page": 72,
                "question": f"Course Hump: первая крайняя формация заканчивается баром {start_end}, "
                f"вторая начинается баром {signal_begin}. Оба противоположных фрактала уже подтверждены. "
                f"Второй фрактал {'верхний' if up else 'нижний'}, его экстремум={extreme}; tick=0.1. "
                "Соответствует ли это Hump PDF 72 и существует ли учебный уровень входа?",
                "answer": ("Нет. Формации перекрываются барами; этот Hump-профиль не подтверждён. "
                    "Не создавать уровень как действительный Hump и не заменять его другим изданием." if overlap else
                    f"Критерий неперекрытия соблюдён. Первый фрактал — start, второй — signal. "
                    f"Учебный {'LONG' if up else 'SHORT'} стоп-триггер={level}. Это ещё не исполнение; "
                    "проверить momentum выбранного профиля, риск и позицию. execution_allowed=false.")})
            crossed = n % 2 == 0
            groups[split].append({"edition": "COURSE", "rule_id": "COURSE.W4.ZERO", "pdf_page": 44,
                "question": f"Предполагаемая старшая W3, последовательность {100+n%40} баров. "
                f"Закрытые значения 5/34: {round(1+n/100,2)}, {(-0.2 if crossed else 0.2)}. "
                "Младшая c/e ещё не подтверждена. Можно ли признать конец W4 и немедленно входить в W5?",
                "answer": ("Переход за ноль выполнил только минимальные требования W4 выбранной степени. " if crossed else
                    "Осциллятор лишь приблизился к нулю; переход за него не подтверждён. ") +
                    "Завершение коррекции не доказано. Проверить a–b–c или треугольник и последнюю младшую c/e. "
                    "Не выдавать минимум W4 за её окончание или приказ на вход."})
            hist = Decimal(n % 9) / 10 - Decimal("0.4") + Decimal(n) / 10000
            signal = hist + (Decimal("0.2") if up else Decimal("-0.2"))
            direction = "SHORT" if hist < signal else "LONG"
            groups[split].append({"edition": "COURSE", "rule_id": "COURSE.AO5345", "pdf_page": 45,
                "question": f"Course PDF 45: 5/34={hist}, его собственная SMA(5)={signal}. "
                "Определи текущий momentum профиля. Структура входа и риск не предоставлены.",
                "answer": f"Momentum={direction}: сравниваю гистограмму с её собственной SMA(5), а не со знаком нуля. "
                    "Это фильтр направления, не самостоятельная заявка. Для входа не хватает структуры, триггера и риска."})
            groups[split].append({"edition": "COURSE", "rule_id": "COURSE.THUMB", "pdf_page": 69,
                "question": f"Thumb SHORT, пример {n}: три счётных пальца имеют растущие HIGH и LOW; "
                "между ними есть один inside bar. Два пальца — GREEN, SQUAT нет. Сравни текст PDF 69 со сводкой PDF 71.",
                "answer": "Inside bar не является пальцем. По тексту PDF 69 два GREEN допускаются как сочетание SQUAT/GREEN. "
                    "Сводка PDF 71 описывает более строгий идеал с одним SQUAT и одним GREEN, без SQUAT он слабее. "
                    "Нужно назвать выбранный профиль; не скрывать это различие. Thumb не равен уже подтверждённому пятибарному фракталу."})
    if set(r["question"] for r in groups["train"]) & set(r["question"] for r in groups["validation"]):
        raise ValueError("CORRECTIVE_PRACTICE_PARAMETER_LEAKAGE")
    return groups


def predict_market(model, tokenizer, questions: Path, output: Path) -> dict:
    from .qwen4b_blind_evaluation import verified_blind_questions, market_prompt, candidate_first_token_ids, CLASSES
    import torch
    cases, digest = verified_blind_questions(questions)
    model.eval(); model.config.use_cache = True
    result = {"baseline": [], "adapted": []}
    with torch.inference_mode():
        for case in cases:
            messages = [{"role": "system", "content": "Исследуй только предоставленные закрытые данные. Не отправляй ордера."},
                        {"role": "user", "content": market_prompt(case)}]
            prefix, targets = candidate_first_token_ids(tokenizer, messages)
            if len(prefix) > 8192:
                raise ValueError("BLIND_PROMPT_TOO_LONG_NO_SILENT_TRUNCATION")
            ids = torch.tensor([prefix], device=next(model.parameters()).device)
            for mode in result:
                with (model.disable_adapter() if mode == "baseline" else nullcontext()):
                    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0, -1, targets].float()
                    if not torch.isfinite(logits).all().item():
                        raise ValueError("NONFINITE_BLIND_PREDICTION")
                    probs = torch.softmax(logits, -1).cpu().tolist()
                    result[mode].append({"case_id": case["case_id"], "probabilities": dict(zip(CLASSES, probs))})
    receipt = {"mode": "PRECOMMITTED_QWEN3_4B_REAL_H1_NEXT_TOKEN_COMPARISON",
               "market_questions_file_sha256": digest, "cases_each_model": len(cases),
               "true_future_outcomes_loaded_in_prediction": False, "execution_allowed": False,
               "scores_are_uncalibrated_class_token_probabilities": True,
               "expert_wave_labels": False, "source_revision": os.environ.get("TRAINING_SOURCE_SHA")}
    output.write_text(json.dumps({"receipt": receipt, **result}, indent=2, allow_nan=False))
    return receipt


def practice_exam(model, tokenizer, output: Path) -> dict:
    """Ten authored rule questions; concept exposure via source is disclosed."""
    import torch
    from .adaptive_coach import learning_cases
    from scripts.ai_real_qwen_source_lora_start import build_public_exercises
    _, held, _ = build_public_exercises()
    cases = learning_cases()
    rows = [r for values in held.values() for r in values]
    scores = {"baseline": [], "adapted": []}
    model.eval()
    with torch.inference_mode():
        for row in rows:
            matched = [c for c in cases if c["rule_id"] == row["rule_id"] and row["question"].startswith(c["question"])]
            if len(matched) != 1:
                raise ValueError("UNIQUE_FIXED_PRACTICE_ANSWER_KEY_REQUIRED")
            text = row["question"].split("\nВыбери и объясни")[0] + "\nОтветь только одной буквой A, B или C."
            messages = [{"role": "user", "content": text}]
            prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
            targets = []
            for label in "ABC":
                full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": label}], tokenize=True,
                    add_generation_prompt=False, enable_thinking=False)
                if full[:len(prefix)] != prefix:
                    raise ValueError("PRACTICE_CLASS_PREFIX_CHANGED")
                targets.append(full[len(prefix)])
            if len(set(targets)) != 3:
                raise ValueError("THREE_DISTINCT_PRACTICE_CLASS_TOKENS_REQUIRED")
            ids = torch.tensor([prefix], device=next(model.parameters()).device)
            for stage in scores:
                with (model.disable_adapter() if stage == "baseline" else nullcontext()):
                    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0, -1, targets].float()
                    chosen = "ABC"[logits.argmax().item()]
                    scores[stage].append({"rule_id": row["rule_id"], "choice": chosen,
                        "expected": matched[0]["correct"], "correct": chosen == matched[0]["correct"]})
    report = {"format": "TEN_RULE_CHOICE_EXAM_AFTER_ORIGINAL_SOURCE_TRAINING_V1", **scores,
        "correct": {s: sum(r["correct"] for r in rr) for s, rr in scores.items()},
        "source_text_can_expose_same_rules": True, "not_independent_market_competency": True,
        "execution_allowed": False}
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    return {"cases": len(rows), "correct": report["correct"], "source_text_can_expose_same_rules": True}


def train(payload_file: Path, questions: Path, output: Path, source_steps: int = 512, practice_steps: int = 96,
          initial_adapter: Path | None = None, course_focus: bool = False, corrective_practice: bool = False,
          balanced_structured_practice: bool = False, compact_validation: bool = False,
          initial_checkpoint_provenance: dict | None = None, transfer_questions: Path | None = None) -> dict:
    from .private_book_source import tokenize_source_pages
    from .private_lora import _tokenized
    from .lora_integrity import validation_admission
    if (not 64 <= source_steps <= 768 or not 16 <= practice_steps <= 160
            or output.exists() or output.is_symlink()):
        raise ValueError("BOUNDED_NEW_DEEP_RESEARCH_RUN_REQUIRED")
    research_started=time.monotonic()
    training_stop_at=research_started+3300  # Reserve final validation/inference time inside a 90min GPU session.
    source = checked_payload(json.loads(payload_file.read_bytes()))
    practice, practice_validation, practice_audit = practice_curriculum()
    if balanced_structured_practice:
        from .structured_rule_practice import balanced_recovery_curriculum
        practice,practice_validation,practice_audit=balanced_recovery_curriculum()
    elif corrective_practice:
        correction = corrective_course_practice()
        practice.extend(correction["train"])
        practice_validation["COURSE"].extend(correction["validation"])
        practice_audit.update({"corrective_course_train": len(correction["train"]),
            "corrective_course_validation": len(correction["validation"]),
            "train_examples": len(practice),
            "validation_examples": sum(map(len, practice_validation.values())),
            "corrected_rules_can_overlap_earlier_rule_exam": True,
            "training_not_a_new_independent_expert_wave_exam": True})
    import torch
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, TrainerCallback, EarlyStoppingCallback, set_seed
    from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("ONE_REAL_NVIDIA_CUDA_GPU_REQUIRED")
    set_seed(20261027)
    info = HfApi().model_info(MODEL, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise ValueError("OFFICIAL_IMMUTABLE_QWEN3_4B_REQUIRED")
    official = {x.rfilename: x.lfs.sha256 for x in info.siblings if x.rfilename.endswith(".safetensors") and x.lfs}
    if not official:
        raise ValueError("OFFICIAL_WEIGHT_SHA256_REQUIRED")
    base = Path(snapshot_download(MODEL, revision=REVISION, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"]))
    for name, digest in official.items():
        if sha256((base / name).read_bytes()).hexdigest() != digest:
            raise ValueError("DOWNLOADED_QWEN_BASE_WEIGHT_HASH_MISMATCH")
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=False, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    selected_source_rows = [r for r in source["train"] if not course_focus or r["edition"] == "COURSE"]
    source_train = tokenize_source_pages(selected_source_rows, tokenizer, max_tokens=384)
    source_val = {e: tokenize_source_pages([r for r in source["validation"] if r["edition"] == e], tokenizer, max_tokens=384) for e in EDITIONS}
    task_train = _tokenized(practice, tokenizer, max_tokens=768)
    task_val = {e: _tokenized(practice_validation[e], tokenizer, max_tokens=768) for e in EDITIONS}
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True, trust_remote_code=False,
        use_safetensors=True, dtype=dtype, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype))
    if base_model.config.model_type != "qwen3" or base_model.config.hidden_size != 2560 or base_model.config.num_hidden_layers != 36:
        raise ValueError("QWEN3_4B_ARCHITECTURE_REQUIRED")
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False})
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if initial_adapter is None:
        model = get_peft_model(base_model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=targets, bias="none", task_type=TaskType.CAUSAL_LM))
    else:
        config = PeftConfig.from_pretrained(initial_adapter, local_files_only=True)
        if config.r != 16 or config.lora_alpha != 32 or set(config.target_modules) != set(targets) or config.bias != "none":
            raise ValueError("OWN_COMPATIBLE_RANK16_ATTENTION_MLP_ADAPTER_REQUIRED")
        model = PeftModel.from_pretrained(base_model, initial_adapter, is_trainable=True, local_files_only=True)
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if not trainable or any(".lora_" not in n for n in trainable):
        raise ValueError("ONLY_REAL_LORA_PARAMETERS_MAY_UPDATE")
    initial = {n: p.detach().cpu().clone() for n, p in trainable.items()}
    output.mkdir(parents=True, mode=0o700)
    report = {"mode": "REAL_QWEN3_4B_PRIVATE_ORIGINALS_AND_AUTHORED_PRACTICE_RESEARCH_V1",
        "model_repo": MODEL, "model_revision": REVISION, "official_weights_sha256": official,
        "source_commit": os.environ.get("TRAINING_SOURCE_SHA"), "gpu": torch.cuda.get_device_name(0),
        "rank": 16, "targets": targets, "trainable_parameters": sum(p.numel() for p in trainable.values()),
        "source_audit": source["audit"], "practice_audit": practice_audit,
        "source_train_windows": len(source_train), "source_validation_windows": {e: len(v) for e, v in source_val.items()},
        "all_original_pages_and_figures_understood": False, "course_partial_and_unreviewed_ocr": True,
        "initialization": "PREVIOUS_OWN_RANK16_WEIGHTS_NEW_OPTIMIZER" if initial_adapter else "OFFICIAL_BASE_NEW_RANK16_ADAPTER",
        "initial_adapter_sha256": sha256((initial_adapter / "adapter_model.safetensors").read_bytes()).hexdigest() if initial_adapter else None,
        "optimizer_state_resumed": False, "course_focused_source_stage": course_focus,
        "selected_source_train_pages": len(selected_source_rows), "corrective_course_practice": corrective_practice,
        "initial_checkpoint_provenance":initial_checkpoint_provenance,
        "compact_validation":compact_validation,"training_wall_budget_seconds":3300,
        "trading_qualification": False, "execution_allowed": False, "trade_action": None}

    def write_report():
        (output / "research_receipt.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))

    class Rows:
        def __init__(self, rows): self.rows = rows
        def __len__(self): return len(self.rows)
        def __getitem__(self, index): return self.rows[index]

    def collate(batch):
        width = max(len(x["input_ids"]) for x in batch)
        return {key: torch.tensor([x[key] + [fill] * (width - len(x[key])) for x in batch])
                for key, fill in (("input_ids", tokenizer.pad_token_id), ("attention_mask", 0), ("labels", -100))}

    class Finite(TrainerCallback):
        def __init__(self,stage_seconds):self.stage_seconds=stage_seconds;self.stop_at=None
        def on_train_begin(self,args,state,control,**kwargs):
            self.stop_at=min(training_stop_at,time.monotonic()+self.stage_seconds)
        def on_step_end(self,args,state,control,**kwargs):
            if self.stop_at is not None and time.monotonic()>=self.stop_at:
                control.should_training_stop=True
                control.should_evaluate=True
                control.should_save=True
        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            gradients = [p.grad for p in trainable.values() if p.grad is not None]
            if not gradients or any(not torch.isfinite(g).all().item() for g in gradients):
                raise ValueError("NONFINITE_OR_MISSING_ADAPTER_GRADIENT")

    def make_trainer(name, rows, dev, steps, lr):
        model.config.use_cache = False
        model.train()
        arguments = TrainingArguments(output_dir=str(output.parent / ("private_checkpoints_" + name)),
            max_steps=steps, per_device_train_batch_size=1, per_device_eval_batch_size=1,
            gradient_accumulation_steps=2, learning_rate=lr, warmup_ratio=0.05, weight_decay=0.01,
            lr_scheduler_type="cosine", max_grad_norm=1.0, optim="adamw_torch", prediction_loss_only=True,
            eval_strategy="steps", eval_steps=32, save_strategy="steps", save_steps=32,
            save_total_limit=2, save_only_model=False, save_safetensors=True, load_best_model_at_end=True,
            metric_for_best_model="eval_loss", greater_is_better=False, logging_steps=8,
            logging_nan_inf_filter=False, report_to="none", disable_tqdm=True, remove_unused_columns=False,
            label_names=["labels"], seed=20261027, data_seed=20261027,
            bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
        # Spread the fixed validation budget across each source/lesson list so
        # typed answers are represented even when appended after legacy cases.
        dev_rows=[]
        for e in EDITIONS:
            n=min(12,len(dev[e]));indices=sorted({round(i*(len(dev[e])-1)/max(1,n-1)) for i in range(n)})
            dev_rows.extend(dev[e][i] for i in indices)
        return Trainer(model=model, args=arguments, train_dataset=Rows(rows), eval_dataset=Rows(dev_rows),
            data_collator=collate, callbacks=[Finite(900 if name=="originals" else 3300), EarlyStoppingCallback(early_stopping_patience=3)])

    def scores(trainer, groups):
        values = {e: float(trainer.evaluate(eval_dataset=Rows(groups[e]))["eval_loss"]) for e in EDITIONS}
        values["ALL"] = sum(values[e] * len(groups[e]) for e in EDITIONS) / sum(map(len, groups.values()))
        if any(not math.isfinite(x) for x in values.values()):
            raise ValueError("NONFINITE_FOUR_BOOK_EVALUATION")
        return values

    # A bounded 1.5 epochs; validation, never market labels, selects the checkpoint.
    source_budget = min(source_steps, math.ceil(len(source_train) * 1.5 / 2))
    source_budget = max(32, source_budget // 32 * 32)
    first = make_trainer("originals", source_train, source_val, source_budget, 6e-5)
    if compact_validation:
        with model.disable_adapter():
            report["official_base_source_loss"]=scores(first,source_val)
            report["official_base_practice_loss"]=scores(first,task_val)
    report["source_baseline_loss"] = scores(first, source_val)
    report["practice_baseline_loss"] = scores(first, task_val)
    write_report()
    first.train()
    report["source_optimizer_steps"] = first.state.global_step
    report["source_selected_checkpoint"] = Path(first.state.best_model_checkpoint).name
    if compact_validation:
        report["source_selected_dev_loss"]=first.state.best_metric
        report["source_full_validation_deferred_until_final"]=True
    else:
        report["source_selected_loss"] = scores(first, source_val)
        report["source_validation_gate"] = validation_admission(report["source_baseline_loss"], report["source_selected_loss"])
    report["all_source_train_windows_seen_at_least_once"] = first.state.global_step * 2 >= len(source_train)
    model.save_pretrained(output / "source_adapter", safe_serialization=True)
    tokenizer.save_pretrained(output / "source_adapter")
    report["source_adapter_sha256"]=sha256((output/"source_adapter/adapter_model.safetensors").read_bytes()).hexdigest()
    write_report()
    del first
    torch.cuda.empty_cache()
    second = make_trainer("practice", task_train, task_val, practice_steps, 3e-5)
    if not compact_validation:report["practice_before_second_stage_loss"] = scores(second, task_val)
    second.train()
    report["practice_optimizer_steps"] = second.state.global_step
    report["practice_selected_checkpoint"] = Path(second.state.best_model_checkpoint).name
    report["practice_final_loss"] = scores(second, task_val)
    report["source_final_loss"] = scores(second, source_val)
    report["final_source_validation_gate"] = validation_admission(report["source_baseline_loss"], report["source_final_loss"])
    report["final_practice_validation_gate"] = validation_admission(report["practice_baseline_loss"], report["practice_final_loss"])
    if compact_validation:
        report["official_base_to_final_source_gate"]=validation_admission(report["official_base_source_loss"],report["source_final_loss"])
        report["official_base_to_final_practice_gate"]=validation_admission(report["official_base_practice_loss"],report["practice_final_loss"])
    changed = any(not torch.equal(p.detach().cpu(), initial[n]) for n, p in trainable.items())
    if not changed or any(not torch.isfinite(p).all().item() for p in trainable.values()):
        raise ValueError("FINITE_ACTUALLY_CHANGED_ADAPTER_WEIGHTS_REQUIRED")
    report["adapter_weights_changed"] = True
    model.save_pretrained(output / "candidate_adapter", safe_serialization=True)
    tokenizer.save_pretrained(output / "candidate_adapter")
    report["candidate_adapter_sha256"] = sha256((output / "candidate_adapter/adapter_model.safetensors").read_bytes()).hexdigest()
    report["source_adapter_sha256"] = sha256((output / "source_adapter/adapter_model.safetensors").read_bytes()).hexdigest()
    report["candidate_accepted_for_further_research"] = (report["final_source_validation_gate"]["accepted"] and report["final_practice_validation_gate"]["accepted"])
    write_report()
    report["blind_prediction_receipt"] = predict_market(model, tokenizer, questions, output / "market_precommit.json")
    report["ten_rule_practice_exam"] = practice_exam(model, tokenizer, output / "practice_choice_exam.json")
    if transfer_questions is not None:
        if initial_adapter is None:raise ValueError("OWN_INITIAL_ADAPTER_REQUIRED_FOR_THREE_MODEL_EXAM")
        from .structured_transfer_exam import predict_loaded
        report["generated_transfer_receipt"]=predict_loaded(model,tokenizer,transfer_questions,initial_adapter,
            output/"three_model_transfer_precommit.json")
    write_report()
    return report
