"""Genuine Qwen3-4B PUBLIC-source 4-book LoRA blind H1 market probe.

Admits verified, locally extracted adapter ONLY. Unseen PRICE outcomes must
never be given to this process. Expert wave/WM labels do not exist yet.
This is research measurement only: never enable trading based on this score.
"""
from __future__ import annotations
from contextlib import nullcontext
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import zipfile

from .qwen4b_blind_evaluation import (
    CLASSES, candidate_first_token_ids, market_prompt, verified_blind_questions,
)

TRAINING_MODE="REAL_QWEN3_4B_NVIDIA_GPU_NF4_PUBLIC_PRACTICE_RESEARCH_ONLY"
BASE_MODEL_REPO="Qwen/Qwen3-4B"
BASE_REVISION="3fb356690bbdae7353c94efb3eecdce276e6ed1c"
TRAINING_COMMIT="b85af751aff558cc24f84fbd49d4602514430cf5"


def checked_public_adapter_zip(archive: Path) -> dict:
    """Real PEFT weight SHA and research validation. No licensed book originals."""
    archive=Path(archive)
    if (archive.is_symlink() or not archive.is_file()
            or not 100000<=archive.stat().st_size<=100_000_000):
        raise ValueError("EXPECTED_LOCAL_BOUNDED_RESEARCH_ADAPTER_ARCHIVE")
    with zipfile.ZipFile(archive) as z:
        members=z.namelist()
        if (len(members)!=len(set(members))
                or any(n.startswith("/") or ".." in Path(n).parts or "\\" in n for n in members)
                or not {"research_receipt.json","adapter/adapter_config.json",
                        "adapter/adapter_model.safetensors"} <= set(members)):
            raise ValueError("SAFE_PUBLIC_ADAPTER_ARCHIVE_STRUCTURE_REQUIRED")
        receipt=json.loads(z.read("research_receipt.json"))
        weight=z.read("adapter/adapter_model.safetensors")
        config=json.loads(z.read("adapter/adapter_config.json"))
        if (not isinstance(receipt,dict)
                or receipt.get("mode")!=TRAINING_MODE
                or receipt.get("model_repo")!=BASE_MODEL_REPO
                or receipt.get("model_revision")!=BASE_REVISION
                or receipt.get("source_commit")!=TRAINING_COMMIT
                or receipt.get("rank")!=8
                or not 1<=receipt.get("steps",0)<=32
                or receipt.get("trained_official_pretrained_qwen3_4b") is not True
                or receipt.get("adapter_weights_changed") is not True
                or receipt.get("validation_gate",{}).get("accepted") is not True
                or receipt.get("full_original_four_books_trained") is not False
                or receipt.get("execution_allowed") is not False
                or receipt.get("adapter_model_sha256")!=sha256(weight).hexdigest()
                or not 100000<=len(weight)<=50_000_000
                or config.get("r")!=8
                or config.get("base_model_name_or_path")!=BASE_MODEL_REPO):
            raise ValueError("PUBLIC_4B_ADAPTER_GENUINE_SOURCE_VALIDATION_REQUIRED")
        return {"adapter_sha256":receipt["adapter_model_sha256"],
                "source_commit":TRAINING_COMMIT,
                "steps":receipt["steps"],"rank":8,
                "validation_relative_improvement":
                    receipt["validation_gate"]["relative_improvement"],
                "original_private_books_trained":False,
                "independently_expert_reviewed_market_charts":False,
                "execution_allowed":False}


def predict_real_public_before_after(*,questions: Path,source_adapter_zip: Path,
                                     new_output: Path)->dict:
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
    from peft import PeftModel
    new_output=Path(new_output)
    if new_output.exists() or new_output.is_symlink():
        raise ValueError("PRECOMMIT_OUTPUT_MUST_BE_NEW")
    rows,question_sha=verified_blind_questions(questions)
    proof=checked_public_adapter_zip(source_adapter_zip)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError("NVIDIA_GPU_REQUIRED_REAL_QWEN4B_BLIND_PROBE")
    with tempfile.TemporaryDirectory(prefix="williams-public4b-adapter-") as temp:
        with zipfile.ZipFile(source_adapter_zip) as z:
            for name in z.namelist():
                if name.startswith("adapter/") and not name.endswith("/"):
                    target=Path(temp)/name
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with target.open("xb") as f:
                        f.write(z.read(name))
        adapter=Path(temp)/"adapter"
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype)
        tok=AutoTokenizer.from_pretrained(BASE_MODEL_REPO,
            revision=BASE_REVISION,trust_remote_code=False)
        base=AutoModelForCausalLM.from_pretrained(BASE_MODEL_REPO,
            revision=BASE_REVISION,quantization_config=quant,
            dtype=dtype,device_map={"":0},trust_remote_code=False)
        model=PeftModel.from_pretrained(base,str(adapter),is_trainable=False,
                                      local_files_only=True)
        model.eval()
        prepared=[]
        for row in rows:
            conversation=[
                {"role":"system","content":"Ты — исследователь рыночных гипотез без права исполнять ордера."},
                {"role":"user","content":market_prompt(row)},
            ]
            prefix,target_ids=candidate_first_token_ids(tok,conversation)
            if len(prefix)>3072:
                raise ValueError("MARKET_BLIND_QWEN_PROMPT_EXCEEDS_FIXED_CONTEXT")
            prepared.append((row["case_id"],prefix,target_ids))
        preds={}
        device=next(model.parameters()).device
        with torch.inference_mode():
            for label in ("baseline","adapted"):
                context=model.disable_adapter() if label=="baseline" else nullcontext()
                values=[]
                with context:
                    for case_id,prefix,ids in prepared:
                        input_ids=torch.tensor([prefix],dtype=torch.long,device=device)
                        logits=model(input_ids=input_ids,use_cache=False).logits[0,-1]
                        logits=logits[ids].float()
                        if not torch.isfinite(logits).all().item():
                            raise ValueError("INVALID_QWEN4B_MARKET_LOGITS")
                        scores=torch.softmax(logits,dim=-1).detach().cpu().tolist()
                        values.append({"case_id":case_id,
                            "probabilities":dict(zip(CLASSES,map(float,scores)))})
                preds[label]=values
        if any(len(preds[k])!=30 for k in ("baseline","adapted")):
            raise RuntimeError("MISSING_PAIRED_30_BLIND_REAL_QWEN_CASES")
        receipt={
            "mode":"PRECOMMITTED_QWEN3_4B_REAL_H1_NEXT_TOKEN_COMPARISON",
            "market_questions_file_sha256":question_sha,
            "model_provenance":{"mode":"PUBLIC_SOURCE_ONLY_RESEARCH",
                                "base_repo":BASE_MODEL_REPO,
                                "base_revision":BASE_REVISION,**proof},
            "cases_each_model":30,
            "scoring":"3_CANDIDATE_FIRST_TOKENS_NOT_CALIBRATED_PROBABILITIES",
            "true_future_outcomes_loaded_in_prediction":False,
            "trading_orders_created":False,
            "independent_expert_wave_labels":False,
            "research_only":True,"execution_allowed":False,
        }
        result={"receipt":receipt,**preds}
        new_output.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        with new_output.open("x",encoding="utf-8") as f:
            json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
        new_output.chmod(0o600)
        print("REAL_QWEN4B_PAIRED_FUTURE_BLIND_PUBLIC_SOURCE_PREDICTIONS_SAVED",
              len(preds["adapted"]),flush=True)
        return receipt
