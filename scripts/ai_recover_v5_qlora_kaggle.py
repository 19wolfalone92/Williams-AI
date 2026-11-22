"""Recover and authenticate an ALREADY trained Kaggle Qwen3-4B LoRA.

No kernel submission, no QLoRA retrain, no trading. Frozen source SHA is the
original GPU run commit, not the later recovery commit.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import sys

# Direct execution sets sys.path[0] to scripts/, not repository root.
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from scripts.ai_submit_v5_qlora_kaggle import (
    KERNEL_SLUG, _verify_and_archive,
)

TRAINED_SOURCE_COMMIT="0914a2cd5a5b5e15dca481029371f055704abd86"


def run() -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglehub.auth import get_username

    token=os.environ.get("KAGGLE_API_TOKEN","")
    if len(token)<20:
        raise ValueError("PRIVATE_KAGGLE_SECRET_REQUIRED")
    api=KaggleApi()
    api.authenticate()
    kernel=get_username()+"/"+KERNEL_SLUG
    state=str(api.kernels_status(kernel).status).upper()
    print("V5_RECOVERY_PREVIOUS_KERNEL_STATE",state,flush=True)
    if "COMPLETE" not in state:
        raise RuntimeError("PREVIOUS_KAGGLE_TRAINING_NOT_COMPLETE_RECOVERY_BLOCKED")

    with tempfile.TemporaryDirectory() as temporary:
        fetched=subprocess.run(
            ["kaggle","kernels","output",kernel,"-p",temporary,"-o"],
            capture_output=True,text=True,timeout=600)
        if fetched.returncode:
            raise RuntimeError("EXISTING_KAGGLE_TRAINED_FILES_DOWNLOAD_FAILED")
        root=Path(temporary)
        receipt,weight,config=(
            root/"williams_v5_model"/"training_receipt.json",
            root/"williams_v5_model"/"adapter"/"adapter_model.safetensors",
            root/"williams_v5_model"/"adapter"/"adapter_config.json",
        )
        print("V5_RECOVERY_FINAL_FILE_PRESENCE",
              json.dumps({"receipt":receipt.is_file(),"weights":weight.is_file(),
                          "config":config.is_file(),"other_output_files_ignored":True}),
              flush=True)
        destination=Path(os.environ["RUNNER_TEMP"])/"williams_v5_encrypted"
        audit=_verify_and_archive(root,token,destination,TRAINED_SOURCE_COMMIT)
    print("ACTUAL_WILLIAMS_V5_TRAINING_COMPLETED_AND_SEALED",
          json.dumps({"steps":audit["optimizer_steps"],
                      "adapter_sha256":audit["adapter_sha256"],
                      "validation_gate":audit["validation_gate"],
                      "training_reused_no_new_GPU":True,
                      "trading_qualification":False},sort_keys=True),flush=True)


if __name__=="__main__":
    run()
