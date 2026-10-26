"""GitHub->Kaggle bridge. Generates private metadata/worker, NEVER uploads books.

The caller chooses independently-approved Kaggle input IDs and paths. The
generated worker fetches only public Williams-AI code, not credentials.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

_RESOURCE=re.compile(r"^[a-z0-9][a-z0-9_-]{0,61}/[a-z0-9][a-z0-9_-]{0,61}$")
_MODEL_RESOURCE=re.compile(r"^[a-z0-9][a-z0-9_-]{0,61}/[a-z0-9][a-z0-9_-]{0,61}/[a-z0-9_-]+/[a-z0-9_-]+/[0-9]+$")
_SHA=re.compile(r"^[a-f0-9]{40}$")


def kaggle_input_path(value:str)->str:
    """Accept only a specified Kaggle input mount, never shell expressions."""
    if (not isinstance(value,str) or not value.startswith("/kaggle/input/")
        or len(value)>250 or any(x in value for x in ("\x00","\n","\r","\\"))
        or any(p in ("",".","..") for p in value.split("/")[3:])):
        raise ValueError("EXPLICIT_ABSOLUTE_KAGGLE_INPUT_REQUIRED")
    return value


def generate(*,kernel_id:str,dataset_source:str,model_source:str,
             dataset_file:str,model_directory:str,revision:str,
             resume_config:dict|None=None)->tuple[dict,str]:
    if not isinstance(kernel_id,str) or not _RESOURCE.fullmatch(kernel_id):
        raise ValueError("KAGGLE_OWNER_AND_KERNEL_SLUG_REQUIRED")
    if not isinstance(dataset_source,str) or not _RESOURCE.fullmatch(dataset_source):
        raise ValueError("KAGGLE_PRIVATE_DATASET_ID_REQUIRED")
    if not isinstance(model_source,str) or not _MODEL_RESOURCE.fullmatch(model_source):
        raise ValueError("KAGGLE_MODEL_SOURCE_REVISION_REQUIRED")
    if not isinstance(revision,str) or not _SHA.fullmatch(revision):
        raise ValueError("EXACT_GITHUB_COMMIT_SHA_REQUIRED")
    dataset_file=kaggle_input_path(dataset_file)
    model_directory=kaggle_input_path(model_directory)
    if not dataset_file.endswith(".jsonl") or dataset_file==model_directory:
        raise ValueError("REVIEWED_JSONL_AND_HF_MODEL_MUST_BE_DISTINCT")
    resume={} if resume_config is None else resume_config
    if not isinstance(resume,dict):
        raise ValueError("EXPLICIT_OWN_RESUME_CONFIG_REQUIRED")
    if resume:
        if (set(resume)!={"dataset_source","directory","trusted_own"}
            or resume["trusted_own"] is not True
            or not isinstance(resume["dataset_source"],str)
            or not _RESOURCE.fullmatch(resume["dataset_source"])):
            raise ValueError("EXPLICIT_OWN_RESUME_CONFIG_REQUIRED")
        kaggle_input_path(resume["directory"])
    sources=list(dict.fromkeys([dataset_source]+([resume["dataset_source"]] if resume else [])))
    metadata={
        # Kaggle derives the slug from title; preserve the requested identity.
        "id":kernel_id,"title":kernel_id.split("/")[1],
        "code_file":"worker.py","language":"python","kernel_type":"script",
        "is_private":"true","enable_gpu":"true","enable_internet":"true",
        "machine_shape":"NvidiaTeslaT4",
        "dataset_sources":sources,"model_sources":[model_source],
        "competition_sources":[],"kernel_sources":[],
    }
    settings={"dataset":dataset_file,"hf_base":model_directory,"revision":revision,
              "resume":resume}
    # Embedding JSON as a Python string literal prevents arbitrary input from
    # becoming code or shell arguments. No API token or book text is embedded.
    literal=repr(json.dumps(settings,sort_keys=True))
    worker='''"""Generated private Kaggle QLoRA worker; not automatically runnable in GitHub."""
import json
import os
import subprocess
import sys
from pathlib import Path

CONFIG=json.loads(CONFIG_PLACEHOLDER)
# This trainer is deliberately single-GPU. T4 x2 must not silently change
# effective batch size or introduce DataParallel on a 4-bit device-mapped model.
os.environ["CUDA_VISIBLE_DEVICES"]="0"
data=Path(CONFIG["dataset"])
base=Path(CONFIG["hf_base"])
if not data.is_file() or not base.is_dir():
    raise RuntimeError("APPROVED_KAGGLE_INPUTS_NOT_ATTACHED")
checkout=Path("/kaggle/working/williams_source")
if checkout.exists():
    raise RuntimeError("ISOLATED_SOURCE_DIR_MUST_BE_NEW")
subprocess.run(["git","clone","https://github.com/19wolfalone92/Williams-AI.git",
                str(checkout)],check=True)
subprocess.run(["git","checkout",CONFIG["revision"]],cwd=checkout,check=True)
# Keep Kaggle's CUDA-enabled torch; install the pinned userspace training stack.
subprocess.run([sys.executable,"-m","pip","install","--disable-pip-version-check",
    "transformers==4.57.1","peft==0.17.1","accelerate==1.10.1",
    "safetensors==0.6.2","bitsandbytes==0.48.1"],check=True)
command=[sys.executable,"-m","williams_ai.qlora_gpu_research",
    "--reviewed-jsonl",str(data),"--hf-base",str(base),
    "--new-output","/kaggle/working/williams-qlora-session",
    "--source-revision",CONFIG["revision"],
    "--max-steps","200","--session-steps","10",
    "--max-tokens","1024","--rank","8",
    "--source-only-research","--authorize-cloud-data-processing"]
if CONFIG["resume"]:
    command.extend(["--resume-directory",CONFIG["resume"]["directory"],"--trusted-own-resume"])
subprocess.run(command,cwd=checkout,check=True)
'''.replace("CONFIG_PLACEHOLDER",literal)
    return metadata,worker


def main()->None:
    p=argparse.ArgumentParser(description="Build PRIVATE Kaggle GPU kernel (no upload)")
    for key in ("kernel-id","dataset-source","model-source","dataset-file",
                "model-directory","revision"):
        p.add_argument("--"+key,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--resume-config",type=json.loads,default={})
    a=p.parse_args()
    meta,worker=generate(kernel_id=a.kernel_id,dataset_source=a.dataset_source,
        model_source=a.model_source,dataset_file=a.dataset_file,
        model_directory=a.model_directory,revision=a.revision,resume_config=a.resume_config)
    if a.output.exists() or a.output.is_symlink():
        raise ValueError("NEW_OUTPUT_DIR_REQUIRED")
    a.output.mkdir(parents=True,mode=0o700)
    (a.output/"kernel-metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    (a.output/"worker.py").write_text(worker,encoding="utf-8")
    # Only metadata and worker; do NOT copy any JSONL, HF weights or credentials.
    print("PRIVATE_KAGGLE_METADATA_AND_WORKER_PREPARED_NO_TRAINING_OR_DATA_UPLOAD")


if __name__=="__main__":main()
