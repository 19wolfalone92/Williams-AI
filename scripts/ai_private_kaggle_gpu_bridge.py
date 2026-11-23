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


def _verified_input_mount(path:str, resource:str, code:str)->str:
    """A declared Kaggle dataset/model must own the exact worker input path.

    Mere membership of /kaggle/input/ is insufficient: another attached
    private dataset or model can silently change the trained data provenance.
    """
    slug=resource.split("/")[1]
    mount="/kaggle/input/"+slug
    if not path.startswith(mount+"/"):
        raise ValueError(code)
    return mount


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
    dataset_mount=_verified_input_mount(dataset_file,dataset_source,
                                       "DATASET_INPUT_DOES_NOT_MATCH_DECLARED_SOURCE")
    model_mount=_verified_input_mount(model_directory,model_source,
                                     "MODEL_INPUT_DOES_NOT_MATCH_DECLARED_SOURCE")
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
        resume_directory=kaggle_input_path(resume["directory"])
        resume_mount=_verified_input_mount(resume_directory,resume["dataset_source"],
                                           "RESUME_INPUT_DOES_NOT_MATCH_DECLARED_SOURCE")
    else:
        resume_mount=None
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
              "dataset_mount":dataset_mount,"model_mount":model_mount,
              "resume":resume,"resume_mount":resume_mount}
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
def source_mount(raw, code):
    """Reject symbolic aliases of other attached Kaggle sources."""
    mount=Path(raw)
    if (not mount.is_dir() or mount.is_symlink() or
        mount.resolve()!=mount.absolute()):
        raise RuntimeError(code)
    return mount
dataset_mount=source_mount(CONFIG["dataset_mount"],
                           "APPROVED_KAGGLE_INPUTS_NOT_ATTACHED")
model_mount=source_mount(CONFIG["model_mount"],
                         "APPROVED_KAGGLE_INPUTS_NOT_ATTACHED")
if (not data.is_file() or not base.is_dir() or
    data.is_symlink() or base.is_symlink() or
    not data.resolve().is_relative_to(dataset_mount) or
    not base.resolve().is_relative_to(model_mount)):
    raise RuntimeError("APPROVED_KAGGLE_INPUTS_NOT_ATTACHED")
if CONFIG["resume"]:
    old=Path(CONFIG["resume"]["directory"])
    resume_mount=source_mount(CONFIG["resume_mount"],
                              "RESUME_INPUT_DOES_NOT_MATCH_DECLARED_SOURCE")
    if (not old.is_dir() or old.is_symlink() or
        not old.resolve().is_relative_to(resume_mount)):
        raise RuntimeError("RESUME_INPUT_DOES_NOT_MATCH_DECLARED_SOURCE")
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
