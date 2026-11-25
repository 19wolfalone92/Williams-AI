"""Retrieve and encrypt diagnostics for a failed private Kaggle kernel; no GPU job is started."""
from __future__ import annotations
import argparse,hashlib,json,os,re,subprocess,sys
from pathlib import Path
from scripts.ai_submit_v5_qlora_kaggle import _seal_data

KERNEL="19wolfalone92/williams-ai-v6-objective-smoke-20261125-a1"

def redact(text,secrets):
    value=str(text)
    for secret in secrets:
        if secret:value=value.replace(secret,"[REDACTED]")
    value=re.sub(r"(?i)(KAGGLE_API_TOKEN|KAGGLE_KEY|KAGGLE_USERNAME)\s*[=:]\s*\S+",
                 r"\1=[REDACTED]",value)
    return value

def _extract_text(path):
    if path.suffix.lower()==".ipynb":
        try:
            nb=json.loads(path.read_text(encoding="utf-8"))
            chunks=[]
            for cell in nb.get("cells",[]):
                for out in cell.get("outputs",[]):
                    for key in ("text",):
                        v=out.get(key)
                        if isinstance(v,list):chunks.extend(str(x) for x in v)
                        elif isinstance(v,str):chunks.append(v)
                    data=out.get("data",{})
                    for key in ("text/plain","text/html"):
                        v=data.get(key)
                        if isinstance(v,list):chunks.extend(str(x) for x in v)
                        elif isinstance(v,str):chunks.append(v)
                    if out.get("ename") or out.get("evalue"):
                        chunks.append(str(out.get("ename",""))+": "+str(out.get("evalue","")))
                        chunks.extend(str(x) for x in out.get("traceback",[]))
            return "\n".join(chunks)
        except Exception as exc:return "IPYNB_PARSE_ERROR:"+type(exc).__name__
    try:return path.read_text(encoding="utf-8",errors="replace")
    except Exception:return ""

def _account_check(token):
    from kaggle.api.kaggle_api_extended import KaggleApi
    os.environ["KAGGLE_API_TOKEN"]=token
    api=KaggleApi();api.authenticate()
    quota=api.get_accelerator_quota_statistics()
    return {"authenticated":True,"quota":str(quota)}

def run(output_dir:Path,secret_env:str,secondary_env:str):
    primary=os.environ.get(secret_env,"")
    secondary=os.environ.get(secondary_env,"")
    if len(primary)<20:raise ValueError("PRIMARY_KAGGLE_TOKEN_MISSING")
    if output_dir.exists() or output_dir.is_symlink():raise ValueError("NEW_DIAGNOSTIC_DIR_REQUIRED")
    output_dir.mkdir(parents=True,exist_ok=False)
    accounts={"primary":_account_check(primary),
              "secondary":_account_check(secondary) if len(secondary)>=20 else {"authenticated":False,"state":"NOT_CONFIGURED"}}
    os.environ["KAGGLE_API_TOKEN"]=primary
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate()
    try:status=str(api.kernels_status(KERNEL))
    except Exception as exc:status="STATUS_QUERY_ERROR:"+type(exc).__name__+":"+str(exc)
    outdir=output_dir/"kernel-output"
    outdir.mkdir(mode=0o700)
    cmd=["kaggle","kernels","output",KERNEL,"-p",str(outdir)]
    proc=subprocess.run(cmd,capture_output=True,text=True,timeout=240,env=os.environ.copy())
    secrets=[primary,secondary]
    files=[];log_tail=[]
    for path in sorted(outdir.rglob("*")):
        if not path.is_file() or path.is_symlink():continue
        raw=path.read_bytes()
        files.append({"name":str(path.relative_to(outdir)),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()})
        text=redact(_extract_text(path),secrets)
        if text:
            log_tail.append({"name":path.name,"tail":text.splitlines()[-240:]})
    report={"schema":"WILLIAMS_AI_KAGGLE_KERNEL_DIAGNOSTIC_V1","kernel":KERNEL,"status":redact(status,secrets),
      "accounts":accounts,"output_command":{"exit_code":proc.returncode,"stdout":redact(proc.stdout,secrets)[-12000:],
      "stderr":redact(proc.stderr,secrets)[-12000:]},"files":files,"log_tail":log_tail,
      "training_claim":False,"new_gpu_run_started":False,"private_output_only":True}
    sealed=_seal_data(json.dumps(report,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode(),primary)
    dest=output_dir/"kaggle-kernel-diagnostic.json.sealed";dest.write_bytes(sealed)
    print("WILLIAMS_AI_KAGGLE_KERNEL_DIAGNOSTIC",json.dumps({"kernel":KERNEL,"status":report["status"],
      "output_files":len(files),"secondary_account":accounts["secondary"]["authenticated"],
      "encrypted_sha256":hashlib.sha256(sealed).hexdigest(),"plaintext_logs_emitted":False},sort_keys=True),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--secret-env",default="KAGGLE_API_TOKEN");p.add_argument("--secondary-env",default="KAGGLE_API_TOKEN_2")
    a=p.parse_args();run(a.output_dir,a.secret_env,a.secondary_env)
if __name__=="__main__":main()
