"""Single-submit guard for private v7.1 T4 diagnostic kernel."""
from __future__ import annotations
import json,os,re,subprocess,tempfile
from pathlib import Path
from scripts.ai_v6_hist100_kaggle_gpu_submit import make_metadata,TRAINING_NAME
from williams_ai.v71_guided_json_diagnostics import ENFORCER_COMMIT
SLUG="williams-v71-patched-lmfe-diagnostics-eight-h1-20261115"
def need(ok,why):
 if not ok:raise ValueError("V71_SUBMIT_"+why)
def main():
 from kaggle.api.kaggle_api_extended import KaggleApi
 from kagglehub.auth import get_username
 from kagglesdk.kernels.types.kernels_api_service import ApiGetAcceleratorQuotaStatisticsRequest
 sha=os.environ.get("GITHUB_SHA","");token=os.environ.get("KAGGLE_API_TOKEN","")
 need(re.fullmatch("[a-f0-9]{40}",sha) is not None and len(token)>=20,"CREDENTIAL_OR_SHA")
 api=KaggleApi();api.authenticate();owner=get_username()
 need("COMPLETE" in str(api.kernels_status(owner+"/"+TRAINING_NAME).status).upper(),"TRAINED_ADAPTER_SOURCE_NOT_READY")
 meta,_=make_metadata(owner,sha);meta["id"]=owner+"/"+SLUG;meta["title"]=SLUG
 need(meta["is_private"]=="true" and meta["machine_shape"]=="NvidiaTeslaT4" and meta["kernel_sources"]==[owner+"/"+TRAINING_NAME] and meta["enable_gpu"]=="true","KERNEL_POLICY")
 try:prior=api.kernels_status(meta["id"])
 except Exception as exc:
  message=str(exc).lower()
  # Kaggle returns 403 rather than 404 for an uncreated private kernel slug.
  # The authenticated owner and access to the source kernel were verified above.
  if "404" in message or "not found" in message or (
      "403" in message and "kernels.get" in message and
      owner.lower()==meta["id"].split("/",1)[0].lower()):
   prior=None
  else:raise
 root=Path(os.environ["RUNNER_TEMP"])
 if prior is not None:
  result={"schema":"V71_EXISTING_KERNEL_REUSED_NOT_RESULTS","status":str(prior.status),"new_kernel_submitted":False,"training_started":False}
  (root/"v71_submission.json").write_text(json.dumps(result,indent=2));print(json.dumps(result));return
 with api.build_kaggle_client() as c:q=c.kernels.kernels_api_client.get_accelerator_quota_statistics(ApiGetAcceleratorQuotaStatisticsRequest()).gpu_quota
 seconds=(q.total_time_allowed-q.time_used).total_seconds() if q else None
 need(seconds is not None and seconds>7200,"GPU_QUOTA_LT_2H")
 kernel=f'''import subprocess,sys,os,torch
from pathlib import Path
assert torch.cuda.is_available()
sha="{sha}";root=Path("/kaggle/working/williams_v71")
subprocess.run(["git","clone","--quiet","https://github.com/19wolfalone92/Williams-AI.git",str(root)],check=True,timeout=180)
subprocess.run(["git","checkout","--quiet",sha],cwd=root,check=True)
subprocess.run([sys.executable,"-m","pip","install","--disable-pip-version-check","transformers==4.57.1","peft==0.17.1","accelerate==1.10.1","safetensors==0.6.2","bitsandbytes==0.48.1","huggingface_hub>=0.30,<2","jinja2>=3.1.6,<4"],check=True,timeout=600)
subprocess.run([sys.executable,"-m","pip","install","--no-deps","git+https://github.com/noamgat/lm-format-enforcer.git@{ENFORCER_COMMIT}"],check=True,timeout=300)
subprocess.run([sys.executable,"-m","scripts.ai_v71_guided_json_8_h1_worker","--plan","research/williams_v6_historical100_selection_20261114.json","--out-dir","/kaggle/working/v71_diagnostics"],cwd=root,env={{**os.environ,"PYTHONPATH":str(root),"HF_HUB_DISABLE_TELEMETRY":"1","TOKENIZERS_PARALLELISM":"false"}},check=True,timeout=6400)
'''
 need(token not in kernel and "KAGGLE_API_TOKEN" not in kernel,"TOKEN_LEAK")
 with tempfile.TemporaryDirectory(prefix="v71-kaggle-") as d:
  q=Path(d);(q/"worker.py").write_text(kernel);(q/"kernel-metadata.json").write_text(json.dumps(meta))
  result=subprocess.run(["kaggle","kernels","push","-p",d,"--accelerator","NvidiaTeslaT4","--timeout","7200"],capture_output=True,text=True,timeout=180)
  need(result.returncode==0,"KERNEL_PUSH_FAILED")
 status=str(api.kernels_status(meta["id"]).status).upper()
 result={"schema":"V71_T4_SUBMISSION_NOT_RESULTS","kernel":meta["id"],"status":status,"new_kernel_submitted":True,"enforcer_commit":ENFORCER_COMMIT,"training_started":False,"trading_allowed":False}
 (root/"v71_submission.json").write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=="__main__":main()
