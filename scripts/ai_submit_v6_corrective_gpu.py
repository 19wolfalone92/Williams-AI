"""Submit a new private corrective notebook using existing authenticated Kaggle machinery."""
import hashlib,io,json,os,re,subprocess,tempfile,time
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from scripts import ai_submit_v6_objective_smoke as prior
from scripts.ai_submit_v5_qlora_kaggle import _seal_data
SLUG="williams-ai-v6-corrective-a-20261126-g1"
SOURCE="williams-ai-v6-objective-smoke-20261124"
TIMEOUT=5400

def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglehub.auth import get_username
    from kagglesdk.kernels.types.kernels_api_service import ApiGetAcceleratorQuotaStatisticsRequest
    token=os.environ.get("KAGGLE_API_TOKEN","");sha=os.environ["GITHUB_SHA"]
    if len(token)<20 or not re.fullmatch("[0-9a-f]{40}",sha):raise ValueError("PRIVATE_CREDENTIAL_AND_SHA_REQUIRED")
    prior.SLUG=SLUG
    prior.CONTROL="https://raw.githubusercontent.com/19wolfalone92/Williams-AI/research/williams-ai-source-sft-v6-20261124/experiments/"+SLUG+".json"
    prior.check_control()
    api=KaggleApi();api.authenticate();owner=get_username()
    if not owner:raise ValueError("UNVERIFIED_OWNER")
    kernel=owner+"/"+SLUG;source=owner+"/"+SOURCE
    def quota():
        with api.build_kaggle_client() as client:
            q=client.kernels.kernels_api_client.get_accelerator_quota_statistics(ApiGetAcceleratorQuotaStatisticsRequest()).gpu_quota
            return (q.total_time_allowed-q.time_used).total_seconds() if q else None
    remaining=quota()
    print("CORRECTIVE_FREE_GPU_QUOTA_SECONDS",remaining,flush=True)
    if remaining is None or remaining<TIMEOUT+600:raise ValueError("INSUFFICIENT_FREE_GPU_QUOTA")
    for slug in (SOURCE,"williams-ai-v6-objective-smoke-20261125-a1"):
        status=str(api.kernels_status(owner+"/"+slug).status).upper()
        if not any(s in status for s in ("COMPLETE","ERROR","CANCEL")):raise ValueError("PREVIOUS_GPU_JOB_NOT_TERMINAL")
    if not prior.status_absent(api,kernel):raise ValueError("NEW_KERNEL_ALREADY_EXISTS")
    worker=prior.worker_source(sha).replace("scripts.ai_v6_objective_gpu_smoke","scripts.ai_v6_corrective_gpu").replace("/kaggle/working/williams_v6_objective_smoke","/kaggle/working/williams_corrective_a")
    if "scripts.ai_v6_corrective_gpu" not in worker or token in worker:raise ValueError("WORKER_CONTRACT")
    with tempfile.TemporaryDirectory() as d:
        folder=Path(d);(folder/"worker.py").write_text(worker)
        (folder/"kernel-metadata.json").write_text(json.dumps({"id":kernel,"title":SLUG,"code_file":"worker.py","language":"python","kernel_type":"script","is_private":"true","enable_gpu":"true","enable_internet":"true","machine_shape":"NvidiaTeslaT4","dataset_sources":[],"model_sources":[],"competition_sources":[],"kernel_sources":[source]}))
        p=subprocess.run(["kaggle","kernels","push","-p",d,"--accelerator","NvidiaTeslaT4","--timeout",str(TIMEOUT)],capture_output=True,text=True,timeout=180)
        if p.returncode:
            print("CORRECTIVE_SUBMIT_FAILED",p.returncode,flush=True)
            raise RuntimeError("PRIVATE_CORRECTIVE_SUBMISSION_FAILED")
    print("CORRECTIVE_GPU_SUBMITTED",SLUG,sha,flush=True)
    status="UNKNOWN"
    for i in range(300):
        current=str(api.kernels_status(kernel).status).upper()
        if current!=status or i%6==0:print("CORRECTIVE_GPU_STATE",current,flush=True)
        status=current
        if any(s in status for s in ("COMPLETE","ERROR","CANCEL")):break
        time.sleep(20)
    if not any(s in status for s in ("COMPLETE","ERROR","CANCEL")):raise RuntimeError("GPU_STILL_RUNNING_NO_COMPLETION_CLAIM")
    out=Path(os.environ["RUNNER_TEMP"])/"williams-corrective-sealed";out.mkdir(mode=0o700)
    with tempfile.TemporaryDirectory() as d:
        p=subprocess.run(["kaggle","kernels","output",kernel,"-p",d],capture_output=True,text=True,timeout=600)
        root=Path(d)/"williams_corrective_a"
        if not root.is_dir():raise RuntimeError("NO_PRIVATE_GPU_OUTPUT")
        buf=io.BytesIO()
        with ZipFile(buf,"w",compression=ZIP_DEFLATED,compresslevel=3) as z:
            for f in sorted(root.rglob("*")):
                if f.is_symlink():raise ValueError("SYMLINK_OUTPUT")
                if f.is_file():z.write(f,str(f.relative_to(root)))
        sealed=_seal_data(buf.getvalue(),token)
        (out/"candidate-a.zip.sealed").write_bytes(sealed)
        report={"kernel":SLUG,"status":status,"source_commit":sha,"sealed_sha256":hashlib.sha256(sealed).hexdigest(),"remaining_quota_seconds":quota(),"private_weights_encrypted":True}
        for name in ("diagnostic","baseline","aggregate"):
            f=root/(name+".json")
            if f.is_file():report[name]=json.loads(f.read_text())
        (out/"receipt.json").write_text(json.dumps(report,sort_keys=True,indent=2))
        print("CORRECTIVE_GPU_RECEIPT",json.dumps(report,sort_keys=True),flush=True)
        if "COMPLETE" not in status or "aggregate" not in report or p.returncode:raise RuntimeError("CORRECTIVE_INCOMPLETE_PARTIAL_ARCHIVE_PRESERVED")

if __name__=="__main__":main()
