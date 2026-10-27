"""Read only our running private kernel; expose numeric progress, never text.

The log endpoint/authentication follows Kaggle/kaggle-cli commit
42f85b6db020f915e32eb35131975537785a8cf3. No kernel is submitted.
"""
from __future__ import annotations
import json
import math
import os
import re
import time


def numerical_progress(text:str)->list[dict]:
    rows=[]
    for line in text.splitlines():
        common={"accuracy","max_source_regression","selection_score","optimizer_step"}
        markers=(("CONTRASTIVE_DEV_SELECTION ",common,"generated_dev"),
                 ("FOCUSED_DEV_SELECTION ",common|{"minimum_rule_accuracy","collapsed_rules"},"generated_dev"),
                 ("FOCUSED_TEST_STAGE_COMPLETED ",{"completed_stages","responses"},"generated_test_progress"))
        for marker,expected,kind in markers:
            if marker in line:
                try:
                    obj=json.loads(line.split(marker,1)[1])
                    if set(obj)==expected and all(type(v) in (int,float) and math.isfinite(v) for v in obj.values()):
                        rows.append({kind:True,**obj})
                except (ValueError,TypeError):pass
        # Capture ONLY standard numerical Trainer fields. Source text, tokens,
        # generated answers, URLs and exception details are never echoed.
        found=re.search(r"\{('(?:loss|eval_loss|train_runtime)':[^\n]{1,1200})\}",line)
        if not found:continue
        row={}
        for name in ("loss","eval_loss","grad_norm","learning_rate","epoch","train_runtime","train_steps_per_second"):
            value=re.search(r"'"+name+r"': (-?[0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)",found.group(0))
            if value:row[name]=float(value.group(1))
        if row:rows.append(row)
    return rows


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglehub.auth import get_username
    from kagglesdk import KaggleEnv
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    owner=get_username();slug=os.environ.get("WILLIAMS_OBSERVE_SLUG","williams-ai-recovered-full-course-rank16-20261027")
    api=KaggleApi();api.authenticate()
    status=api.kernels_status(owner+"/"+slug)
    print("OWN_CONTINUATION_STATE",str(status.status),flush=True)
    seen=set()
    def emit(raw):
        for line in raw.splitlines():
            if "ACTUAL_DUAL_T4_DDP " in line:
                try:
                    payload=json.loads(line.split("ACTUAL_DUAL_T4_DDP ",1)[1])
                    if (set(payload)=={"world_size","gpu_names"} and payload["world_size"]==2
                        and len(payload["gpu_names"])==2 and all(isinstance(n,str) and re.fullmatch(r"[A-Za-z0-9 ()_-]{1,80}",n) for n in payload["gpu_names"])):
                        print("OWN_OBSERVED_ACTUAL_GPUS",json.dumps(payload),flush=True)
                except (ValueError,TypeError):pass
        for row in numerical_progress(raw):
            serial=json.dumps(row,sort_keys=True,allow_nan=False)
            if serial not in seen:seen.add(serial);print("OWN_TRAINER_NUMERICAL_PROGRESS",serial,flush=True)
    with api.build_kaggle_client() as client:
        request=ApiListKernelSessionOutputRequest();request.user_name=owner;request.kernel_slug=slug
        response=client.kernels.kernels_api_client.list_kernel_session_output(request)
        emit(response.log or "")
        print("OWN_SAFE_OUTPUT_FILE_COUNT",len(response.files),flush=True)
        # Official live SSE endpoint; same authenticated session, CPU only.
        # A bounded probe replays progress without revealing log text.
        try:
            http=client._http_client;http._init_session()
            base=http._endpoint if http._env==KaggleEnv.PROD else http._endpoint+"/api"
            headers=dict(http._session.headers);headers["Accept"]="text/event-stream, */*";headers.pop("Content-Type",None)
            with http._session.get(base+f"/v1/kernels/logs/stream/{owner}/{slug}",stream=True,
                    headers=headers,auth=http._session.auth,timeout=(10,15)) as stream:
                stream.raise_for_status();deadline=time.monotonic()+25
                for raw in stream.iter_lines(decode_unicode=True):
                    if time.monotonic()>deadline:break
                    if not raw:continue
                    if isinstance(raw,bytes):raw=raw.decode("utf-8",errors="replace")
                    if raw.startswith("data:"):
                        body=raw[5:].strip()
                        if body=="END_OF_LOG":break
                        try:emit(str(json.loads(body).get("data","")))
                        except (ValueError,AttributeError):pass
                    else:emit(raw)
        except Exception as error:
            print("OWN_LIVE_LOG_PROBE_RESULT",type(error).__name__,flush=True)
        try:
            from kagglesdk.kernels.types.kernels_api_service import ApiGetAcceleratorQuotaStatisticsRequest
            quota=client.kernels.kernels_api_client.get_accelerator_quota_statistics(ApiGetAcceleratorQuotaStatisticsRequest()).gpu_quota
            if quota:
                print("OWN_GPU_QUOTA_HOURS",json.dumps({"used":quota.time_used.total_seconds()/3600,
                    "total":quota.total_time_allowed.total_seconds()/3600}),flush=True)
        except Exception as error:
            print("OWN_QUOTA_PROBE_RESULT",type(error).__name__,flush=True)


if __name__=="__main__":main()
