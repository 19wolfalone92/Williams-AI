"""Offline chart-picture observation with an actual multimodal GGUF + mmproj.

The adapter runs locally and is READ_ONLY. A text-only GGUF cannot see
images. CI mocks the model: passing CI DOES NOT verify visual ability.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

MIMES={".png":"image/png",".jpg":"image/jpeg",".jpeg":"image/jpeg"}


def local_image_data(path:str|Path)->tuple[str,str]:
    src=Path(path).expanduser()
    if src.is_symlink() or not src.is_file() or src.suffix.lower() not in MIMES:
        raise ValueError("Only regular local PNG/JPEG charts supported")
    if not 32<=src.stat().st_size<=10*1024*1024:
        raise ValueError("Image must be 32 bytes to 10MiB")
    raw=src.read_bytes()
    mime=MIMES[src.suffix.lower()]
    if mime=="image/png" and not raw.startswith(bytes.fromhex("89504e470d0a1a0a")):
        raise ValueError("Invalid PNG magic")
    if mime=="image/jpeg" and not raw.startswith(bytes.fromhex("ffd8ff")):
        raise ValueError("Invalid JPEG magic")
    return ("data:"+mime+";base64,"+base64.b64encode(raw).decode("ascii"),
            hashlib.sha256(raw).hexdigest())


class LocalChartVision:
    def __init__(self,model_path:str|Path,mmproj_path:str|Path,*,
                 loader:Callable[...,Any]|None=None,n_ctx:int=8192,gpu_layers:int=0):
        model=Path(model_path).expanduser()
        proj=Path(mmproj_path).expanduser()
        for f in (model,proj):
            if f.is_symlink() or not f.is_file() or f.suffix.lower()!=".gguf":
                raise FileNotFoundError("Need local GGUF vision model and matching mmproj GGUF")
        if not isinstance(n_ctx,int) or not 4096<=n_ctx<=32768:
            raise ValueError("Invalid image context size")
        if not isinstance(gpu_layers,int) or not -1<=gpu_layers<=256:
            raise ValueError("Invalid GPU layers")
        if loader is None:
            try:
                from llama_cpp import Llama
            except ImportError as exc:
                raise RuntimeError("Install a modern MTMD llama-cpp-python") from exc
            loader=Llama
        try:
            self._llm=loader(model_path=str(model.resolve()),
                             mmproj_path=str(proj.resolve()),
                             n_ctx=n_ctx,n_gpu_layers=gpu_layers,verbose=False)
        except TypeError as exc:
            raise RuntimeError("Upgrade to a multimodal llama-cpp-python build accepting mmproj_path") from exc

    def observe(self,image_path:str|Path,question:str,*,
                source_id:str="USER_PRIVATE_CHART")->dict[str,Any]:
        if not isinstance(question,str) or not 1<=len(question.strip())<=900:
            raise ValueError("Question must have 1..900 chars")
        if not isinstance(source_id,str) or not 1<=len(source_id)<=80:
            raise ValueError("Invalid source identifier")
        uri,digest=local_image_data(image_path)
        system=(
            "Отдели непосредственно видимые факты на рисунке от волновых гипотез. "
            "Не считай нарисованные справа прогнозные волны историей рынка. "
            "Не придумывай уровни/цены/подволны, которых не можешь прочитать. "
            "Верни JSON: observations (массив фактов), hypotheses (альтернативы), "
            "missing_evidence (что нужно проверить). "
            "Никаких торговых приказов, не подтверждай исполнение/SL по пикселям."
        )
        obj=self._llm.create_chat_completion(
            messages=[
                {"role":"system","content":system},
                {"role":"user","content":[
                    {"type":"text","text":question.strip()},
                    {"type":"image_url","image_url":{"url":uri}},
                ]},
            ],
            temperature=0.0,max_tokens=900,
            response_format={"type":"json_object"},stream=False,
        )
        items=obj.get("choices",[]) if isinstance(obj,dict) else []
        raw=items[0].get("message",{}).get("content","") if items else ""
        try:
            ans=json.loads(raw) if isinstance(raw,str) else {}
        except json.JSONDecodeError:
            ans={}
        if not isinstance(ans,dict):
            ans={}
        def extract(key:str)->list[str]:
            value=ans.get(key)
            return [x[:500] for x in value[:8] if isinstance(x,str)] if isinstance(value,list) else []
        observations=extract("observations")
        hypotheses=extract("hypotheses")
        missing=extract("missing_evidence")
        if not any((observations,hypotheses,missing)):
            missing=["Модель не вернула проверяемый JSON; необходимо повторить исследование"]
        return {
            "source":source_id,"image_sha256":digest,"mode":"PRIVATE_LOCAL_VISION_RESEARCH_ONLY",
            "observations_unverified":observations,
            "hypotheses_unverified":hypotheses,
            "missing_evidence":missing,"verified_wave_degree":False,
            "trade_action":None,"execution_allowed":False,
            "limitations":[
                "A matching real GGUF and multimodal projector are required.",
                "CI uses a mock, not the user's actual vision model.",
                "Pixels are not independently verified OHLC or Binance fills.",
            ],
        }


def main()->None:
    import argparse
    p=argparse.ArgumentParser(description="Offline private multimodal Williams chart observation")
    p.add_argument("image",type=Path)
    p.add_argument("--local-model",type=Path,required=True)
    p.add_argument("--mmproj",type=Path,required=True)
    p.add_argument("--question",default="Что видно на графике и какие волновые версии остаются непроверенными?")
    args=p.parse_args()
    model=LocalChartVision(args.local_model,args.mmproj)
    print(json.dumps(model.observe(args.image,args.question),ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
