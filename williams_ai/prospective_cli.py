"""Two-step LOCAL model evaluation with private closed H1 snapshots.

Run PREPARE now with only the actually observed candles. Keep the resulting
commitment outside git and publish its hash to an independent timestamping
source BEFORE the future 24h unfolds. SETTLE only with subsequently received
complete hourly candles; model cannot revise the frozen response.
This CLI does not fetch candles, trade, or certify a market edge.

python -m williams_ai.prospective_cli prepare \
 --history /private/past_h1.json --model /private/model.gguf \
 --code-id <pinned-git-sha> --out /private/commitment.json
python -m williams_ai.prospective_cli settle \
 --history /private/extended_h1.json --seal /private/commitment.json \
 --out /private/observed_result.json

Input JSON: {"symbol":"ETHUSDT","tick_size":0.01,"bars":[...]}.
Do NOT include private API keys/credentials in these local files.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .prospective_qualification import (
    freeze_prospective_call, settle_prospective_call,
)


def _read_json(path:Path)->Any:
    if path.is_symlink() or not path.is_file() or path.stat().st_size>20000000:
        raise ValueError("Expected existing private regular JSON under 20MB")
    with path.open("r",encoding="utf-8") as stream:
        return json.load(stream)


def _market_file(path:Path)->dict[str,Any]:
    value=_read_json(path)
    if not isinstance(value,dict) or set(value)!={"symbol","tick_size","bars"}:
        raise ValueError("Only symbol, tick_size and closed bars allowed; no credentials")
    if not isinstance(value["bars"],list) or len(value["bars"])>50000:
        raise ValueError("Invalid number of closed H1 candles")
    return value


def _write_once_private(path:Path,data:dict[str,Any])->None:
    """No silent mutation of a sealed prediction or completed outcome."""
    path.parent.mkdir(parents=True,exist_ok=True)
    descriptor=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    try:
        with os.fdopen(descriptor,"w",encoding="utf-8") as stream:
            json.dump(data,stream,ensure_ascii=False,indent=2,allow_nan=False)
            stream.write("\n")
    except Exception:
        path.unlink(missing_ok=True)
        raise


def main()->None:
    parser=argparse.ArgumentParser(
        description="Prospectively freeze, then score real local Williams-AI GGUF forecasts")
    sub=parser.add_subparsers(dest="action",required=True)
    prepare=sub.add_parser("prepare",help="Only past closed H1 bars; invokes REAL local GGUF")
    settle=sub.add_parser("settle",help="24 hours later: compare with previously sealed answer")
    for p in (prepare,settle):
        p.add_argument("--history",type=Path,required=True)
        p.add_argument("--out",type=Path,required=True)
    prepare.add_argument("--model",type=Path,required=True)
    prepare.add_argument("--code-id",type=str,required=True)
    prepare.add_argument("--gpu-layers",type=int,default=0)
    settle.add_argument("--seal",type=Path,required=True)
    a=parser.parse_args()
    source=_market_file(a.history)
    if a.action=="prepare":
        from .local_model import LocalWilliamsModel
        from .study_cli import _model_identity
        student=LocalWilliamsModel(a.model,n_ctx=8192,gpu_layers=a.gpu_layers)
        report=freeze_prospective_call(
            source["symbol"],source["bars"],
            tick_size=source["tick_size"],
            model_id=_model_identity(a.model),
            code_id=a.code_id,
            predictor=student.propose_24h_research_forecast,
        )
    else:
        sealed=_read_json(a.seal)
        if sealed.get("symbol")!=source["symbol"]:
            raise ValueError("Sealed pair does not match observed pair")
        report=settle_prospective_call(sealed,source["bars"])
    _write_once_private(a.out,report)
    # Only send the SHA and non-sensitive mode to stdout; local price
    # source, the whole history and private GGUF paths are never logged.
    print("WILLIAMS_AI_PROSPECTIVE_"+a.action.upper(),
          report.get("commitment_sha256"),
          "NO_TRADE",flush=True)


if __name__=="__main__":
    main()
