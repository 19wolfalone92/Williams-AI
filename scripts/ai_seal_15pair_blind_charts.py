"""Export REAL archived market forecast prompts with outcome keys kept separate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from williams_ai.independent_chart_holdout import build_blind_market_cases


def main():
    p=argparse.ArgumentParser(description="Seal market forecast cases; no Qwen calls or trades")
    p.add_argument("--archive",type=Path,default=Path(
        "examples/ai_binance_15pair_h1_prior_holdout_20260902.json"))
    p.add_argument("--new-output",type=Path,required=True)
    a=p.parse_args()
    if a.new_output.exists() or a.new_output.is_symlink():
        raise ValueError("SEALED_CASE_OUTPUT_MUST_NOT_BE_OVERWRITTEN")
    archive=json.loads(a.archive.read_text(encoding="utf-8"))
    prompts,keys,seal=build_blind_market_cases(archive)
    a.new_output.mkdir(parents=True,mode=0o700,exist_ok=False)
    (a.new_output/"model_questions.jsonl").write_text("\n".join(
        json.dumps(row,ensure_ascii=False,separators=(",",":")) for row in prompts)+"\n")
    (a.new_output/"WITHHELD_OUTCOME_KEYS.jsonl").write_text("\n".join(
        json.dumps(row,ensure_ascii=False,separators=(",",":")) for row in keys)+"\n")
    (a.new_output/"evaluation_receipt.json").write_text(
        json.dumps(seal,ensure_ascii=False,indent=2))
    print(json.dumps(seal,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
