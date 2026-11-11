"""After-fix strategy regression probes against real Williams AI v2 code.

These are software correctness checks, NOT evidence of learned market skill,
human expert verification, independent model evaluation or trading safety.
"""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from tests.test_williams_ai_reasoning_v2_blocker_fixes import full_case, reseal, ToyTokenizer
from williams_ai.reasoning_research_v2 import (
    HypothesisMemory, adaptive_schedule, check_corpus, digest, grade_blind,
    prediction_commitment, question_only,
)
from williams_ai.reasoning_qlora_bridge_v2 import prepare_gpu_handoff, tokenize_verified
from williams_ai.reasoning_training_readiness_v2 import readiness


def refuses(call):
    try: call()
    except ValueError: return True
    return False


def audit():
    issues=[]
    def record(problem,passed,evidence,scope="SOFTWARE_CONTRACT"):
        issues.append({"blocker":problem,"regression_check_passed":bool(passed),
                       "evidence":evidence,"scope":scope})
    basic=full_case()
    edited=deepcopy(basic);edited["question"]+=" Alternate study phrase.";reseal(edited)
    record("A1_FULL_CONTENT_CORPUS_SHA",
           check_corpus([basic])["source_sha256"]!=check_corpus([edited])["source_sha256"],
           "Case content, approved answer and market prefix now enter the digest.")
    held=deepcopy(basic);held["split"]="test";reseal(held)
    p={"case_id":held["case_id"],"cutoff":held["cutoff"],
       "visible_prefix_sha256":held["visible_prefix_sha256"],
       "prediction_at":held["cutoff"],
       "answer":{"decision":"WAIT","execution_allowed":False}}
    p["commitment"]=prediction_commitment(p)
    record("A2_FULL_REASONING_ANSWER_REQUIRED",
           refuses(lambda: grade_blind([held],[p])),
           "A naked WAIT answer is rejected; schema and reasoning fields required.")
    tokens=tokenize_verified([basic],ToyTokenizer(),max_tokens=8192)
    record("A3_REVIEWED_REASONING_TO_QLORA_TOKEN_BRIDGE",
           len(tokens)==1 and -100 in tokens[0]["labels"] and
           any(x!=-100 for x in tokens[0]["labels"]) and
           refuses(lambda: prepare_gpu_handoff([basic])),
           "Real QLoRA adapter generates answer-only tensor labels, while incomplete reviewed corpora cannot start.",
           scope="CPU_TOKENIZATION_AND_GATE_ONLY_GPU_NOT_RUN")
    feedback=adaptive_schedule([basic],
        [{"case_id":"valid-001","split":"validation","rule_id":"NTD.AC",
          "error_code":"MISUNDERSTOOD_AC"}],rounds=4)
    record("A4_SKILL_ADAPTATION_ACROSS_RULE_HOLDOUT",
           feedback["error_skill_counts"].get("MOMENTUM")==1 and
           any(x["focus_skill"]=="MOMENTUM" and x["validation_failures_for_skill"]>=1
               for x in feedback["schedule"]),
           "An NTD.AC validation error prioritizes a compatible train-side skill.")
    alias=deepcopy(basic);alias["symbol"]="BTCUSDC";alias["split"]="test"
    alias["case_id"]="alias-fixture";reseal(alias)
    record("A5_QUOTE_ALIAS_NO_CROSS_SPLIT",
           refuses(lambda: check_corpus([basic,alias])),
           "BTCUSDT vs BTCUSDC cannot occupy different splits.")
    prompt=question_only(basic)
    incomplete=deepcopy(basic)
    incomplete.pop("higher_timeframes");incomplete.pop("book_evidence");reseal(incomplete)
    record("A6_CAUSAL_THREE_FRAME_BOOK_EVIDENCE_GATE",
           prompt["complete_h1_h4_d1_context"] and
           bool(prompt["book_evidence"]) and
           not readiness([incomplete])["ready_for_gpu_experiment"],
           "H1/H4/D1 closed histories, strict confirmed fractals and source page evidence; missing context blocks GPU.")
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/"learning.sqlite"
        parent=HypothesisMemory(path,run_id="trial-001",model_id="qwen-parent-a")
        cand=HypothesisMemory(path,run_id="trial-001",model_id="qwen-candidate-b")
        for mem,thought in ((parent,"parent"),(cand,"candidate")):
            mem.append(case_id="same-case",symbol="BTCUSDT",
                       cutoff="2026-11-01T01:00:00Z",
                       available_at="2026-11-01T01:00:00Z",
                       snapshot_sha256="a"*64,hypothesis=thought,
                       alternative="another path",evidence=["close"],
                       counterevidence=["unknown"],invalidation="stop")
        pa=parent.as_of(symbol="BTCUSDT",timestamp="2026-11-02T00:00:00Z")
        ca=cand.as_of(symbol="BTCUSDT",timestamp="2026-11-02T00:00:00Z")
        record("A7_ISOLATED_EXPERIMENT_AND_MODEL_MEMORY",
               pa[0]["hypothesis"]!=ca[0]["hypothesis"],
               "Two models share SQLite without state leakage or primary-key collision.")
    return {"schema":"WILLIAMS_AI_2_STRATEGY_BLOCKER_POSTFIX_AUDIT_V2",
            "checked":len(issues),
            "software_contracts_passed":sum(x["regression_check_passed"] for x in issues),
            "regressions":issues,
            "new_expert_labels_supplied":False,
            "real_gpu_training_performed":False,
            "semantic_quality_evaluated":False,
            "trade_execution_allowed":False}


if __name__=="__main__":
    report=audit()
    output=ROOT/"audit_output/strategy_blockers_after_fixes.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if report["checked"] != report["software_contracts_passed"]:
        raise SystemExit("UNRESOLVED_STRATEGY_SOFTWARE_REGRESSION")
