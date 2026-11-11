"""Synthetic-only regression probes; never expert trading gold."""
from copy import deepcopy
from datetime import datetime,timedelta
from pathlib import Path
import tempfile
import unittest
import re
from tests.test_williams_ai_reasoning_research_v2 import example
from williams_ai.reasoning_research_v2 import (
    HypothesisMemory,adaptive_schedule,check_corpus,digest,
    grade_blind,prediction_commitment,question_only,verify_case)
from williams_ai.reasoning_qlora_bridge_v2 import (
    prepare_gpu_handoff,tokenize_verified,validate_gpu_handoff)

def reseal(x):
    signature=digest({k:v for k,v in x.items() if k!="reviewers"})
    for r in x["reviewers"]:r["case_sha256"]=signature

def full_case():
    x=example("fixture-case","train","BTCUSDT")
    last=datetime.fromisoformat(x["cutoff"])
    frames={}
    for tf,delta in (("H4",timedelta(hours=4)),("D1",timedelta(days=1))):
        end=last.replace(hour=0,minute=0,second=0,microsecond=0)
        bars=[]
        for i in range(220):
            bars.append({"close_time":(end-(219-i)*delta).isoformat(),
                         "open":100.0+i,"high":102.0+i,"low":99.0+i,
                         "close":101.0+i,"volume":50.0,"closed":True})
        frames[tf]={"bars":bars,"visible_prefix_sha256":digest(bars)}
    x["higher_timeframes"]=frames
    x["book_evidence"]={"edition":"TC2","pdf_page":124,
      "source_sha256":x["source_sha256"],
      "excerpt":"Synthetic-only test excerpt; no real human book review.",
      "checked_by_reviewers":True}
    reseal(x)
    return x

class ToyTokenizer:
    def apply_chat_template(self,messages,*,tokenize,add_generation_prompt,enable_thinking):
        raw="".join("<"+m["role"]+">"+m["content"]+"</"+m["role"]+">" for m in messages)
        if add_generation_prompt:raw+="<assistant>"
        return [len(x) for x in re.findall(r"<[^>]+>|\w+|[^\s]",raw)]

class BlockerFixes(unittest.TestCase):
    def test_full_digest_covers_gold_question_and_reviews(self):
        a=full_case();b=deepcopy(a);b["question"]+=" Supplement.";reseal(b)
        self.assertNotEqual(check_corpus([a])["source_sha256"],
                            check_corpus([b])["source_sha256"])
    def test_incomplete_answer_cannot_pass_exam(self):
        a=full_case();a["split"]="test";reseal(a)
        p={"case_id":a["case_id"],"cutoff":a["cutoff"],
           "visible_prefix_sha256":a["visible_prefix_sha256"],
           "prediction_at":a["cutoff"],
           "answer":{"decision":"WAIT","execution_allowed":False}}
        p["commitment"]=prediction_commitment(p)
        with self.assertRaisesRegex(ValueError,"EXACT_EVIDENCE"):
            grade_blind([a],[p])
        p["answer"]=deepcopy(a["answer"]);p["commitment"]=prediction_commitment(p)
        report=grade_blind([a],[p])
        self.assertEqual(report["decision_accuracy"],1.0)
        self.assertIn("confidence_correctness_brier",report)
    def test_quote_aliases_never_cross_split(self):
        a=full_case();b=deepcopy(a);b["split"]="test"
        b["case_id"]="another-case";b["symbol"]="BTCUSDC";reseal(b)
        with self.assertRaisesRegex(ValueError,"UNDERLYING"):
            check_corpus([a,b])
    def test_adapt_by_skill_across_rule_disjoint_train_validation(self):
        a=full_case()
        report=adaptive_schedule([a],[{"case_id":"valid-case","split":"validation",
                  "rule_id":"NTD.AC","error_code":"MISUNDERSTOOD_AC"}],rounds=4)
        self.assertEqual(report["error_skill_counts"]["MOMENTUM"],1)
        self.assertTrue(any(x["focus_skill"]=="MOMENTUM" and
                    x["validation_failures_for_skill"]>0 for x in report["schedule"]))
    def test_full_closed_h1_h4_d1_source_and_no_future(self):
        a=full_case();verify_case(a);prompt=question_only(a)
        self.assertTrue(prompt["complete_h1_h4_d1_context"])
        self.assertEqual(set(prompt["timeframe_observations"]),{"H1","H4","D1"})
        self.assertTrue(prompt["book_evidence"]["excerpt"])
        b=deepcopy(a);b["higher_timeframes"]["D1"]["bars"][-1]["close_time"]="2026-12-01T00:00:00Z"
        b["higher_timeframes"]["D1"]["visible_prefix_sha256"]=digest(b["higher_timeframes"]["D1"]["bars"])
        reseal(b)
        with self.assertRaisesRegex(ValueError,"FUTURE_LEAKAGE"):
            verify_case(b)
    def test_actual_chat_tokens_answer_only_and_no_test_gold(self):
        a=full_case()
        rows=tokenize_verified([a],ToyTokenizer(),max_tokens=8192)
        self.assertEqual(len(rows),1)
        self.assertIn(-100,rows[0]["labels"])
        self.assertTrue(any(x!=-100 for x in rows[0]["labels"]))
        b=deepcopy(a);b["split"]="test";reseal(b)
        with self.assertRaisesRegex(ValueError,"NEVER_TOKENIZE_TEST_GOLD"):
            tokenize_verified([b],ToyTokenizer(),max_tokens=8192)
    def test_not_ready_corpus_cannot_go_to_gpu(self):
        with self.assertRaisesRegex(ValueError,"NOT_GPU_READY"):
            prepare_gpu_handoff([full_case()])
        with self.assertRaisesRegex(ValueError,"FROZEN_GPU"):
            validate_gpu_handoff([full_case()],{})
    def test_isolated_models_can_write_same_case_in_one_db(self):
        with tempfile.TemporaryDirectory() as root:
            a=HypothesisMemory(Path(root)/"memory.db",model_id="qwen-parent-aa",run_id="replay-0001")
            b=HypothesisMemory(Path(root)/"memory.db",model_id="qwen-candidate-bb",run_id="replay-0001")
            for m,thought in ((a,"parent hypothesis"),(b,"candidate hypothesis")):
                m.append(case_id="case-one",symbol="BTCUSDT",cutoff="2026-11-01T01:00:00Z",
                         available_at="2026-11-01T01:00:00Z",snapshot_sha256="a"*64,
                         hypothesis=thought,alternative="uncertainty",evidence=["fact"],
                         counterevidence=["doubt"],invalidation="invalidate")
            self.assertNotEqual(a.as_of(symbol="BTCUSDT",timestamp="2026-11-02T00:00:00Z")[0]["hypothesis"],
                                b.as_of(symbol="BTCUSDT",timestamp="2026-11-02T00:00:00Z")[0]["hypothesis"])

if __name__=="__main__":unittest.main()
