"""CPU-only contract tests: no model weights, Kaggle credentials or GPU."""
import copy
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from williams_ai.independent_chart_holdout import build_blind_market_cases
from williams_ai.qwen4b_blind_evaluation import (
    CLASSES, candidate_first_token_ids, grade_pair, market_prompt,
    verified_blind_questions,
)

ARCHIVE=Path(__file__).resolve().parents[1]/"examples"/"ai_binance_15pair_h1_prior_holdout_20260902.json"


class FakeQwenTokenizer:
    def __init__(self, alias=False): self.alias=alias
    def apply_chat_template(self,messages,tokenize=True,add_generation_prompt=False,enable_thinking=False):
        prefix=[9,10,11,12]
        if add_generation_prompt: return prefix
        name=messages[-1]["content"]
        mapping={"UP":101,"DOWN":102,"RANGE":103}
        return prefix + [101 if self.alias else mapping[name],2]


class Qwen4BBlindEvaluationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        archive=json.loads(ARCHIVE.read_text(encoding="utf-8"))
        cls.questions,cls.keys,_=build_blind_market_cases(archive)

    @staticmethod
    def write_questions(path,rows):
        path.write_text("\n".join(json.dumps(q,ensure_ascii=False) for q in rows)+"\n",
                        encoding="utf-8")

    def test_read_real_15pair_30_questions_without_keys(self):
        with TemporaryDirectory() as td:
            p=Path(td)/"questions.jsonl"
            self.write_questions(p,self.questions)
            rows,digest=verified_blind_questions(p)
            self.assertEqual(len(rows),30)
            self.assertEqual(len(digest),64)
            prompt=market_prompt(rows[0])
            self.assertIn("closed_tail",prompt)
            self.assertIn("market_sha256",prompt)
            self.assertNotIn("true_class",prompt)
            self.assertNotIn("future_return",prompt)
            self.assertNotIn("result_available_at",prompt)

    def test_injection_of_future_label_into_any_question_is_rejected(self):
        with TemporaryDirectory() as td:
            p=Path(td)/"questions.jsonl"
            cases=copy.deepcopy(self.questions)
            cases[0]["future_return"]=0.1
            self.write_questions(p,cases)
            with self.assertRaisesRegex(ValueError,"FUTURE_BLIND"):
                verified_blind_questions(p)

    def test_answer_classes_have_unique_first_token_after_same_prompt(self):
        prefix,ids=candidate_first_token_ids(FakeQwenTokenizer(),
            [{"role":"user","content":"independent past candles only"}])
        self.assertEqual(prefix,[9,10,11,12])
        self.assertEqual(ids,[101,102,103])
        with self.assertRaisesRegex(ValueError,"NOT_UNIQUELY"):
            candidate_first_token_ids(FakeQwenTokenizer(alias=True),
                [{"role":"user","content":"bar"}])

    def test_future_outcomes_graded_only_after_prediction_file_committed(self):
        with TemporaryDirectory() as td:
            root=Path(td)
            baseline=[{"case_id":key["case_id"],
                "probabilities":dict(zip(CLASSES,(1/3,1/3,1/3)))}
                for key in self.keys]
            payload={"receipt":{
                "mode":"PRECOMMITTED_QWEN3_4B_REAL_H1_NEXT_TOKEN_COMPARISON",
                "true_future_outcomes_loaded_in_prediction":False},
                "baseline":baseline,"adapted":copy.deepcopy(baseline)}
            predictions=root/"precommitted.json"
            predictions.write_text(json.dumps(payload))
            keys=root/"WITHHELD_OUTCOME_KEYS.jsonl"
            keys.write_text("\n".join(json.dumps(x) for x in self.keys)+"\n")
            report=grade_pair(predictions=predictions,withheld_outcomes=keys,
                              new_report=root/"report.json")
            self.assertEqual(report["before"]["cases"],30)
            self.assertAlmostEqual(report["before"]["multiclass_brier"],2/9)
            self.assertEqual(report["brier_improvement"],0)
            self.assertFalse(report["trading_qualification"])
            self.assertFalse(report["simple_holdout_improved_brier_only"])

    def test_no_scoring_without_full_precommitted_pair(self):
        with TemporaryDirectory() as td:
            root=Path(td)
            (root/"partial.json").write_text(json.dumps({"receipt":{"mode":"fake"},
                "baseline":[],"adapted":[]}))
            (root/"keys.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(ValueError,"PRECOMMIT_REQUIRED"):
                grade_pair(predictions=root/"partial.json",
                           withheld_outcomes=root/"keys.jsonl",
                           new_report=root/"report.json")


if __name__=="__main__": unittest.main()
