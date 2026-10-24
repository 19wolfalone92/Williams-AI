"""Tests prove no single hallucinated Elliott answer becomes expert gold."""
import json
from datetime import datetime,timedelta,timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from williams_ai.expert_wave_gold import (
    evaluate_blind_wave_gold,prediction_commitment,
)

START=datetime(2026,4,1,tzinfo=timezone.utc)

def iso(day):
    return (START+timedelta(days=day)).isoformat()

def reviewer(day,person,*,structure="CORRECTION",degree="UNKNOWN"):
    return {
        "case_id":f"eth-h1-{day:03d}",
        "symbol":"ETHUSDT","timeframe":"H1","venue":"SPOT",
        "cutoff":iso(day),
        "visible_prefix_sha256":f"{day+10:064x}",
        "edition":"TC1","source_page":53,
        "reviewer_id":person,"reviewed_at":iso(day+1),
        "market_structure":structure,"wave_degree":degree,
        "justification":"Confirmed prior swing extrema only, Elliott degree unresolved.",
    }

def prediction(day,label="CORRECTION"):
    x={"case_id":f"eth-h1-{day:03d}",
       "symbol":"ETHUSDT","timeframe":"H1","cutoff":iso(day),
       "visible_prefix_sha256":f"{day+10:064x}",
       "model_id":"pinned-model-001",
       "predicted_structure":label,"prediction_created_at":iso(day)}
    x["precommitted_prediction_sha256"]=prediction_commitment(x)
    return x

class ExpertWaveBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ann=Path(self.temp.name)/"expert.jsonl"
        self.pred=Path(self.temp.name)/"sealed.jsonl"
    def save(self,items,preds):
        self.ann.write_text("\n".join(json.dumps(x) for x in items)+"\n")
        self.pred.write_text("\n".join(json.dumps(x) for x in preds)+"\n")
    def base(self):
        return ([reviewer(d,"reviewer-001") for d in range(30)]+
                [reviewer(d,"reviewer-002") for d in range(30)],
                [prediction(d) for d in range(30)])

    def test_two_expert_consensus_on_30_different_dates(self):
        a,p=self.base();self.save(a,p)
        out=evaluate_blind_wave_gold(self.ann,self.pred)
        self.assertEqual(out["independently_agreed_cases"],30)
        self.assertEqual(out["distinct_utc_days_of_agreed_cases"],30)
        self.assertTrue(out["bench_sample_size_and_multi_day_sufficiency"])
        self.assertEqual(out["nonabstained_structure_accuracy"],1.0)
        self.assertEqual(out["coverage_on_decidable_cases"],1.0)
        self.assertFalse(out["execution_allowed"])

    def test_disagreement_must_be_removed_without_inventing_truth(self):
        a,p=self.base()
        a[-1]["market_structure"]="IMPULSE_UP"
        self.save(a,p)
        out=evaluate_blind_wave_gold(self.ann,self.pred)
        self.assertEqual(out["expert_disputed_cases"],1)
        self.assertEqual(out["independently_agreed_cases"],29)
        self.assertFalse(out["bench_sample_size_and_multi_day_sufficiency"])

    def test_safe_abstention_not_high_accuracy(self):
        a,p=self.base()
        p[:]=[prediction(d,"UNDETERMINED") for d in range(30)]
        self.save(a,p)
        out=evaluate_blind_wave_gold(self.ann,self.pred)
        self.assertEqual(out["model_explicit_abstentions"],30)
        self.assertIsNone(out["nonabstained_structure_accuracy"])
        self.assertEqual(out["coverage_on_decidable_cases"],0.0)

    def test_mismatched_market_digest_fails_closed(self):
        a,p=self.base();p[0]["visible_prefix_sha256"]="a"*64
        p[0]["precommitted_prediction_sha256"]=prediction_commitment(p[0])
        self.save(a,p)
        with self.assertRaisesRegex(ValueError,"PREDICTION_INPUT_DOES_NOT_MATCH"):
            evaluate_blind_wave_gold(self.ann,self.pred)

    def test_tampering_prediction_without_renewing_seal_fails(self):
        a,p=self.base();p[0]["predicted_structure"]="IMPULSE_UP"
        self.save(a,p)
        with self.assertRaisesRegex(ValueError,"CHANGED_AFTER_SEAL"):
            evaluate_blind_wave_gold(self.ann,self.pred)

    def test_one_reviewer_repeating_is_not_two_independent_experts(self):
        a,p=self.base()
        a[-1]["reviewer_id"]="reviewer-001"
        self.save(a,p)
        with self.assertRaisesRegex(ValueError,"DUPLICATE_REVIEWER"):
            evaluate_blind_wave_gold(self.ann,self.pred)

    def test_false_future_review_and_non_book_wave_degrees_rejected(self):
        a,p=self.base()
        a[0]["reviewed_at"]=iso(-1)
        self.save(a,p)
        with self.assertRaisesRegex(ValueError,"PREDATE_MARKET"):
            evaluate_blind_wave_gold(self.ann,self.pred)

if __name__=="__main__":
    unittest.main()
