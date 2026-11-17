"""Offline synthetic v7.2 comparison; never mock expert gold as real."""
import unittest
import copy
from scripts.ai_v72_existing_kernel_status import _summary,_safe_output_error,compare_v72
from scripts.ai_v71_existing_kernel_status import EXPECTED
from williams_ai.v72_eos_guard import SCHEMA,ENFORCER_COMMIT
from williams_ai.v6_v5_lora_binding import MODEL_COMMIT,RAW_ADAPTER_SHA

class FakeProcess:
    returncode=1
    stdout="SECRET_DO_NOT_LEAK"
    stderr="ModuleNotFoundError: No module named 'interegular' SECRET_DO_NOT_LEAK"

class V72SafeComparisonTests(unittest.TestCase):
    def test_unreleased_kaggle_error_output_is_sanitized(self):
        x=_safe_output_error(FakeProcess(),0)
        self.assertNotIn("SECRET_DO_NOT_LEAK",str(x))
        self.assertEqual(x["matching_file_count"],0)
    def test_abstain_is_valid_but_error_is_not(self):
        x=_summary([
          {"finish_reason":"stop","decision":"ABSTAIN","valid_model_decision":True,
           "tokens_out":30,"metrics":{"eos_blocked":5},"failure_code":None},
          {"finish_reason":"error","decision":None,"valid_model_decision":False,
           "tokens_out":80,"metrics":{"eos_blocked":2},"failure_code":"INVALID_JSON"}])
        self.assertEqual(x["valid_json_responses"],1)
        self.assertEqual(x["technical_failure_codes"],{"INVALID_JSON":1})
        self.assertEqual(x["eos_block_events"],7)
    def test_failed_contract_never_claims_eight_complete(self):
        with self.assertRaisesRegex(ValueError,"NOT_EXACT_8"):
            compare_v72({"schema":SCHEMA,"complete":False,"paired_cases":7},{},{})
    def test_expected_original_ids_still_eight(self):
        self.assertEqual(len(EXPECTED),8)
        self.assertEqual(MODEL_COMMIT,"3fb356690bbdae7353c94efb3eecdce276e6ed1c")
        self.assertEqual(len(RAW_ADAPTER_SHA),64)
if __name__=="__main__":
    unittest.main()
