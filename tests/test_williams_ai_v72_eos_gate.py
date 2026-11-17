"""Source-agnostic decoding tests; no real model weights, Kaggle or trading."""
import unittest
from williams_ai.v72_eos_guard import SCHEMA,filter_eos_until_json_complete

class V72EOSProofTests(unittest.TestCase):
    def test_detect_premature_eos_inside_unterminated_json_string(self):
        prefix='{"decision":"ABSTAIN","reason":"Н'
        allowed,blocked=filter_eos_until_json_complete(
            [2,151645,3],151645,prefix,"ENTRY")
        self.assertTrue(blocked)
        self.assertEqual(allowed,[2,3])
    def test_schema_complete_json_may_emit_eos(self):
        prefix='{"decision":"ABSTAIN","reason":"Insufficient observed facts."}'
        allowed,blocked=filter_eos_until_json_complete(
            [151645,13],151645,prefix,"ENTRY")
        self.assertFalse(blocked)
        self.assertEqual(allowed,[151645,13])
    def test_malformed_but_closed_json_is_not_a_valid_termination(self):
        for prefix in ('{"decision":"BUY","reason":"x"}',
                       '{"decision":"ABSTAIN","reason":""}',
                       '{"decision":"ABSTAIN","reason":"x","junk":7}',
                       '{"decision":"ABSTAIN","reason":"x","reason":"y"}'):
            allowed,blocked=filter_eos_until_json_complete(
                [5,151645],151645,prefix,"ENTRY")
            self.assertTrue(blocked)
            self.assertEqual(allowed,[5])
    def test_no_repair_when_eos_is_not_allowed(self):
        allowed,blocked=filter_eos_until_json_complete(
            [1,2,3],151645,"{","ENTRY")
        self.assertFalse(blocked)
        self.assertEqual(allowed,[1,2,3])
    def test_only_early_eos_fail_closed(self):
        with self.assertRaisesRegex(ValueError,"ONLY_PREMATURE_EOS"):
            filter_eos_until_json_complete(
                [151645],151645,'{"decision":"ABSTAIN","reason":"N',"ENTRY")
    def test_multiple_eos_ids_all_blocked(self):
        allowed,blocked=filter_eos_until_json_complete(
            [151643,151645,17],[151643,151645],"{","ENTRY")
        self.assertTrue(blocked)
        self.assertEqual(allowed,[17])
    def test_malformed_allowed_token_types_rejected(self):
        with self.assertRaisesRegex(ValueError,"BAD_ALLOWED"):
            filter_eos_until_json_complete([True,151645],151645,"{","ENTRY")
        with self.assertRaisesRegex(ValueError,"BAD_EOS_IDS"):
            filter_eos_until_json_complete([2],False,"{","ENTRY")
    def test_eos_guard_is_not_an_expert_gold_or_training_schema(self):
        self.assertIn("NO_GOLD",SCHEMA)
        self.assertNotIn("TRAINED",SCHEMA)
if __name__=="__main__":
    unittest.main()
