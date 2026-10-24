"""No fake current market, auto orders or claims of independently verified books."""
import unittest
from williams_ai.fresh_real_book_chart_exam import CASES,run_fresh_book_chart_exam

class FreshRealBookChartCases(unittest.TestCase):
    def test_all_four_editions_get_precommitted_new_market_pairs(self):
        self.assertEqual(len(CASES),5)
        self.assertEqual({x[1] for x in CASES},{"TC1","NTD","TC2","COURSE"})
        self.assertEqual(len({x[0] for x in CASES}),5)

    def test_fake_model_rejected_before_any_public_data_access(self):
        class NotAModel:
            def propose_chart_reasoning(self,*args,**kwargs):
                raise AssertionError("Fake model must never run")
        def never_fetch(url):
            raise AssertionError("Public market must not be read without model weights")
        for false_identity in ({}, {"verified_model_magic":"FAKE"},
                               {"verified_model_magic":"GGUF","model_sha256":"bad"}):
            with self.assertRaises(ValueError):
                run_fresh_book_chart_exam(
                    NotAModel(),weights=false_identity,fetch_json=never_fetch)


if __name__=="__main__":
    unittest.main()
