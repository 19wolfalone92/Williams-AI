import unittest
from scripts.ai_v6_hist100_kaggle_last2_worker import CASE_IDS, FALLBACK_MAX_INPUT_TOKENS, FALLBACK_MIN_H1_BARS,SCHEMA
from scripts.ai_v6_hist100_kaggle_last2_submit import NEW_SLUG

class Final2ModelContract(unittest.TestCase):
    def test_locked_remaining_cases(self):
        self.assertEqual(CASE_IDS,("HIST100-25-3","HIST100-25-4"))
        self.assertEqual(FALLBACK_MAX_INPUT_TOKENS,4096)
        self.assertEqual(FALLBACK_MIN_H1_BARS,16)
        self.assertTrue("NO_GOLD" not in SCHEMA or "REAL" in SCHEMA)
        self.assertTrue(NEW_SLUG.startswith("williams-v6-hist100-final-two"))
if __name__=="__main__":unittest.main()
