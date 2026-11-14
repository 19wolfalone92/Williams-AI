"""Gold-free historic100 actual model input isolation, with zero real LLM calls."""
import unittest
from scripts.ai_v6_historical100_models import QUESTIONS, FAMILIES, SCHEMA, ROLES


class Historical100ModelContract(unittest.TestCase):
    def test_editions_have_frozen_questions_and_families(self):
        self.assertEqual(set(QUESTIONS), {"TC1", "NTD", "TC2", "COURSE"})
        self.assertEqual(FAMILIES, {"TC1":"ENTRY","NTD":"ENTRY","TC2":"EXIT","COURSE":"WAVE"})
        self.assertEqual(len(set(QUESTIONS.values())),4)

    def test_models_are_not_gold_or_trade_engines(self):
        self.assertEqual(ROLES, ("matched_base", "candidate_v5"))
        self.assertIn("NO_EXPERT_GOLD", SCHEMA)


if __name__ == "__main__":
    unittest.main()
