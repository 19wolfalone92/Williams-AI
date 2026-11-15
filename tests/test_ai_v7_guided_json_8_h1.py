import unittest
from scripts.ai_v7_guided_json_8_h1_worker import IDS,SCHEMA
from scripts.ai_v7_guided_json_8_h1_submit import SLUG
class FrozenEightResearchPilotTests(unittest.TestCase):
 def test_exact_cases_and_protect_existing_v6_outputs(self):
  self.assertEqual(IDS,tuple(f"HIST100-{n:02d}-{k}" for n in (1,2) for k in (1,2,3,4)))
  self.assertEqual(len(IDS),8)
  self.assertTrue(SLUG.startswith("williams-v7-"))
  self.assertIn("NO_GOLD",SCHEMA)
if __name__=="__main__":unittest.main()
