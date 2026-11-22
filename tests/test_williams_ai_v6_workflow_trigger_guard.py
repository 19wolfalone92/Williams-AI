"""Prevent runaway duplicate 4B CPU jobs on generic PR updates.

Independent model CPU inference is kept opt-in via the existing separate
ai-v6-now-4book-real-market-cpu.yml workflow.
"""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent.parent

class V6HeavyJobTriggerTests(unittest.TestCase):
    def test_public_market_smoke_is_data_only_for_push_and_pr(self):
        source = (ROOT / ".github/workflows/ai-v6-public-ohlcv-smoke.yml").read_text()
        self.assertIn("  real-public-data-no-model:", source)
        self.assertIn("  pull_request:", source)
        self.assertIn("williams-v6-real-ohlcv-smoke-dataonly-v2-", source)
        self.assertNotIn("  actual-four-book-retrospective-comparison:", source)
        self.assertNotIn("python -m scripts.ai_v6_real_market_4book_cpu_smoke", source)

    def test_existing_heavy_job_is_opt_in_only_not_generic_pr(self):
        source = (ROOT / ".github/workflows/ai-v6-now-4book-real-market-cpu.yml").read_text()
        self.assertIn("  actual-four-book-retrospective-comparison:", source)
        self.assertIn("    paths: ['.github/workflows/ai-v6-now-4book-real-market-cpu.yml']", source)
        self.assertNotIn("  pull_request:", source)
        self.assertIn("  cancel-in-progress: false", source)


if __name__ == "__main__":
    unittest.main()
