"""Regression: four-book retrospective report must source symbol from chart, not case.

Pure offline unit test: no exchange network, Kaggle, weights, GPU or broker.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import ai_v6_real_market_4book_cpu_smoke as smoke
from williams_ai.independent_exam_v6 import canonical


class FourBookReportTest(unittest.TestCase):
    def test_source_symbol_is_normalized_chart_symbol(self):
        cases = []
        expected = []
        answers = []
        for cid, symbol, edition, family, question in smoke.CASES:
            # Intentionally NO case["symbol"]: the production build_cases contract.
            case = {
                "id": cid, "edition": edition, "family": family,
                "cutoff_utc": smoke.CUTOFF, "question": question,
            }
            chart = {"symbol": symbol, "closed_candles": [1, 2, 3]}
            provenance = {"archive_sha256": "a" * 64}
            cases.append((case, chart, {}, provenance))
            expected.append({
                "symbol": symbol,
                "edition": edition,
                "chart_sha256": sha256(canonical(chart)).hexdigest(),
                "official_archive_provenance": provenance,
            })
            answers.append({
                "id": cid, "edition": edition,
                "chart_sha256": sha256(canonical(chart)).hexdigest(),
                "prompt_sha256": "b" * 64,
                "retrieved_rule_ids": [], "decision": "ABSTAIN",
                "finish_reason": "stop",
            })

        with (
            patch.object(smoke, "verified_receipt"),
            patch.object(smoke, "verified_gguf",
                         side_effect=[smoke.BASE_SHA256, "c" * 64]),
            patch.object(smoke, "build_cases", return_value=cases),
            patch.object(smoke, "model_results", side_effect=[
                [row.copy() for row in answers],
                [row.copy() for row in answers],
            ]),
            patch.object(smoke, "protocol_sha256", return_value="d" * 64),
        ):
            report = smoke.run(
                base=Path("fake-base.gguf"),
                adapter=Path("fake-v5.gguf"),
                receipt_file=Path("fake-receipt.json"),
            )

        self.assertEqual(report["cases_total"], 4)
        self.assertEqual(report["sources"], expected)
        self.assertEqual(len(report["observed"]["matched_base"]), 4)
        self.assertEqual(len(report["observed"]["candidate_v5"]), 4)
        self.assertTrue(report["no_expert_labels"])
        self.assertIsNone(report["real_market_book_accuracy"])
        self.assertFalse(report["research_execution_allowed"])


if __name__ == "__main__":
    unittest.main()
