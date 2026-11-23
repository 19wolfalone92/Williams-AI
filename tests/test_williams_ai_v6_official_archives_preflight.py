"""Offline, deterministic QEP regression for the 200-source pinned pilot."""
from __future__ import annotations

from datetime import date
from pathlib import Path
import unittest

from scripts.ai_v6_official_archives_preflight import (
    load_pinned_plan, required_sources, probe_one, evaluate,
)

ROOT = Path(__file__).resolve().parent.parent
PLAN = ROOT / "research" / "williams_v6_prospective_chart_plan_20261115_22.json"


class V6ArchiveInventoryTests(unittest.TestCase):
    def setUp(self):
        self.plan = load_pinned_plan(PLAN)
        self.rows = required_sources(self.plan)

    def test_plan_is_pinned_and_200_sources_are_exact(self):
        self.assertEqual(len(self.plan["cases"]), 32)
        self.assertEqual(len(self.rows), 200)
        self.assertEqual(len({r["checksum_url"] for r in self.rows}), 200)
        self.assertEqual(self.rows[0]["symbol"], "BTCUSDT")
        self.assertEqual(self.rows[0]["utc_date"], "2026-10-03")
        self.assertEqual(self.rows[-1]["utc_date"], "2026-11-21")
        self.assertTrue(all(r["zip_url"] + ".CHECKSUM" == r["checksum_url"]
                            for r in self.rows))

    def test_plan_tamper_is_rejected_before_network(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            changed = Path(tmp) / "plan.json"
            changed.write_bytes(PLAN.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "UNPINNED_PLAN"):
                load_pinned_plan(changed)

    def test_404_aborts_all_windows_with_exact_symbol_and_date(self):
        def get(sym: str, day: date):
            if sym == "BTCUSDT" and day == date(2026, 10, 3):
                raise ValueError("OFFICIAL_BINANCE_H1_ARCHIVE_HTTP_FAILURE "
                                 "status=404 url=https://data.binance.vision/...")
            return [], {
                "source": f"https://data.binance.vision/data/futures/um/daily/klines/"
                          f"{sym}/1h/{sym}-1h-{day}.zip",
                "official_checksum_verified": True, "H1_count": 24,
                "sha256": "a" * 64,
            }
        def forbid(*args):
            raise AssertionError("NO_1024_WINDOW_AFTER_BLOCKED_SOURCE")
        result = evaluate(self.plan, self.rows, fetch=get,
                          verify_windows=forbid)
        self.assertFalse(result["pass"])
        self.assertEqual(result["blocked_count"], 1)
        self.assertEqual(result["verified_1024_h1_windows"], 0)
        self.assertEqual(result["actual_source_probes"], 200)
        self.assertIn("BTCUSDT", result["failures"][0]["checksum_url"])
        self.assertIn("2026-10-03", result["failures"][0]["checksum_url"])

    def test_provenance_mismatch_rejects_even_200(self):
        def wrong(sym: str, day: date):
            return [], {"source": "https://other.example/data.zip",
                        "official_checksum_verified": True, "H1_count": 24}
        result = probe_one(self.rows[0], fetch=wrong)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertIn("RECEIPT_MISMATCH", result["reason"])

    def test_all_sources_and_32_windows_stubbed_contract(self):
        def verified(sym: str, day: date):
            return [], {"source": (
                f"https://data.binance.vision/data/futures/um/daily/klines/"
                f"{sym}/1h/{sym}-1h-{day}.zip"),
                "official_checksum_verified": True, "H1_count": 24,
                "sha256": "a" * 64}
        def windows(sym, start, end):
            self.assertEqual(end + 1 - start, 1024 * 3_600_000)
            return [None] * 1024, {"official_checksum_verified_all": True}
        result = evaluate(self.plan, self.rows, fetch=verified,
                          verify_windows=windows)
        self.assertTrue(result["pass"])
        self.assertEqual(result["verified_daily_archives"], 200)
        self.assertEqual(result["verified_1024_h1_windows"], 32)
        self.assertFalse(result["training_started"])
        self.assertFalse(result["independent_expert_review_done"])

    def test_corrupt_zip_exception_is_diagnostic_not_silent(self):
        from zipfile import BadZipFile
        def corrupt(sym, day):
            raise BadZipFile("not a zip")
        outcome = probe_one(self.rows[0], fetch=corrupt)
        self.assertEqual(outcome["status"], "BLOCKED")
        self.assertEqual(outcome["error_type"], "BadZipFile")


if __name__ == "__main__":
    unittest.main()
