"""Deterministic mock tests; the real HTTP GET happens ONLY in Actions smoke."""
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile

from williams_ai.v6_public_chart_collector import BARS, MS
from williams_ai.v6_real_market_smoke import (
    CUTOFF, SYMBOLS, smoke, archive,
)


class RealPublicHistorySmokeContracts(unittest.TestCase):
    def mock_data(self, symbol, start, end):
        self.assertIn(symbol, SYMBOLS)
        self.assertEqual(end, start + BARS * MS - 1)
        base = {"BTCUSDT": 100, "ETHUSDT": 70,
                "SOLUSDT": 35, "XRPUSDT": 2}[symbol]
        result = []
        for i in range(BARS):
            opening = base + i * 0.017
            result.append([
                start + i * MS, str(opening), str(opening + 0.15),
                str(opening - 0.08), str(opening + 0.06),
                str(20 + i % 17), start + (i + 1)*MS - 1,
                "0", 7, "0", "0", "0",
            ])
        return result

    def test_four_symbols_have_real_shape_and_full_causal_daily_context(self):
        report, charts = smoke(fetcher=self.mock_data,
                               now=datetime(2026,11,13,tzinfo=timezone.utc))
        self.assertEqual(set(report["results"]), set(SYMBOLS))
        self.assertEqual(len(charts), 4)
        self.assertEqual(report["purpose"],
                         "NON_HOLDOUT_PUBLIC_MARKET_TRANSPORT_SMOKE_NOT_EXPERT_EXAM")
        self.assertFalse(report["four_book_expert_gold"])
        self.assertFalse(report["new_unseen_heldout_market_data"])
        self.assertFalse(report["execution_allowed"])
        for name, row in report["results"].items():
            self.assertEqual(row["completed_H1_bars"], BARS)
            self.assertEqual(row["completed_H4_bars"], 256)
            self.assertGreaterEqual(row["completed_D1_bars"], 40)
            self.assertIsNotNone(row["D1_AC"])
            self.assertFalse(row["book_rule_verified"])
            self.assertFalse(row["expert_label_available"])

    def test_archive_contains_only_unreviewed_public_market_source_data(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "NON_HOLDOUT_PUBLIC_CANDLES.zip"
            report = archive(output, fetcher=self.mock_data,
                             now=datetime(2026,11,13,tzinfo=timezone.utc))
            self.assertTrue(output.is_file())
            with ZipFile(output) as z:
                self.assertEqual(len(z.namelist()), 5)
                result = json.loads(z.read("NON_HOLDOUT_real_market_receipt.json"))
                self.assertEqual(result, report)
                self.assertFalse(result["trading_keys_used"])
                self.assertFalse(result["lora_inference_performed"])
                for member in z.namelist():
                    if member.endswith(".json") and member.startswith("charts/"):
                        self.assertEqual(len(json.loads(z.read(member))["bars"]), BARS)
            with self.assertRaisesRegex(FileExistsError, "NO_OVERWRITE"):
                archive(output, fetcher=self.mock_data,
                        now=datetime(2026,11,13,tzinfo=timezone.utc))

    def test_bad_candle_gap_is_error_not_silent_repair(self):
        def faulty(symbol, start, end):
            bars = self.mock_data(symbol, start, end)
            bars[47][0] += MS
            return bars
        with self.assertRaisesRegex(ValueError, "FUTURE_OR_DISCONTINUOUS_H1"):
            smoke(fetcher=faulty, now=datetime(2026,11,13,tzinfo=timezone.utc))

    def test_future_smoke_cutoff_is_never_allowed(self):
        with self.assertRaisesRegex(ValueError, "CUTOFF_NOT_YET_AVAILABLE"):
            smoke(fetcher=self.mock_data, now=datetime(2026,11,11,tzinfo=timezone.utc))


if __name__ == "__main__":
    unittest.main()
