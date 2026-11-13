"""Forward-plan source collection tests using MOCK kline data only."""
import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile

from williams_ai.v6_public_chart_collector import (
    MS, BARS, check_plan, collect, normalized_chart, build_archive,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "research" / "williams_v6_prospective_chart_plan_20261115_22.json"
AFTER_ALL = datetime(2026, 11, 24, tzinfo=timezone.utc)


def mock_klines(symbol, start_ms, end_ms):
    assert end_ms == start_ms + BARS * MS - 1
    offset = {"BTCUSDT": 100, "ETHUSDT": 80,
              "SOLUSDT": 60, "XRPUSDT": 40}[symbol]
    return [[start_ms + n*MS, str(offset+1), str(offset+2),
             str(offset-1), str(offset+1.5), "12.0",
             start_ms + (n+1)*MS-1, "0", 12, "0", "0", "0"]
            for n in range(BARS)]


class ForwardUnreviewedChartTests(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads(PLAN.read_text(encoding="utf-8"))

    def test_immutable_forward_schedule_32_all_books_and_8_dates(self):
        check_plan(self.plan, now=AFTER_ALL)
        self.assertEqual(len(self.plan["cases"]), 32)
        self.assertEqual(len({x["cutoff_utc"] for x in self.plan["cases"]}), 8)
        self.assertEqual({x["edition"] for x in self.plan["cases"]},
                         {"TC1", "NTD", "TC2", "COURSE"})
        self.assertFalse(self.plan["examination_labels_present"])

    def test_final_cohort_all_1024_h1_bars_are_post_training(self):
        path = (ROOT / "research" /
                "williams_v6_full_forward_expert_plan_20261228_20270104.json")
        future = json.loads(path.read_text(encoding="utf-8"))
        check_plan(future, now=datetime(2027, 1, 5, tzinfo=timezone.utc))
        self.assertEqual(len(future["cases"]), 32)
        for row in future["cases"]:
            cutoff = datetime.fromisoformat(row["cutoff_utc"].replace("Z", "+00:00"))
            earliest = cutoff - timedelta(hours=BARS)
            self.assertGreater(earliest,
                               datetime(2026, 11, 13, tzinfo=timezone.utc))
        with self.assertRaisesRegex(ValueError, "PROSPECTIVE_CANDLE_NOT_YET_CLOSED"):
            check_plan(future, now=datetime(2026, 11, 13, tzinfo=timezone.utc))

    def test_scheduled_cutoffs_not_ready_today(self):
        with self.assertRaisesRegex(ValueError, "PROSPECTIVE_CANDLE_NOT_YET_CLOSED"):
            check_plan(self.plan, now=datetime(2026, 11, 13, tzinfo=timezone.utc))

    def test_mock_data_is_valid_but_not_expert_gold(self):
        cases, receipts, blobs = collect(self.plan, fetcher=mock_klines, now=AFTER_ALL)
        self.assertEqual(len(blobs), 32)
        self.assertEqual(len(receipts["rows"]), 32)
        self.assertTrue(receipts["data_only_unreviewed"])
        self.assertFalse(receipts["chart_labels_authorized"])
        self.assertEqual({x["edition"] for x in cases["cases"]},
                         {"TC1", "NTD", "TC2", "COURSE"})
        self.assertEqual(len(blobs[next(iter(blobs))]), len(
            next(iter(blobs.values()))))

    def test_complete_immutable_candidate_zip_but_never_grade(self):
        with TemporaryDirectory() as tmp:
            archive = Path(tmp) / "unreviewed_charts.zip"
            manifest = build_archive(self.plan, archive, fetcher=mock_klines,
                                     now=AFTER_ALL)
            self.assertEqual(manifest["case_count"], 32)
            self.assertFalse(manifest["training_allowed"])
            with ZipFile(archive) as z:
                self.assertEqual(len(z.namelist()), 35)
                saved = json.loads(z.read("frozen_manifest.json"))
                self.assertEqual(saved, manifest)
                self.assertFalse(json.loads(z.read("provenance.json"))[
                    "chart_labels_authorized"])
            with self.assertRaisesRegex(ValueError, "NO_OVERWRITE"):
                build_archive(self.plan, archive, fetcher=mock_klines,
                              now=AFTER_ALL)

    def test_missing_or_future_bar_fails_closed(self):
        row = self.plan["cases"][0]
        cutoff = datetime.fromisoformat(row["cutoff_utc"].replace("Z", "+00:00"))
        start = int(cutoff.timestamp()*1000) - BARS*MS
        raw = mock_klines(row["symbol"], start, int(cutoff.timestamp()*1000)-1)
        with self.assertRaisesRegex(ValueError, "INCOMPLETE_H1_HISTORY"):
            normalized_chart(row["symbol"], cutoff, raw[:-1])
        corrupt = copy.deepcopy(raw)
        corrupt[-1][0] += MS
        with self.assertRaisesRegex(ValueError, "FUTURE_OR_DISCONTINUOUS_H1"):
            normalized_chart(row["symbol"], cutoff, corrupt)
        corrupt = copy.deepcopy(raw)
        corrupt[0][3] = str(1000.0)
        with self.assertRaisesRegex(ValueError, "INVALID_KLINE_OHLC"):
            normalized_chart(row["symbol"], cutoff, corrupt)

    def test_forbidden_source_or_duplicated_windows_fail(self):
        corrupt = copy.deepcopy(self.plan)
        corrupt["prospective_chart_source"] = "https://example.com/fake"
        with self.assertRaisesRegex(ValueError, "PLAN_PROVENANCE_INVALID"):
            check_plan(corrupt, now=AFTER_ALL)
        corrupt = copy.deepcopy(self.plan)
        corrupt["cases"][1]["symbol"] = corrupt["cases"][0]["symbol"]
        corrupt["cases"][1]["cutoff_utc"] = corrupt["cases"][0]["cutoff_utc"]
        with self.assertRaisesRegex(ValueError, "DUPLICATE_MARKET_WINDOW"):
            check_plan(corrupt, now=AFTER_ALL)


if __name__ == "__main__":
    unittest.main()
