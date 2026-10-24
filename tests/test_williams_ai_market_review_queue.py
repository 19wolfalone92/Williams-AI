"""Frozen pending tasks cannot launder fabricated labels into training."""
import copy
from datetime import datetime, timedelta
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from test_williams_ai_chart_supervision import context
from williams_ai.chart_supervision import market_digest
from williams_ai.market_review_queue import build_market_review_queue, write_pending_market_queue
from williams_ai.private_lora import checked_private_training_data
from williams_ai.private_review_queue import source_rules_by_split


def historical(symbol):
    ctx = context(symbol)
    first = datetime.fromisoformat(ctx["bars"][0]["close_time"])
    template = ctx["bars"][0]
    ctx["bars"] = [{**template, "close_time": (first + timedelta(hours=i)).isoformat()}
                   for i in range(500)]
    ctx["cutoff"] = ctx["bars"][-1]["close_time"]
    ctx["market_sha256"] = market_digest(ctx)
    return ctx


class MarketQueueTests(unittest.TestCase):
    def setUp(self):
        self.contexts = [historical(s) for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT")]
        self.parts = {"BTCUSDT": "train", "ETHUSDT": "validation", "SOLUSDT": "test"}

    def test_whole_symbols_and_rules_have_one_partition_with_all_four_books(self):
        out = build_market_review_queue(self.contexts, symbol_partitions=self.parts)
        self.assertEqual(out["manifest"]["cases"], 24)
        self.assertEqual(out["manifest"]["labels_completed"], 0)
        rules = source_rules_by_split()
        for row in out["lessons"]:
            self.assertIn(row["rule_id"], {r["id"] for r in rules[row["split"]][row["edition"]]})
            self.assertEqual(row["split"], self.parts[row["market_context"]["symbol"]])
            self.assertEqual(row["answer"], "")
            self.assertEqual(row["independent_reviews"], [])
            self.assertEqual(row["market_context"]["bars"][-1]["close_time"], row["market_context"]["cutoff"])
        for split in self.parts.values():
            self.assertEqual(set(out["manifest"]["chart_editions_by_split"][split]),
                             {"TC1", "NTD", "TC2", "COURSE"})

    def test_pending_queue_cannot_train_and_existing_file_cannot_be_replaced(self):
        out = build_market_review_queue(self.contexts, symbol_partitions=self.parts)
        with TemporaryDirectory() as folder:
            path = Path(folder) / "queue.jsonl"
            write_pending_market_queue(out, path)
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "UNREVIEWED_SOURCE"):
                checked_private_training_data(path)
            with self.assertRaises(FileExistsError):
                write_pending_market_queue(out, path)
            self.assertEqual(path.read_bytes(), before)

    def test_selection_does_not_read_future_prices_or_engine_labels(self):
        baseline = build_market_review_queue(self.contexts, symbol_partitions=self.parts)
        # Final capture date is always outside the selected reviewer dates.
        changed = copy.deepcopy(self.contexts)
        for ctx in changed:
            ctx["bars"][-1]["high"] = 99999.0
            ctx["market_sha256"] = market_digest(ctx)
        after = build_market_review_queue(changed, symbol_partitions=self.parts)
        self.assertEqual(baseline["manifest"]["queue_sha256"], after["manifest"]["queue_sha256"])
        self.assertNotIn("predicted_structure", json.dumps(after["lessons"]))

    def test_duplicate_symbols_wrong_partition_and_mixed_venues_rejected(self):
        for changes in ("duplicate", "partitions", "venue"):
            contexts = copy.deepcopy(self.contexts)
            parts = dict(self.parts)
            if changes == "duplicate":
                contexts[1] = copy.deepcopy(contexts[0])
            elif changes == "partitions":
                parts["OTHERUSDT"] = "train"
            else:
                contexts[0]["venue"] = "SPOT"
                contexts[0]["market_sha256"] = market_digest(contexts[0])
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                build_market_review_queue(contexts, symbol_partitions=parts)

    def test_tampered_capture_is_rejected_before_queue_generation(self):
        self.contexts[0]["bars"][50]["close"] += 0.01
        with self.assertRaisesRegex(ValueError, "CHANGED_AFTER_REVIEW"):
            build_market_review_queue(self.contexts, symbol_partitions=self.parts)


if __name__ == "__main__":
    unittest.main()
