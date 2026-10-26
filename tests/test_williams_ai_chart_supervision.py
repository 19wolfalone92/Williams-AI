"""Adversarial supervision tests, never fake expert labels for real training."""
import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from williams_ai.chart_supervision import (
    SCHEMA, checked_chart_context, checked_chart_lesson, chart_prompt_observations,
    check_market_partitions, lesson_commitment, market_digest, require_market_supervision,
)
from williams_ai.private_lora import checked_private_training_data, _tokenized, train_private_lora
from test_williams_ai_private_lora import candidates


def context(symbol="ETHUSDT", shift=0):
    start = datetime(2026, 4, 1, tzinfo=timezone.utc) + timedelta(days=shift)
    bars = []
    for i in range(220):
        at = start + timedelta(hours=i + 1, milliseconds=-1)
        bars.append({"close_time": at.isoformat(), "open": 100 + i / 10,
                     "high": 102 + i / 10, "low": 99 + i / 10,
                     "close": 101 + i / 10, "volume": 50.0, "closed": True})
    result = {"schema": SCHEMA, "symbol": symbol, "venue": "FUTURES",
              "timeframe": "H1", "tick_size": 0.01,
              "cutoff": bars[-1]["close_time"], "bars": bars}
    result["market_sha256"] = market_digest(result)
    return result


def chart_case(item=None, symbol="ETHUSDT"):
    row = copy.deepcopy(item or candidates()[0])
    row["market_context"] = context(symbol)
    row["independent_reviews"] = []
    seal = lesson_commitment(row)
    row["independent_reviews"] = [
        {"reviewer_id": who, "reviewed_at": "2026-04-20T00:00:00+00:00",
         "lesson_sha256": seal} for who in ("expert-001", "expert-002")]
    return row


class ChartSupervisionTests(unittest.TestCase):
    def test_exact_double_approval_and_bounded_observations(self):
        row = chart_case()
        checked_chart_lesson(row)
        view = chart_prompt_observations(row["market_context"])
        self.assertEqual(view["closed_bars"], 220)
        self.assertEqual(len(view["tail"]), 16)
        self.assertNotIn(row["answer"], json.dumps(view))
        self.assertNotIn("predicted_structure", view)
        self.assertNotIn("setup", view)

    def test_answer_question_rule_or_partition_change_invalidates_approvals(self):
        for field in ("answer", "question", "rule_id", "split"):
            row = chart_case()
            row[field] += " altered"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "STALE_CHART_APPROVAL"):
                checked_chart_lesson(row)

    def test_single_reviewer_cannot_be_two_experts(self):
        row = chart_case()
        row["independent_reviews"][1]["reviewer_id"] = "expert-001"
        with self.assertRaisesRegex(ValueError, "DISTINCT_OR_STALE"):
            checked_chart_lesson(row)
        row["independent_reviews"].pop()
        with self.assertRaisesRegex(ValueError, "TWO_INDEPENDENT"):
            checked_chart_lesson(row)

    def test_review_must_follow_market_data(self):
        row = chart_case()
        row["independent_reviews"][0]["reviewed_at"] = "2026-03-01T00:00:00Z"
        with self.assertRaisesRegex(ValueError, "PREDATES_VISIBLE"):
            checked_chart_lesson(row)

    def test_future_bar_is_rejected_even_with_new_hash(self):
        ctx = context()
        ctx["cutoff"] = ctx["bars"][-2]["close_time"]
        ctx["market_sha256"] = market_digest(ctx)
        with self.assertRaisesRegex(ValueError, "NO_LOOKAHEAD"):
            checked_chart_context(ctx)

    def test_teacher_labels_and_irregular_candles_never_enter_context(self):
        ctx = context()
        ctx["bars"][0]["future_wave"] = "W3"
        ctx["market_sha256"] = market_digest(ctx)
        with self.assertRaisesRegex(ValueError, "NO_LABEL_FIELDS"):
            checked_chart_context(ctx)
        ctx = context()
        ctx["bars"][10]["close_time"] = (
            datetime.fromisoformat(ctx["bars"][10]["close_time"]) + timedelta(minutes=1)).isoformat()
        ctx["market_sha256"] = market_digest(ctx)
        with self.assertRaisesRegex(ValueError, "CONTIGUOUS_ALIGNED"):
            checked_chart_context(ctx)

    def test_changed_candle_invalidates_approval_even_if_resealed_market(self):
        row = chart_case()
        row["market_context"]["bars"][0]["volume"] += 1
        with self.assertRaisesRegex(ValueError, "CHANGED_AFTER_REVIEW"):
            checked_chart_lesson(row)
        row["market_context"]["market_sha256"] = market_digest(row["market_context"])
        with self.assertRaisesRegex(ValueError, "STALE_CHART_APPROVAL"):
            checked_chart_lesson(row)

    def test_overlapping_sliding_windows_and_timeframes_cannot_cross_splits(self):
        first = checked_chart_lesson(chart_case())
        other = {**first, "timeframe": "H4", "split": "test",
                 "start": first["start"] + timedelta(hours=2)}
        with self.assertRaisesRegex(ValueError, "OVERLAPPING_MARKET_HISTORY"):
            check_market_partitions([first, other])
        other["start"] = first["end"] + timedelta(hours=12)
        other["end"] = other["start"] + timedelta(days=3)
        with self.assertRaisesRegex(ValueError, "OVERLAPPING_MARKET_HISTORY"):
            check_market_partitions([first, other])
        other["start"] = first["end"] + timedelta(hours=25)
        self.assertTrue(check_market_partitions([first, other])[
            "cross_timeframe_overlap_checked"])

    def test_spot_futures_and_quote_aliases_do_not_leak_a_market_history(self):
        first = checked_chart_lesson(chart_case())
        for alias in ({"symbol": "ETHUSDT", "venue": "SPOT"},
                      {"symbol": "ETHUSDC", "venue": "SPOT"},
                      {"symbol": "ETHBTC", "venue": "FUTURES"}):
            second = {**first, **alias, "split": "test"}
            with self.subTest(alias=alias), self.assertRaisesRegex(
                    ValueError, "OVERLAPPING_MARKET_HISTORY"):
                check_market_partitions([first, second])

    def test_rephrased_same_candles_are_not_independent_market_experience(self):
        symbols = {"train": "ETHUSDT", "validation": "SOLUSDT", "test": "BTCUSDT"}
        rows = [chart_case(row, symbols[row["split"]]) for row in candidates()]
        with TemporaryDirectory() as folder:
            path = Path(folder) / "reviewed.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows))
            audited = checked_private_training_data(path)
        self.assertEqual(audited["chart_examples_by_split"],
                         {"train": 64, "validation": 18, "test": 18})
        self.assertEqual(audited["unique_market_windows_by_split"],
                         {"train": 1, "validation": 1, "test": 1})
        with self.assertRaisesRegex(ValueError, "INDEPENDENT_MARKET_WINDOWS"):
            require_market_supervision(audited)

    def test_multimarket_multitimeframe_distinct_windows_are_required(self):
        split_assets = {"train": ("ETHUSDT", "BTCUSDT", "SOLUSDT"),
                        "validation": ("XRPUSDT", "LTCUSDT"),
                        "test": ("ADAUSDT", "BNBUSDT")}
        editions = ("TC1", "NTD", "TC2", "COURSE")
        timeframes = ("H1", "H4", "D1")
        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        rows = []
        for split, assets in split_assets.items():
            for i in range(24 if split == "train" else 8):
                at = start + timedelta(days=20 * i)
                rows.append({"split": split, "edition": editions[i % 4],
                             "venue": "FUTURES", "symbol": assets[i % len(assets)],
                             "timeframe": timeframes[i % (3 if split == "train" else 2)],
                             "start": at, "end": at + timedelta(days=9)})
        audited = check_market_partitions(rows)
        self.assertEqual(audited["unique_market_windows_by_split"],
                         {"train": 24, "validation": 8, "test": 8})
        self.assertTrue(audited["cross_venue_and_quote_alias_overlap_checked"])
        require_market_supervision(audited)
        for row in rows:
            row["timeframe"] = "H1"
        with self.assertRaisesRegex(ValueError, "MULTI_TIMEFRAME"):
            require_market_supervision(check_market_partitions(rows))

    def test_full_loader_reports_chart_splits_and_never_returns_test_answers_to_trainer(self):
        symbols = {"train": "ETHUSDT", "validation": "SOLUSDT", "test": "BTCUSDT"}
        rows = [chart_case(row, symbols[row["split"]]) for row in candidates()]
        with TemporaryDirectory() as folder:
            path = Path(folder) / "reviewed.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows))
            audit = checked_private_training_data(path)
        self.assertEqual(audit["chart_examples_by_split"], {"train": 64, "validation": 18, "test": 18})
        self.assertEqual(set(audit["items_for_local_fit"]), {"train", "validation"})

    def test_book_prose_alone_cannot_launch_market_training(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "reviewed.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in candidates()))
            with self.assertRaisesRegex(ValueError, "MARKET_SUPERVISION_REQUIRED"):
                train_private_lora(data=path, model_dir=Path(folder) / "model",
                                   output_dir=Path(folder) / "adapter")

    def test_real_training_turn_template_must_preserve_prompt(self):
        class BadTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return [1, 2, 3] if kwargs["add_generation_prompt"] else [1, 5, 6, 7]
        with self.assertRaisesRegex(ValueError, "PRESERVE_PROMPT_PREFIX"):
            _tokenized(candidates()[:1], BadTokenizer(), max_tokens=128)

    def test_chart_features_come_from_approved_candles_in_masked_prompt(self):
        seen = []
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                seen.append(copy.deepcopy(messages))
                self_check = messages[1]["content"]
                if "market_sha256" not in self_check:
                    raise AssertionError("missing exact approved chart input")
                return [1, 2] if kwargs["add_generation_prompt"] else [1, 2, 3, 4]
        row = chart_case()
        result = _tokenized([row], Tokenizer(), max_tokens=4096)[0]
        self.assertEqual(result["labels"], [-100, -100, 3, 4])
        self.assertNotIn(row["answer"], seen[0][1]["content"])
        self.assertEqual(seen[1][-1], {"role": "assistant", "content": row["answer"]})


if __name__ == "__main__":
    unittest.main()
