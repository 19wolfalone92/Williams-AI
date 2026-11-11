from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import unittest

from scripts.ai_verify_historical_rule_curriculum import independent_key, verify_market_observations
from williams_ai.historical_rule_curriculum import (FAMILIES, INTERVALS, _window_rows,
    cases_from_pack, check_partitions, checked_pack, digest, load_pack, score)


ROOT = Path("examples/historical_rule_20261029")


class HistoricalCurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packs = {s: load_pack(ROOT/(s+".json.gz")) for s in ("train", "validation", "test")}
        cls.rows = cases_from_pack(cls.packs["validation"])

    def test_real_history_is_closed_chronological_and_cross_timeframe_purged(self):
        self.assertEqual(set(check_partitions(self.packs)), {"train", "validation", "test"})
        self.assertEqual(sum(len(d["bars"]) for p in self.packs.values() for d in p["datasets"]), 60000)
        mutated = deepcopy(self.packs["train"]); mutated["datasets"][0]["bars"][3][0] += 1
        with self.assertRaises(ValueError): checked_pack(mutated)
        mutated = deepcopy(self.packs["train"]); mutated["split"] = "validation"
        with self.assertRaises(ValueError): checked_pack(mutated)
        mutated = deepcopy(self.packs["train"]); mutated["datasets"][0]["bars"][3][2] = "NaN"
        with self.assertRaises(ValueError): checked_pack(mutated)

    def test_future_candles_cannot_change_a_past_lesson(self):
        dataset = self.packs["train"]["datasets"][0]
        before = list(_window_rows(dataset, 239))
        changed = deepcopy(dataset)
        for row in changed["bars"][240:]:
            row[1:5] = ["1", "9000000", "0.01", "2"]
        self.assertEqual(before, list(_window_rows(changed, 239)))

    def test_visible_answers_recompute_without_reading_the_authored_answer(self):
        self.assertEqual(set(r["rule_id"] for r in self.rows), set(FAMILIES))
        datasets = {(d["symbol"], INTERVALS[d["interval"]][2]): d for d in self.packs["validation"]["datasets"]}
        for row in self.rows:
            self.assertEqual(independent_key(row), json.loads(row["answer"]))
            verify_market_observations(row, datasets[(row["symbol"], row["timeframe"])])
        fake = {**self.rows[0], "answer": '{"fabricated":true}'}
        self.assertEqual(independent_key(fake), independent_key(self.rows[0]))

    def test_ac_ready_and_not_ready_and_both_sides_are_present(self):
        rows = cases_from_pack(self.packs["train"])
        ac = [r for r in rows if r["rule_id"] == "NTD.AC"]
        self.assertEqual({json.loads(r["answer"])["signal_ready"] for r in ac}, {True, False})
        self.assertEqual({json.loads(r["answer"])["required_bars"] for r in ac}, {2, 3})
        self.assertEqual({r["stratum"].split(":")[0] for r in ac}, {"LONG", "SHORT"})

    def test_exact_completed_answers_required_and_constant_fields_are_reported(self):
        answers = [{"case_id": r["case_id"], "response": r["answer"], "thinking_complete": True, "response_time_limit_reached": False} for r in self.rows]
        self.assertEqual(score(self.rows, answers)["correct"], len(self.rows))
        truncated = deepcopy(answers); truncated[0]["thinking_complete"] = False
        self.assertEqual(score(self.rows, truncated)["correct"], len(self.rows)-1)
        with self.assertRaises(ValueError): score(self.rows, answers+[answers[0]])
        altered = deepcopy(answers)
        for response, row in zip(altered, self.rows):
            if row["rule_id"] == "NTD.AC":
                answer = json.loads(response["response"]); answer["signal_ready"] = True
                response["response"] = json.dumps(answer)
        self.assertIn("signal_ready", score(self.rows, altered)["constant_fields_despite_varied_expected_values"]["NTD.AC"])


if __name__ == "__main__": unittest.main()
