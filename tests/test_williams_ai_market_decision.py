from collections import Counter
from decimal import Decimal
import gzip
import json
import unittest

from scripts.ai_run_market_decision_kaggle import DATA, frozen, worker_source
from scripts.ai_verify_market_decision import _observed_safety
from williams_ai.causal_timeframes import aggregate_closed_h1
from williams_ai.chart_reader import indicators, validate_closed_bars
from williams_ai.deep_book_research import EDITIONS
from williams_ai.historical_rule_curriculum import iso, number, score
from williams_ai.joint_rule_curriculum import selection
from williams_ai.market_decision_curriculum import _campaign
from williams_ai.market_decision_gpu import RUN
from williams_ai.wave_evidence import alternating_impulse_skeleton


class MarketDecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.groups, cls.design = frozen()

    def test_context(self):
        cases = self.groups["test"]["context"]
        self.assertEqual(len(cases), 72)
        self.assertEqual({json.loads(row["answer"])["ac_side"] for row in cases}, {"LONG", "SHORT", "NONE"})
        self.assertEqual({json.loads(row["answer"])["ntd_next"] for row in cases},
                         {"WAIT_FIRST_FRACTAL", "NO_AC_SETUP", "REVIEW_ADD"})
        data = self.design["curriculum"]["market_history"]
        for early, late in zip(("train", "validation"), ("validation", "test")):
            self.assertLess(max(x["last_close"] for x in data[early].values()),
                            min(x["first_close"] for x in data[late].values()))
        # Reusing market symbols must not make chart inputs identical across sets.
        windows = {part: {row["window"]["prefix_sha256"] for row in subset["context"]}
                   for part, subset in self.groups.items()}
        self.assertEqual(sum(map(len, windows.values())), len(set.union(*windows.values())))
        history = json.loads(gzip.decompress((DATA / "test.json.gz").read_bytes()))
        for row in cases[::12]:
            cutoff = row["window"]["cutoff"]
            prefix = [{"close_time": iso(r[0]), "open": float(r[1]), "high": float(r[2]),
                       "low": float(r[3]), "close": float(r[4]), "volume": float(r[5]), "closed": True}
                      for r in history[row["window"]["symbol"]] if iso(r[0]) <= cutoff]
            for tf in ("H1", "H4", "D1"):
                closed = prefix if tf == "H1" else aggregate_closed_h1(prefix, tf)
                bars, gaps = validate_closed_bars(closed, tf, cutoff)
                self.assertEqual(gaps, 0)
                expected = {k: number(v[-1]) for k, v in indicators(bars).items()}
                expected["close"] = number(bars[-1].close)
                self.assertEqual(expected, row["operands"][tf.lower()])

    def test_waves(self):
        cases = self.groups["test"]["waves"]
        counts = Counter(json.loads(row["answer"])["compatible"] for row in cases)
        self.assertTrue(all(counts[k] >= 16 for k in ("A", "B", "BOTH", "NONE")))
        for row in cases:
            a, b = (row["operands"][k] for k in ("A", "B"))
            self.assertEqual(a["bars_ago"][-1], b["bars_ago"][-1])
            self.assertGreaterEqual(a["bars_ago"][-1], 2)
            key = json.loads(row["answer"])
            self.assertIs(key["wave_confirmed"], False)
            for k, alt in (("a", a), ("b", b)):
                pivot = [{"price": price, "type": "L" if (i % 2 == 0) == alt["first_is_low"] else "H",
                          "time": str(at), "index": 160 - at, "confirmed_at_index": 162 - at}
                         for i, (price, at) in enumerate(zip(alt["prices_0_to_5"], alt["bars_ago"]))]
                compatible = any(x["pattern"] == "IMPULSE_SKELETON_POSSIBLE"
                                 for x in alternating_impulse_skeleton(pivot))
                self.assertEqual(compatible, key["reject_" + k] == "NONE")

    def test_campaign(self):
        cases = self.groups["test"]["campaign"]
        labels = [(row, json.loads(row["answer"])) for row in cases]
        for action in ("PLAN_STOP_ENTRY", "PLAN_ADD_STOP", "WATCH_FRACTAL_BREAK"):
            self.assertEqual({answer["side"] for _, answer in labels if answer["decision"] == action},
                             {"LONG", "SHORT"})
        for row, answer in labels:
            values = row["operands"]
            journal = values["journal"]
            if answer["decision"] == "PLAN_STOP_ENTRY":
                direction = 1 if answer["side"] == "LONG" else -1
                self.assertGreater(direction * (Decimal(str(answer["trigger"])) - Decimal(str(answer["stop"]))), 0)
                patched = {**values, "journal": {**journal, "angulation_review": False}}
                self.assertEqual(_campaign(patched)["decision"], "WAIT")
            if answer["decision"] == "REVIEW_TIGHTEN":
                direction = 1 if answer["side"] == "LONG" else -1
                self.assertGreater(direction * (Decimal(str(answer["stop"])) - Decimal(journal["stop"])), 0)
            if journal["fills"] and not journal["stop_ack"]:
                self.assertEqual(answer["decision"], "AUDIT_PROTECTION")
            self.assertIs(answer["execution_allowed"], False)

    def test_precision(self):
        cases = self.groups["test"]["precision"]
        self.assertEqual(len(cases), 64)
        self.assertEqual(Counter(row["rule_id"] for row in cases), {"TC1.PSY.PLAN": 32, "NTD.AC": 32})
        risk = next(row for row in cases if row["rule_id"] == "TC1.PSY.PLAN")
        invalid = json.dumps({"quantity": 1e12, "risk_budget": 1, "execution_allowed": False})
        counts = _observed_safety([risk], [{"case_id": risk["case_id"], "response": invalid}], "precision")
        self.assertEqual(counts["oversize_risk"], 1)
        self.assertEqual(set.union(*(set(x) for x in RUN.rank_cohorts)), set(RUN.cohorts))
        files = {n: {"sha256": "0" * 64, "url": "https://example.invalid/file"}
                 for n in ("adapter_config.json", "adapter_model.safetensors")}
        worker = worker_source("0" * 40, b"{}", b"{}", b"{}", files, {})
        compile(worker, "private_worker", "exec")
        self.assertIn("from williams_ai.market_decision_gpu import train", worker)
        self.assertIn("CLOSED_CHART_20261107.zip", worker)
        self.assertIn("is_dir()", worker)
        dev = {cohort: score(rows, [{"case_id": row["case_id"], "response": row["answer"],
                                    "thinking_complete": True, "response_time_limit_reached": False}
                                   for row in rows])
               for cohort, rows in self.groups["validation"].items()}
        loss = {k: 1. for k in (*EDITIONS, "ALL")}
        result = selection(dev, loss, loss, {"parent_source": loss, "parent_practice": loss},
                           cohorts=RUN.cohorts)
        self.assertEqual(result["macro_accuracy"], 1.)
        self.assertTrue(result["preserved_within_two_percent"])
