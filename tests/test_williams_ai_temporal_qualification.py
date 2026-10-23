"""Research-only temporal learning: no Jan label leaking into Feb."""
from __future__ import annotations

import unittest
from datetime import datetime,timezone,timedelta

from williams_ai.paper_learning import ResearchSample
from williams_ai.temporal_qualification import (
    split_calendar_samples,qualify_verified_training,
)


def sample(start,finish):
    return ResearchSample(
        decision_index=1,resolved_index=2,decision_at=start,
        resolved_at=finish,side="LONG",features=(1.,0.,0.,0.,0.,.1),
        net_return=.002,stopped=False,
    )


class RealWalkForwardCalendarTests(unittest.TestCase):
    def test_never_train_on_label_revealed_only_after_february_begins(self):
        jan="2025-01-30T22:59:59+00:00"
        jan31="2025-01-31T19:59:59+00:00"
        feb="2025-02-01T07:59:59+00:00"
        feb2="2025-02-02T07:59:59+00:00"
        feb28="2025-02-28T18:59:59+00:00"
        mar="2025-03-01T05:59:59+00:00"
        training,valid=split_calendar_samples([
            sample(jan,jan31),
            sample(jan31,feb),  # purged: training label from the future
            sample(feb,feb2),
            sample(feb28,mar),  # purged: February label from March
        ])
        self.assertEqual(len(training),1)
        self.assertEqual(len(valid),1)
        self.assertEqual(training[0].decision_at,jan)
        self.assertEqual(valid[0].decision_at,feb)

    def test_decision_timestamp_must_be_offset_aware(self):
        with self.assertRaises(ValueError):
            split_calendar_samples([sample(
                "2025-01-20T10:00:00","2025-01-20T20:00:00")])
        with self.assertRaises(ValueError):
            split_calendar_samples([sample(
                "2025-01-20T10:00:00+00:00","2025-01-20T09:00:00+00:00")])

    @staticmethod
    def hist(symbol_offset:float):
        start=datetime(2025,1,1,tzinfo=timezone.utc)
        rows=[]
        # 31 Jan + 28 Feb: 1,416 CLOSED H1 bars per source.
        for i in range(1416):
            # Distinct price paths, no perfectly monotone synthetic trend
            phase=(i%19-9)*.045+(i%107-53)*.012
            price=120+symbol_offset+(i%170-85)*.009+phase
            rows.append({
                "close_time":(start+timedelta(hours=i+1)
                              -timedelta(milliseconds=1)).isoformat(),
                "open":price-.01,
                "high":price+.2,
                "low":price-.2,
                "close":price,
                "volume":1+i%13,
                "closed":True,
            })
        return rows

    def test_twenty_market_validation_rejects_trading_even_when_brier_is_good(self):
        # Fixtures are artificial: this test proves the Jan->Feb splits,
        # not a claim about real Binance training performance.
        markets={f"COIN{k}USDT":self.hist(k*2.) for k in range(20)}
        report=qualify_verified_training(
            markets,min_train=300,min_validation=300,
        )
        self.assertEqual(report["tested_symbols"],20)
        self.assertGreater(report["training_samples"],300)
        self.assertGreater(report["validation_samples"],300)
        self.assertTrue(report["purged_cross_month_labels"])
        self.assertEqual(report["training_month"],"2025-01")
        self.assertEqual(report["independent_validation_month"],"2025-02")
        self.assertFalse(report["promotion_to_execution_allowed"])
        self.assertFalse(report["auto_adopted_model"])
        self.assertFalse(report["neural_GGUF_weights_changed"])
        self.assertIn(report["qualification"],{
            "MODEL_PASSED_STATISTICAL_RESEARCH_GATE_NOT_TRADING",
            "REJECT_RESEARCH_MODEL_NO_PROVEN_EDGE",
        })
        self.assertEqual(len(report["per_symbol"]),20)
        self.assertTrue(all(
            item["jan_train"]>0 and item["feb_validation"]>0
            for item in report["per_symbol"].values()
        ))

    def test_no_loose_or_data_tuned_gate(self):
        with self.assertRaisesRegex(ValueError,"at least 20"):
            qualify_verified_training({"BTCUSDT":self.hist(0)})
        with self.assertRaisesRegex(ValueError,"threshold"):
            qualify_verified_training(
                {f"COIN{k}USDT":self.hist(k) for k in range(20)},
                required_symbol_fraction=.01,
            )


if __name__=="__main__":
    unittest.main()
