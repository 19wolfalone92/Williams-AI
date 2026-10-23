"""H1 may correct H4 without automatically proving its reversal (and vice versa)."""
import unittest

from williams_ai.market_hierarchy import reconcile_market_hierarchy
from williams_ai.causal_timeframes import top_down_market_evidence
from williams_ai.causal_paper_lab import frozen_markets


class HierarchyCausalityTests(unittest.TestCase):
    def test_opposing_parent_child_keeps_three_alternatives_open(self):
        passport={
            "D1":{"status":"OBSERVED_CLOSED_BARS",
                  "market_context":"UP_CONTEXT"},
            "H4":{"status":"OBSERVED_CLOSED_BARS",
                  "market_context":"DOWN_CONTEXT"},
            "H1":{"status":"OBSERVED_CLOSED_BARS",
                  "market_context":"UP_CONTEXT"},
        }
        a=reconcile_market_hierarchy(passport)
        self.assertEqual(len(a["parent_child_relations"]),2)
        for link in a["parent_child_relations"]:
            self.assertEqual(link["relation"],
                             "COUNTER_DIRECTION_UNVERIFIED_CORRECTION_OR_REVERSAL")
            self.assertEqual(len(link["unconfirmed_alternatives"]),3)
            self.assertFalse(link["wave_degree_confirmed"])
            self.assertFalse(link["may_close_position_or_enter_market"])
        self.assertIsNone(a["trade_action"])
        self.assertFalse(a["execution_allowed"])

    def test_aligned_trend_is_not_automatically_confirmed_w3_or_w5(self):
        all_up={k:{"status":"OBSERVED_CLOSED_BARS","market_context":"UP_CONTEXT"}
                for k in ("D1","H4","H1")}
        r=reconcile_market_hierarchy(all_up)
        for link in r["parent_child_relations"]:
            self.assertEqual(link["relation"],
                             "DIRECTIONALLY_ALIGNED_NOT_WAVE_DEGREE_PROVEN")
            self.assertIn("POSSIBLE_LATE_TREND_EXHAUSTION",
                          link["unconfirmed_alternatives"])
        self.assertFalse(r["parent_wave_degree_confirmed"])

    def test_unknown_parent_not_labeled_as_confirmed_correction(self):
        partial={
            "D1":{"status":"INSUFFICIENT_COMPLETE_BARS",
                  "market_context":"UNKNOWN"},
            "H4":{"status":"OBSERVED_CLOSED_BARS",
                  "market_context":"DOWN_CONTEXT"},
            "H1":{"status":"OBSERVED_CLOSED_BARS",
                  "market_context":"UP_CONTEXT"},
        }
        r=reconcile_market_hierarchy(partial)
        self.assertEqual(r["parent_child_relations"][0]["relation"],
                         "RELATION_UNRESOLVED")
        self.assertEqual(r["parent_child_relations"][1]["relation"],
                         "COUNTER_DIRECTION_UNVERIFIED_CORRECTION_OR_REVERSAL")

    def test_on_real_binance_15_symbols_hierarchy_never_issues_orders(self):
        data=frozen_markets()
        self.assertEqual(len(data),15)
        for symbol,info in data.items():
            with self.subTest(symbol=symbol):
                p=top_down_market_evidence(symbol,info["bars"],
                                            tick_size=info["tick_size"])
                rel=p["conditional_parent_child_market_structure"]
                self.assertEqual(len(rel["parent_child_relations"]),2)
                self.assertFalse(rel["execution_allowed"])
                self.assertIsNone(rel["trade_action"])
                self.assertEqual(p["wave_hierarchy_status"],"UNCONFIRMED_NESTED_DEGREES")
                for x in rel["parent_child_relations"]:
                    self.assertFalse(x["wave_degree_confirmed"])


if __name__=="__main__":
    unittest.main()
