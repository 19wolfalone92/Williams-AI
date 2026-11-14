"""Regressions for strict non-scoring 4-book rationale auditing; no model/network."""
import copy
import unittest
from williams_ai.v6_rationale_diagnostic import audit


def sample():
    books = ("TC1", "NTD", "TC2", "COURSE")
    families = ("ENTRY", "ENTRY", "EXIT", "WAVE")
    assets = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
    sources = [{"symbol": assets[i], "edition": ed,
                "chart_sha256": f"{i+1:x}" * 64} for i, ed in enumerate(books)]
    rows = []
    for i, ed in enumerate(books):
        rows.append({"id": "RETRO-%s-%s" % (ed, assets[i]),
                     "symbol": assets[i], "edition": ed, "family": families[i],
                     "cutoff_utc": "2026-11-01T00:00:00Z",
                     "chart_sha256": sources[i]["chart_sha256"],
                     "prompt_sha256": f"{i+5:x}" * 64,
                     "retrieved_rule_ids": [ed + ".R1"],
                     "source_missing": False, "finish_reason": "stop",
                     "decision": ("HOLD" if ed == "TC2" else
                        "WAVE_HYPOTHESIS" if ed == "COURSE" else "ABSTAIN")})
    candidate = copy.deepcopy(rows)
    candidate[2]["decision"] = "INSUFFICIENT_DATA"
    return {"schema": "WILLIAMS_V6_IMMEDIATE_REAL_RETRO_4BOOK_CPU_SMOKE_V1",
            "cases_total": 4, "no_expert_labels": True,
            "real_market_book_accuracy": None,
            "research_execution_allowed": False, "gpu_training_allowed": False,
            "protocol_sha256": "a"*64, "base_gguf_sha256": "b"*64,
            "v5_original_raw_adapter_sha256_attested": "c"*64,
            "v5_converted_lora_gguf_sha256": "d"*64,
            "sources": sources,
            "observed": {"matched_base": rows, "candidate_v5": candidate}}


class TestV6ReasonReceipt(unittest.TestCase):
    def test_base_vs_v5_without_reason_has_zero_verified_explanations(self):
        result = audit(sample())
        self.assertEqual(len(result["changed_decisions"]), 1)
        self.assertEqual(result["changed_decisions"][0]["id"], "RETRO-TC2-SOLUSDT")
        self.assertEqual(result["reason_text_available"], {
            "matched_base": 0, "candidate_v5": 0})
        self.assertIsNone(result["independent_accuracy"])
        self.assertFalse(result["trading_allowed"])

    def test_full_reason_still_is_not_expert_gold(self):
        raw = sample()
        for name in ("matched_base", "candidate_v5"):
            for row in raw["observed"][name]:
                row["model_reason"] = "Rule %s.R1; p. 42 (unverified)." % row["edition"]
        result = audit(raw)
        self.assertEqual(result["reason_text_available"], {
            "matched_base": 4, "candidate_v5": 4})
        self.assertFalse(result["book_reasoning_proven"])
        self.assertEqual(result["checks"]["matched_base"][2]["H_explanation_quality"]["status"],
                         "TEXT_PRESENT_NOT_VALIDATED_AS_BOOK_REASONING")

    def test_reject_mismatched_prompt(self):
        raw = sample()
        raw["observed"]["candidate_v5"][2]["prompt_sha256"] = "e"*64
        with self.assertRaisesRegex(ValueError, "DIFFERENT_INPUTS_BETWEEN_MODELS"):
            audit(raw)

    def test_reject_cross_book_rag(self):
        raw = sample()
        raw["observed"]["matched_base"][2]["retrieved_rule_ids"] = ["TC1.R1"]
        with self.assertRaisesRegex(ValueError, "CROSS_EDITION_RAG"):
            audit(raw)

    def test_reject_false_expert_accuracy(self):
        raw = sample()
        raw["real_market_book_accuracy"] = 1
        with self.assertRaisesRegex(ValueError, "FALSE_EXPERT_OR_TRADE_CLAIM"):
            audit(raw)

    def test_detect_future_timestamp_in_model_text(self):
        raw = sample()
        raw["observed"]["matched_base"][2]["model_reason"] = \
            "Reason uses 2026-11-02T00:00:00Z (future bar)."
        result = audit(raw)
        self.assertEqual(result["checks"]["matched_base"][2]["C_causal_consistency"],
                         "FAIL_FUTURE_TIMESTAMP_IN_EXPLANATION")

    def test_truncated_response_cannot_pass_as_signal(self):
        raw = sample()
        raw["observed"]["matched_base"][2]["finish_reason"] = "length"
        with self.assertRaisesRegex(ValueError, "NONSTOP_AS_DECISION"):
            audit(raw)


if __name__ == "__main__":
    unittest.main()
