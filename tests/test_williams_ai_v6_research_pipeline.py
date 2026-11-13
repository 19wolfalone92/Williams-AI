"""Read-only v6 Qwen+RAG+algorithm_guard integration and adversarial fixtures."""
from __future__ import annotations
import copy
import unittest

from williams_ai.v6_research_pipeline import review_chart
from williams_ai.book_rules import registry


class FakeResearchModel:
    def __init__(self, report=None, error=None):
        self.report = report
        self.error = error
        self.calls = []

    def propose_chart_reasoning(self, chart, **kwargs):
        self.calls.append((chart, kwargs))
        if self.error:
            raise self.error
        return dict(self.report)


def chart():
    return {"symbol": "BTCUSDT", "as_of": "2026-11-01T12:00:00Z",
            "execution_allowed": False, "gaps": 0, "candidates": [],
            "higher_timeframe_context": {
                "H4": {"closed_through": "2026-11-01T08:00:00Z"},
                "D1": {"closed_through": "2026-11-01T00:00:00Z"}}}


def proposal(*, edition="TC2", rule_ids=None, trade_action=None):
    return {"symbol": "BTCUSDT", "as_of": "2026-11-01T12:00:00Z",
            "edition": edition, "setup": "NO_SETUP", "side": "NONE",
            "entry_state": "NO_SETUP", "rule_ids": rule_ids if rule_ids is not None else [],
            "exit_basis": "NONE", "wave_degree_verified": False,
            "exchange_fill_confirmed": False,
            "exchange_sl_confirmed": False, "reason": "NO_SETUP: нужна проверка завершённых баров",
            "trade_action": trade_action, "execution_allowed": False,
            "learning_need": "NONE", "learning_question": ""}


class ReadOnlyV6ResearchPipeline(unittest.TestCase):
    def test_all_four_sources_are_scoped_and_provenance_pinned(self):
        queries = {
            "TC1": "TC1 пятиволновая волна 3 дивергенция фрактал",
            "NTD": "NTD AC ускорение AO зелёные столбцы",
            "TC2": "TC2 третий мудрец фрактал зубы Alligator",
            "COURSE": "COURSE 5/34/5 momentum волны гистограмма",
        }
        books = {x["id"]: x for x in registry()["books"]}
        for edition, question in queries.items():
            with self.subTest(edition=edition):
                ret = review_chart(chart(), edition=edition, question=question)
                self.assertEqual(ret["edition"], edition)
                self.assertEqual(ret["state"], "SOURCE_ONLY_REVIEW_REQUIRED")
                self.assertTrue(ret["source_evidence"])
                for src in ret["source_evidence"]:
                    self.assertTrue(src["rule_id"].startswith(edition + "."))
                    self.assertEqual(src["source_pdf_sha256"], books[edition]["sha256"])
                    self.assertEqual(src["review_status"], "REVIEWED_PAGES")
                    self.assertFalse(ret["execution_allowed"])
                    self.assertFalse(ret["gpu_training_allowed"])
                    self.assertFalse(ret["main_merge_allowed"])

    def test_local_model_and_rag_use_no_hidden_gold(self):
        m = FakeResearchModel(proposal(rule_ids=["TC2.WM3"]))
        report = review_chart(chart(), edition="TC2",
            question="TC2 третий мудрец пятибарный фрактал пробой зубы Teeth", model=m)
        self.assertEqual(report["state"], "PROVISIONAL_BOOK_REASONING_NOT_TRADE")
        self.assertEqual(report["deterministic_audit"]["verdict"], "PROVISIONAL_REASONING_ONLY")
        self.assertEqual(len(m.calls), 1)
        passed_chart, kw = m.calls[0]
        self.assertEqual(passed_chart["symbol"], "BTCUSDT")
        self.assertEqual(kw["edition"], "TC2")
        self.assertTrue(kw["prior_lessons"])
        self.assertTrue(all(item["source"].startswith("TC2:RULE:")
                            for item in kw["prior_lessons"]))
        self.assertNotIn("expected", str(kw))
        self.assertNotIn("gold", str(kw))
        self.assertFalse(report["model_quality_gate_passed"])
        self.assertIsNone(report["trade_action"])

    def test_unretrieved_rule_citation_blocked_even_if_critic_accepts(self):
        # Pin the retrieved source independently of changing lexical ranks;
        # exercise the citation boundary, not a test-specific top-3 ranking.
        from unittest.mock import patch
        cards = {c["id"]: c for c in registry()["rule_cards"]}
        matched = cards["TC2.WM3"]
        source = {"rule_id": "TC2.WM3", "edition": "TC2",
                  "pdf_pages": matched["pdf_pages"],
                  "source_pdf_sha256": next(b["sha256"] for b in registry()["books"]
                                            if b["id"] == "TC2"),
                  "review_status": "REVIEWED_PAGES", "excerpt": matched["summary"]}
        m = FakeResearchModel(proposal(rule_ids=["TC2.CAMPAIGN"]))
        with patch("williams_ai.v6_research_pipeline._frozen_sources",
                   return_value=[source]):
            report = review_chart(chart(), edition="TC2",
                question="TC2 третий мудрец пятибарный фрактал пробой зубы Teeth", model=m)
        self.assertIn("MODEL_CITES_UNRETRIEVED_RULE", report["blockers"])
        self.assertEqual(report["state"], "BLOCKED_UNSUPPORTED_RESEARCH_CLAIM")
        self.assertIsNone(report["trade_action"])

    def test_unreviewed_correction_never_silently_promoted(self):
        m = FakeResearchModel(proposal(edition="NTD", rule_ids=["NTD.ALLIGATOR"]))
        report = review_chart(chart(), edition="NTD",
            question="NTD Alligator jaw lips teeth smoothed сдвиг губы челюсть", model=m)
        self.assertIn("SOURCE_CORRECTION_PENDING_SECOND_EXPERT_REVIEW", report["blockers"])
        self.assertFalse(report["gpu_training_allowed"])

    def test_model_trade_request_is_rejected_and_not_echoed(self):
        m = FakeResearchModel(proposal(rule_ids=["TC2.WM3"],
                                       trade_action={"OPEN_LONG": "BTCUSDT"}))
        report = review_chart(chart(), edition="TC2",
            question="TC2 третий мудрец пятибарный фрактал пробой зубы Teeth", model=m)
        self.assertIn("EXECUTION_REQUEST_FORBIDDEN", report["blockers"])
        self.assertIsNone(report["trade_action"])
        self.assertNotIn("OPEN_LONG", str(report))

    def test_cross_edition_model_source_is_blocked(self):
        m = FakeResearchModel(proposal(rule_ids=["NTD.AC"]))
        report = review_chart(chart(), edition="TC2",
            question="TC2 третий мудрец пятибарный фрактал пробой зубы Teeth", model=m)
        self.assertEqual(report["state"], "BLOCKED_UNSUPPORTED_RESEARCH_CLAIM")
        self.assertIn("CROSS_EDITION_RULE_MIXING", report["blockers"])

    def test_empty_source_has_no_model_call(self):
        m = FakeResearchModel(proposal())
        report = review_chart(chart(), edition="TC2",
            question="zzzxxyyssqq no known cards", model=m)
        self.assertEqual(report["state"], "ABSTAIN_NO_REVIEWED_SOURCE")
        self.assertEqual(m.calls, [])

    def test_future_candidate_rejected_before_source_or_model(self):
        bad = chart()
        bad["candidates"] = [{"type": "WM3_FRACTAL",
                              "confirmed_at": "2026-11-02T11:00:00Z"}]
        m = FakeResearchModel(proposal())
        with self.assertRaisesRegex(ValueError, "FUTURE_MARKET_EVIDENCE"):
            review_chart(bad, edition="TC2",
                question="TC2 третий мудрец пятибарный фрактал", model=m)
        self.assertEqual(m.calls, [])

    def test_future_high_timeframe_rejected(self):
        bad = chart()
        bad["higher_timeframe_context"]["H4"]["closed_through"] = "2026-11-02T12:00:00Z"
        with self.assertRaisesRegex(ValueError, "FUTURE_MARKET_EVIDENCE"):
            review_chart(bad, edition="TC2",
                question="TC2 третий мудрец пятибарный фрактал")

    def test_future_native_frame_rejected(self):
        bad = chart()
        bad["all_binance_timeframes_compact"] = {
            "H1": {"close": "2026-11-01T13:00:00Z"}}
        with self.assertRaisesRegex(ValueError, "FUTURE_MARKET_EVIDENCE"):
            review_chart(bad, edition="TC2",
                question="TC2 третий мудрец пятибарный фрактал")

    def test_gap_or_executable_chart_rejected(self):
        for field, value in (("gaps", 1), ("gaps", True),
                             ("execution_allowed", True)):
            with self.subTest(field=field, value=value):
                bad = chart()
                bad[field] = value
                with self.assertRaisesRegex(ValueError,
                                            "NONCAUSAL_OR_EXECUTABLE_CHART"):
                    review_chart(bad, edition="TC2",
                        question="TC2 третий мудрец пятибарный фрактал")

    def test_runtime_failure_never_masquerades_as_no_setup(self):
        m = FakeResearchModel(error=RuntimeError("model failed with sensitive data"))
        report = review_chart(chart(), edition="TC2",
            question="TC2 третий мудрец пятибарный фрактал пробой зубы Teeth", model=m)
        self.assertEqual(report["state"], "ABSTAIN_INVALID_MODEL_OR_GUARD")
        self.assertEqual(report["blockers"], ["MODEL_OR_GUARD_REJECTED"])
        self.assertNotIn("sensitive", str(report))
        self.assertFalse(report["execution_allowed"])

    def test_invalid_model_schema_fails_closed(self):
        m = FakeResearchModel({"schema": "bad"})
        report = review_chart(chart(), edition="TC2",
            question="TC2 третий мудрец пятибарный фрактал пробой зубы Teeth", model=m)
        self.assertEqual(report["state"], "ABSTAIN_INVALID_MODEL_OR_GUARD")

    def test_question_and_edition_validation(self):
        for edition, question in (("TC2+NTD", "TC2 неверный вопрос"),
                                  ("TC2", "x" * 1201),
                                  ("TC2", "small")):
            with self.subTest(edition=edition):
                with self.assertRaises(ValueError):
                    review_chart(chart(), edition=edition, question=question)


if __name__ == "__main__":
    unittest.main()
