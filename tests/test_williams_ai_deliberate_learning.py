"""Integration/adversarial tests of consultation, not a claim of AI skill."""
from copy import deepcopy
import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from williams_ai.algorithm_guard import audit_reasoning
from williams_ai.book_index import index_pages
from williams_ai.book_rules import registry
from williams_ai.causal_paper_lab import frozen_markets
from williams_ai.chart_reader import analyze_chart
from williams_ai.deliberate_learning import DeliberateBookModel
from williams_ai.grounded_inference import verified_private_book_pages
from williams_ai.reasoning_trial import evaluate_historical_thought
from williams_ai.learning_agenda import prior_reading_focus


def proposal(obs, *, setup="NO_SETUP", need="NONE"):
    return {"symbol":obs["symbol"], "as_of":obs["as_of"], "edition":"TC2",
        "setup":setup, "side":"LONG" if setup=="WM2" else "NONE",
        "entry_state":"CLOSED_CANDIDATE" if setup=="WM2" else "NO_SETUP",
        "exit_basis":"NONE", "rule_ids":["TC2.WM2"] if setup=="WM2" else [],
        "learning_need":need, "reason":"Research interpretation.",
        "execution_allowed":False, "trade_action":None}


class Student:
    def __init__(self, first=None, second=None, invented_quote=False):
        self.first=first or {"setup":"UNKNOWN", "need":"MANAGEMENT"}
        self.second=second or {}
        self.invented_quote=invented_quote
        self.calls=[]
        self.passages=[]

    def propose_chart_reasoning(self, obs, **kwargs):
        self.calls.append((deepcopy(obs),deepcopy(kwargs)))
        return proposal(obs, **(self.first if len(self.calls)==1 else self.second))

    def study_passage(self, text, **kwargs):
        self.passages.append((text,kwargs))
        return json.dumps({"summary":"Сопровождение стоп exit stop требует закрытых баров.",
            "claims":[{"statement":"Текст источника нужно проверить на рынке.",
                "support":"THIS SUPPORT DOES NOT EXIST IN THE SOURCE" if self.invented_quote else text[:60]}],
            "questions":["Какие факты подтвердят перенос стопа?"]})


class DeliberateLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.market=frozen_markets()["DOGEUSDT"]
        cls.rows=cls.market["bars"][:924]
        cls.obs=analyze_chart("DOGEUSDT","H1",cls.rows,as_of=cls.rows[-1]["close_time"],
            tick_size=cls.market["tick_size"])

    def setUp(self):
        self.temp=TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.books=self.root/"books.db"
        self.notes=self.root/"notes.db"
        ed=next(x for x in registry()["books"] if x["id"]=="TC2")
        pages=[""]*ed["pages"]
        # Artificial source fixtures, never real original-book validation.
        pages[130]="Защитный стоп проверяется по противоположному экстремуму закрытого бара. "*2
        pages[134]="Второй мудрец Super AO требует последовательности столбцов и ценового пробоя. "*2
        pages[160]="Сопровождение: стоп переносится за экстремумы завершённых баров; не расширять риск. "*2
        index_pages(self.books,"TC2",ed["sha256"],pages,declared_pages=ed["pages"])

    def wrap(self, student, model_id="gguf-student"):
        return DeliberateBookModel(student,self.books,self.notes,model_id=model_id)

    def test_rule_gap_reads_exact_pages_studies_and_reconsiders_same_cutoff(self):
        student=Student()
        model=self.wrap(student)
        result=model.propose_chart_reasoning(self.obs)
        self.assertEqual(result["setup"],"NO_SETUP")
        self.assertEqual(len(student.calls),2)
        self.assertEqual(student.calls[0][0],student.calls[1][0])
        self.assertEqual(student.calls[1][0]["as_of"],self.rows[-1]["close_time"])
        self.assertEqual(len(student.passages),2)
        refs=[x["source"] for x in student.calls[1][1]["primary_pages"]]
        self.assertEqual(set(refs),{"TC2: PDF 161","TC2: PDF 131"})
        self.assertEqual(student.calls[1][1]["study_lessons"],[])
        self.assertTrue(model.latest["recalled_notes"])
        self.assertTrue(model.latest["unverified_self_notes_quarantined"])
        self.assertEqual(model.latest["status"],"RECONSIDERED_NOT_INDEPENDENTLY_VALIDATED")
        self.assertFalse(model.latest["model_weights_updated"])
        self.assertFalse(model.latest["independent_review_completed"])
        with sqlite3.connect(self.notes) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM study_notes").fetchone()[0],2)
            event=json.loads(db.execute("SELECT diagnostics_json FROM decision_study_events").fetchone()[0])
            self.assertEqual(event["source_pages"],refs)
            self.assertNotIn("excerpt",json.dumps(event))

    def test_competent_wait_is_not_retried_to_manufacture_entry(self):
        student=Student(first={})
        student.first={}
        model=self.wrap(student)
        model.propose_chart_reasoning(self.obs)
        self.assertEqual(len(student.calls),1)
        self.assertEqual(student.passages,[])
        self.assertFalse(model.latest["reconsideration_performed"])

    def test_wave_gap_is_not_replaced_by_an_unrelated_angulation_rule(self):
        student=Student(first={"setup":"UNKNOWN","need":"WAVE"})
        model=self.wrap(student)
        result=model.propose_chart_reasoning(self.obs)
        self.assertEqual(result["setup"],"UNKNOWN")
        self.assertEqual(len(student.calls),1)
        self.assertEqual(student.passages,[])
        self.assertEqual(model.latest["source_rules"],[])
        self.assertEqual(model.latest["status"],"NO_REVIEWED_RULE_FOR_TOPIC_IN_SELECTED_EDITION")

    def test_critic_detects_unsupported_claim_even_without_declared_gap(self):
        student=Student(first={"setup":"WM2"})
        model=self.wrap(student)
        model.propose_chart_reasoning(self.obs)
        self.assertIn("WM2_SUPER_AO_NOT_OBSERVED",model.latest["first_failures"])
        self.assertIn("TC2.WM2",model.latest["source_rules"])
        self.assertTrue(all(x["source"]=="TC2: PDF 135"
            for x in student.calls[-1][1]["primary_pages"]))

    def test_named_rule_is_not_crowded_out_by_generic_entry_chapters(self):
        class Specific(Student):
            def propose_chart_reasoning(self,obs,**kwargs):
                result=super().propose_chart_reasoning(obs,**kwargs)
                if len(self.calls)==1:
                    result["rule_ids"]=["TC2.WM2"]
                return result
        student=Specific(first={"setup":"UNKNOWN","need":"ENTRY"})
        wrapper=self.wrap(student)
        wrapper.propose_chart_reasoning(self.obs)
        self.assertEqual(wrapper.latest["source_rules"][0],"TC2.WM2")
        self.assertIn("TC2: PDF 135",wrapper.latest["source_pages"])

    def test_precise_super_ao_question_finds_rule_without_model_knowing_its_identifier(self):
        class Question(Student):
            def propose_chart_reasoning(self,obs,**kwargs):
                result=super().propose_chart_reasoning(obs,**kwargs)
                if len(self.calls)==1:
                    result["learning_question"]="Super AO second wise man three histogram"
                return result
        student=Question(first={"setup":"UNKNOWN","need":"ENTRY"})
        wrapper=self.wrap(student)
        wrapper.propose_chart_reasoning(self.obs)
        self.assertEqual(wrapper.latest["source_rules"][0],"TC2.WM2")
        self.assertIn("TC2: PDF 135",wrapper.latest["source_pages"])

    def test_ntd_ac_error_reads_ac_pages_rather_than_ao_rules(self):
        ed=next(x for x in registry()["books"] if x["id"]=="NTD")
        pages=[""]*ed["pages"]
        pages[71]="AC acceleration: число зелёных столбцов зависит от знака AC, не AO. "*2
        index_pages(self.books,"NTD",ed["sha256"],pages,declared_pages=ed["pages"])
        class NtdStudent(Student):
            def propose_chart_reasoning(self,obs,**kwargs):
                self.calls.append((deepcopy(obs),deepcopy(kwargs)))
                first=len(self.calls)==1
                return {**proposal(obs),"edition":"NTD",
                    "setup":"AC" if first else "NO_SETUP",
                    "side":"LONG" if first else "NONE",
                    "rule_ids":["NTD.AC"] if first else []}
        obs=deepcopy(self.obs)
        obs["edition_evidence"]["NTD"]["signals"]=[]
        student=NtdStudent()
        wrapper=self.wrap(student)
        wrapper.propose_chart_reasoning(obs,edition="NTD")
        self.assertIn("NTD_SIGNAL_NOT_OBSERVED",wrapper.latest["first_failures"])
        self.assertEqual(wrapper.latest["source_rules"],["NTD.AC"])
        self.assertEqual(wrapper.latest["source_pages"],["NTD: PDF 72"])
        self.assertTrue(all(kwargs["edition"]=="NTD" for _,kwargs in student.passages))

    def test_two_bad_attempts_remain_failed_after_consultation(self):
        student=Student(first={"setup":"WM2"},second={"setup":"WM2"})
        model=self.wrap(student)
        result=model.propose_chart_reasoning(self.obs)
        self.assertEqual(len(student.calls),2)
        self.assertEqual(model.latest["status"],"UNSUPPORTED_AFTER_STUDY")
        self.assertIn("WM2_SUPER_AO_NOT_OBSERVED",audit_reasoning(self.obs,result)["hard_failures"])

    def test_fabricated_quote_never_enters_memory_but_source_can_still_be_read(self):
        student=Student(invented_quote=True)
        model=self.wrap(student)
        model.propose_chart_reasoning(self.obs)
        self.assertEqual(model.latest["failed_study_chunks"],2)
        self.assertEqual(model.latest["studied_chunks"],0)
        self.assertEqual(student.calls[1][1]["study_lessons"],[])
        self.assertTrue(student.calls[1][1]["primary_pages"])

    def test_restart_reuses_only_same_model_notes_without_claiming_new_training(self):
        self.wrap(Student()).propose_chart_reasoning(self.obs)
        resumed=Student()
        wrapper=self.wrap(resumed)
        wrapper.propose_chart_reasoning(self.obs)
        self.assertEqual(resumed.passages,[])
        self.assertEqual(wrapper.latest["studied_chunks"],0)
        self.assertTrue(wrapper.latest["recalled_notes"])
        other=Student()
        self.wrap(other,"gguf-other").propose_chart_reasoning(self.obs)
        self.assertEqual(len(other.passages),2)
        self.assertEqual(other.calls[-1][1]["study_lessons"],[])
        with sqlite3.connect(self.notes) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM study_notes WHERE model_id='gguf-other'").fetchone()[0],2)

    def test_later_chart_reads_original_pages_from_past_gap_before_first_answer(self):
        self.wrap(Student()).propose_chart_reasoning(self.obs)
        later=analyze_chart("DOGEUSDT","H1",self.market["bars"][:925],
            as_of=self.market["bars"][924]["close_time"],tick_size=self.market["tick_size"])
        student=Student();student.first={}
        wrapper=self.wrap(student)
        wrapper.propose_chart_reasoning(later)
        self.assertEqual(len(student.calls),1)
        pages=student.calls[0][1]["primary_pages"]
        self.assertEqual({x["source"] for x in pages},{"TC2: PDF 131","TC2: PDF 161"})
        self.assertEqual(wrapper.latest["prior_reading_requests"],1)
        self.assertEqual(student.passages,[])
        self.assertNotIn("SOURCE_DERIVED_UNVERIFIED",json.dumps(student.calls[0][1]))
        self.assertFalse(wrapper.latest["model_weights_updated"])

    def test_agenda_ignores_same_cutoff_future_other_model_and_changed_source_identity(self):
        self.wrap(Student()).propose_chart_reasoning(self.obs)
        def focus(model="gguf-student",as_of=None,edition="TC2"):
            return prior_reading_focus(self.notes,model_id=model,edition=edition,
                                      as_of=as_of or self.market["bars"][924]["close_time"])
        self.assertEqual(focus()["past_requests"],1)
        self.assertEqual(focus(as_of=self.obs["as_of"])["past_requests"],0)
        self.assertEqual(focus(as_of=self.market["bars"][920]["close_time"])["past_requests"],0)
        self.assertEqual(focus(model="gguf-other")["past_requests"],0)
        self.assertEqual(focus(edition="NTD")["past_requests"],0)
        with sqlite3.connect(self.notes) as db:
            raw=json.loads(db.execute("SELECT diagnostics_json FROM decision_study_events").fetchone()[0])
            raw["source_sha256"]="0"*64
            db.execute("UPDATE decision_study_events SET diagnostics_json=?",(json.dumps(raw),))
        self.assertEqual(focus()["past_requests"],0)

    def test_agenda_reloads_changed_ocr_instead_of_reusing_old_summary(self):
        self.wrap(Student()).propose_chart_reasoning(self.obs)
        fresh="CURRENT_OCR стоп сопровождение закрытые бары без прежнего пересказа. "*2
        with sqlite3.connect(self.books) as db:
            db.execute("UPDATE excerpts SET content=? WHERE book='TC2' AND pdf_page=161",(fresh,))
        later=deepcopy(self.obs);later["as_of"]=self.market["bars"][924]["close_time"]
        student=Student();student.first={}
        self.wrap(student).propose_chart_reasoning(later)
        self.assertTrue(any("CURRENT_OCR" in p["excerpt"] for p in student.calls[0][1]["primary_pages"]))
        self.assertEqual(student.passages,[])

    def test_unmapped_wave_question_reads_same_book_without_certifying_topic(self):
        wave="Elliott nested degree correction: волна степень вложенность коррекция. "*2
        with sqlite3.connect(self.books) as db:
            db.execute("INSERT INTO excerpts VALUES('TC2',175,?,'UNVERIFIED_EXTRACTION')",(wave,))
        class WaveStudent(Student):
            def propose_chart_reasoning(self,obs,**kwargs):
                answer=super().propose_chart_reasoning(obs,**kwargs)
                if len(self.calls)==1:
                    answer["learning_question"]="Elliott nested degree correction"
                return answer
        student=WaveStudent(first={"setup":"UNKNOWN","need":"WAVE"})
        wrapper=self.wrap(student)
        wrapper.propose_chart_reasoning(self.obs)
        self.assertEqual(len(student.calls),2)
        self.assertEqual(wrapper.latest["source_pages"],["TC2: PDF 175"])
        self.assertEqual(wrapper.latest["source_rules"],[])
        self.assertTrue(wrapper.latest["needs_independent_topic_review"])
        self.assertEqual(wrapper.latest["status"],"READ_UNREVIEWED_TOPIC_REQUIRES_INDEPENDENT_REVIEW")
        self.assertEqual(student.calls[1][1]["prior_lessons"],[])
        self.assertNotIn(wave,json.dumps(wrapper.latest))
        self.assertNotIn("Elliott nested degree correction",json.dumps(wrapper.latest))
        with sqlite3.connect(self.notes) as db:
            event=json.loads(db.execute("SELECT diagnostics_json FROM decision_study_events").fetchone()[0])
            self.assertEqual(event["private_learning_question"],"Elliott nested degree correction")
        # A later, generic wave chapter must not replace the precise topic
        # which this model actually requested before the current cutoff.
        with sqlite3.connect(self.books) as db:
            db.execute("INSERT INTO excerpts VALUES('TC2',176,?,'UNVERIFIED_EXTRACTION')",
                ("волна степень вложенность коррекция "*10,))
        later=deepcopy(self.obs);later["as_of"]=self.market["bars"][924]["close_time"]
        resumed=Student();resumed.first={}
        later_wrapper=self.wrap(resumed)
        later_wrapper.propose_chart_reasoning(later)
        self.assertEqual(len(resumed.calls),1)
        self.assertEqual(later_wrapper.latest["prior_reading_pages"],["TC2: PDF 175"])
        self.assertTrue(later_wrapper.latest["prior_question_reused"])
        self.assertTrue(later_wrapper.latest["needs_independent_topic_review"])
        self.assertEqual(later_wrapper.latest["status"],"PAST_UNREVIEWED_TOPIC_REQUIRES_INDEPENDENT_REVIEW")

    def test_unmapped_wave_reading_cannot_pass_historical_qualification_after_abstention(self):
        wave="Elliott волна степень вложенность коррекция. "*2
        with sqlite3.connect(self.books) as db:
            db.execute("INSERT INTO excerpts VALUES('TC2',175,?,'UNVERIFIED_EXTRACTION')",(wave,))
        result=evaluate_historical_thought("DOGEUSDT",self.market["bars"],
            tick_size=self.market["tick_size"],bar_index=923,
            model=Student(first={"setup":"UNKNOWN","need":"WAVE"}),
            include_topdown=False,book_index=self.books,study_memory=self.notes,
            model_id="gguf-student")
        self.assertEqual(result["audit"]["hard_failures"],[])
        self.assertFalse(result["research_reasoning_consistent"])
        self.assertTrue(result["deliberate_book_learning"]["needs_independent_topic_review"])

    def test_unanswered_book_question_and_unknown_setup_cannot_pass_consistency(self):
        for setup, need in (("NO_SETUP", "ENTRY"), ("UNKNOWN", "NONE")):
            student=Student(first={"setup":setup,"need":need},
                            second={"setup":setup,"need":need})
            result=evaluate_historical_thought("DOGEUSDT",self.market["bars"],
                tick_size=self.market["tick_size"],bar_index=923,model=student,
                include_topdown=False,book_index=self.books,study_memory=self.notes,
                model_id="gguf-student")
            self.assertEqual(result["audit"]["hard_failures"],[])
            self.assertFalse(result["book_question_resolved"])
            self.assertFalse(result["research_reasoning_consistent"])

    def test_dominant_entry_topic_does_not_erase_pending_independent_wave_review(self):
        self.wrap(Student(first={"setup":"UNKNOWN","need":"WAVE"})).propose_chart_reasoning(self.obs)
        for index in (924, 925):
            chart=deepcopy(self.obs);chart["as_of"]=self.market["bars"][index]["close_time"]
            self.wrap(Student(first={"setup":"UNKNOWN","need":"ENTRY"})).propose_chart_reasoning(chart)
        cutoff=self.market["bars"][926]["close_time"]
        agenda=prior_reading_focus(self.notes,model_id="gguf-student",edition="TC2",as_of=cutoff)
        self.assertEqual(agenda["topic"],"ENTRY")
        student=Student();student.first={}
        chart=deepcopy(self.obs);chart["as_of"]=cutoff
        wrapper=self.wrap(student);wrapper.propose_chart_reasoning(chart)
        self.assertTrue(wrapper.latest["needs_independent_topic_review"])
        self.assertEqual(wrapper.latest["status"],"PAST_UNREVIEWED_TOPIC_REQUIRES_INDEPENDENT_REVIEW")

    def test_numeric_stop_error_selects_campaign_source_in_next_fresh_model(self):
        from williams_ai.stop_review import audit_stop_review
        market=frozen_markets()["ETHUSDT"]
        opened=market["bars"][250]["close_time"]
        def journal(index):
            return {"symbol":"ETHUSDT","edition":"TC2",
                "as_of":market["bars"][index]["close_time"],"events":[
                    {"at":opened,"type":"OPEN_FILL","side":"LONG","price":1000,
                     "exchange_confirmed":True},
                    {"at":opened,"type":"STOP_ACK","stop_price":100,
                     "exchange_confirmed":True}]}
        class NumericStudent(Student):
            def propose_chart_reasoning(self,obs,**kwargs):
                self.calls.append((deepcopy(obs),deepcopy(kwargs)))
                return proposal(obs)
            def review_protective_stop(self,context):
                return {"level_relation":"LOWER","risk_direction":"WIDER",
                    "candidate_protects_side":True,"assessment":"KEEP_PROTECTION",
                    "reason":"A deliberately incorrect numerical interpretation."}
        first=evaluate_historical_thought("ETHUSDT",market["bars"],
            tick_size=market["tick_size"],bar_index=299,model=NumericStudent(),
            include_topdown=False,study_memory=self.notes,model_id="gguf-student",
            campaign_journal=journal(299),public_rule_learning=True)
        self.assertFalse(first["research_reasoning_consistent"])
        self.assertTrue(first["campaign_stop_review"]["next_reading_agenda_updated"])
        self.assertTrue(first["deliberate_book_learning"]["numeric_stop_review_learning_queued"])
        class Recovered(NumericStudent):
            def review_protective_stop(self,context):
                return {**audit_stop_review(context,None)["reference"],"reason":"Fixed test model."}
        student=Recovered()
        second=evaluate_historical_thought("ETHUSDT",market["bars"],
            tick_size=market["tick_size"],bar_index=300,model=student,
            include_topdown=False,study_memory=self.notes,model_id="gguf-student",
            campaign_journal=journal(300),public_rule_learning=True)
        self.assertEqual(second["deliberate_book_learning"]["prior_reading_rules"],["TC2.CAMPAIGN"])
        self.assertIn("TC2:RULE:TC2.CAMPAIGN",
            [x["source"] for x in student.calls[0][1]["prior_lessons"]])
        self.assertFalse(second["campaign_stop_review"]["next_reading_agenda_updated"])
        with sqlite3.connect(self.notes) as db:
            events=" ".join(x[0] for x in db.execute("SELECT diagnostics_json FROM decision_study_events"))
        self.assertNotIn("A deliberately incorrect numerical interpretation",events)
        self.assertNotIn("candidate_stop",events)
        self.assertNotIn('"reference"',events)

    def test_numeric_review_memory_rejects_another_chart_and_keeps_both_topics(self):
        from williams_ai.stop_review import diagnostic_cases, review_with_model
        context=diagnostic_cases()[0]["context"]
        chart=deepcopy(self.obs)
        chart["symbol"]=context["symbol"];chart["as_of"]=context["as_of"]
        wrapper=self.wrap(Student(first={"setup":"UNKNOWN","need":"ENTRY"}))
        wrapper.propose_chart_reasoning(chart)
        class Wrong:
            def review_protective_stop(self,context):
                return {"level_relation":"HIGHER","risk_direction":"TIGHTER",
                    "candidate_protects_side":True,"assessment":"REVIEW_TIGHTENING",
                    "reason":"Incorrect model response."}
        review=review_with_model(Wrong(),context)
        # A cached or edited success flag cannot override numerical facts.
        review["audit"]={"hard_failures":[],"verdict":"NUMERIC_RESEARCH_CONSISTENT"}
        wrong=deepcopy(chart);wrong["symbol"]="ETHUSDT"
        with self.assertRaisesRegex(ValueError,"CONTEXT_MISMATCH"):
            wrapper.remember_stop_review(wrong,context,review)
        self.assertTrue(wrapper.remember_stop_review(chart,context,review))
        later="2026-10-21T00:59:59.999000+00:00"
        agenda=prior_reading_focus(self.notes,model_id="gguf-student",edition="TC2",as_of=later)
        self.assertEqual(agenda["past_requests"],2)
        self.assertEqual(agenda["rule_ids"][0],"TC2.CAMPAIGN")
        self.assertTrue(agenda["numeric_stop_review_pending"])
        at_same=prior_reading_focus(self.notes,model_id="gguf-student",edition="TC2",as_of=chart["as_of"])
        self.assertEqual(at_same["past_requests"],0)
        other=prior_reading_focus(self.notes,model_id="gguf-other",edition="TC2",as_of=later)
        self.assertEqual(other["past_requests"],0)
        self.assertFalse(wrapper.remember_stop_review(chart,context,review_with_model(object(),context)))
        still_pending=prior_reading_focus(self.notes,model_id="gguf-student",edition="TC2",as_of=later)
        self.assertTrue(still_pending["numeric_stop_review_pending"])
        from williams_ai.stop_review import audit_stop_review
        class Correct:
            def review_protective_stop(self,context):
                return {**audit_stop_review(context,None)["reference"],"reason":"Verified fixture."}
        next_context={**context,"as_of":later}
        next_chart={**chart,"as_of":later}
        self.assertFalse(wrapper.remember_stop_review(next_chart,next_context,
            review_with_model(Correct(),next_context)))
        after=prior_reading_focus(self.notes,model_id="gguf-student",edition="TC2",
            as_of="2026-10-22T00:59:59.999000+00:00")
        self.assertFalse(after["numeric_stop_review_pending"])
        self.assertFalse(after["mastery_certified"])

    def test_non_object_model_answer_uses_typed_error_instead_of_attribute_crash(self):
        class Broken(Student):
            def propose_chart_reasoning(self,obs,**kwargs):
                if not self.calls:
                    self.calls.append((obs,kwargs))
                    return []
                return proposal(obs)
        wrapper=self.wrap(Broken())
        try:
            wrapper.propose_chart_reasoning(self.obs)
        except ValueError:
            self.assertTrue(wrapper.latest["first_failures"])

    def test_focused_question_is_bounded_and_cannot_hide_a_book_gap(self):
        bad={**proposal(self.obs),"learning_question":"Какой стоп нужен?"}
        self.assertIn("BOOK_QUESTION_WITHOUT_DECLARED_TOPIC",audit_reasoning(self.obs,bad)["hard_failures"])
        for value in ([],"x"*401):
            bad["learning_question"]=value
            self.assertIn("INVALID_MODEL_LEARNING_QUESTION",audit_reasoning(self.obs,bad)["hard_failures"])

    def test_public_rule_mode_uses_same_learning_loop_without_faking_original_pages(self):
        student=Student()
        wrapper=DeliberateBookModel(student,None,self.notes,model_id="gguf-student",public_rules_only=True)
        wrapper.propose_chart_reasoning(self.obs)
        self.assertEqual(student.passages,[])
        self.assertEqual(len(student.calls),2)
        self.assertEqual(student.calls[1][1]["primary_pages"],[])
        self.assertTrue(student.calls[1][1]["prior_lessons"])
        self.assertEqual(wrapper.latest["source_kind"],"PUBLIC_REVIEWED_RULE_PARAPHRASE")
        self.assertEqual(wrapper.latest["source_pages"],[])
        self.assertFalse(wrapper.latest["private_original_pages_read"])
        self.assertEqual(wrapper.latest["studied_chunks"],0)
        later=deepcopy(self.obs);later["as_of"]=self.market["bars"][924]["close_time"]
        resumed=Student();resumed.first={}
        later_wrapper=DeliberateBookModel(resumed,None,self.notes,model_id="gguf-student",public_rules_only=True)
        later_wrapper.propose_chart_reasoning(later)
        self.assertEqual(len(resumed.calls),1)
        self.assertTrue(resumed.calls[0][1]["prior_lessons"])
        self.assertTrue(later_wrapper.latest["prior_reading_public_references"])
        private_focus=prior_reading_focus(self.notes,model_id="gguf-student",edition="TC2",as_of=later["as_of"])
        self.assertEqual(private_focus["past_requests"],0)

    def test_public_learning_historical_entry_point_and_source_modes_are_explicit(self):
        student=Student()
        result=evaluate_historical_thought("DOGEUSDT",self.market["bars"],
            tick_size=self.market["tick_size"],bar_index=923,model=student,
            include_topdown=False,study_memory=self.notes,model_id="gguf-student",
            public_rule_learning=True)
        self.assertTrue(result["public_rule_learning_only"])
        self.assertEqual(result["private_book_pages_used"],[])
        self.assertTrue(result["deliberate_book_learning"]["public_rule_references"])
        self.assertFalse(result["execution_allowed"])
        for kwargs in ({},{"public_rules_only":True}):
            with self.assertRaises(ValueError):
                DeliberateBookModel(Student(),self.books if kwargs else None,self.notes,
                    model_id="gguf-student",**kwargs)

    def test_all_four_editions_can_request_their_own_public_book_rules(self):
        from williams_ai.deliberate_learning import _GAPS
        for edition in ("TC1","NTD","TC2","COURSE"):
            class EditionStudent(Student):
                def propose_chart_reasoning(self,obs,**kwargs):
                    answer=super().propose_chart_reasoning(obs,**kwargs)
                    answer["edition"]=kwargs["edition"]
                    return answer
            student=EditionStudent(first={"setup":"UNKNOWN","need":"ENTRY"})
            wrapper=DeliberateBookModel(student,None,self.notes,
                model_id="gguf-student",public_rules_only=True)
            wrapper.propose_chart_reasoning(self.obs,edition=edition)
            sources=student.calls[1][1]["prior_lessons"]
            self.assertEqual({x["source"] for x in sources},
                {edition+":RULE:"+rule for rule in _GAPS[edition]["ENTRY"]})
            self.assertFalse(wrapper.latest["private_original_pages_read"])

    def test_missing_source_does_not_invent_a_repaired_answer(self):
        with sqlite3.connect(self.books) as db:
            db.execute("DELETE FROM excerpts")
        student=Student()
        wrapper=self.wrap(student)
        result=wrapper.propose_chart_reasoning(self.obs)
        self.assertEqual(result["setup"],"UNKNOWN")
        self.assertEqual(len(student.calls),1)
        self.assertEqual(wrapper.latest["status"],"SOURCE_TEXT_MISSING_VISUAL_REVIEW_REQUIRED")

    def test_declared_gap_cannot_also_authorize_an_entry(self):
        result=audit_reasoning(self.obs,proposal(self.obs,setup="WM2",need="MANAGEMENT"))
        self.assertIn("ENTRY_CLAIM_WHILE_BOOK_RULE_NOT_UNDERSTOOD",result["hard_failures"])
        self.assertIn("INVALID_MODEL_LEARNING_NEED",audit_reasoning(
            self.obs,{**proposal(self.obs),"learning_need":"execute Binance now"})["hard_failures"])

    def test_focus_rejects_cross_edition_rules_and_false_book_identity(self):
        with self.assertRaisesRegex(ValueError,"same-edition"):
            verified_private_book_pages(self.books,"TC2",rule_ids=("NTD.EXITS",))
        with sqlite3.connect(self.books) as db:
            db.execute("UPDATE volumes SET sha256=?",("0"*64,))
        student=Student()
        wrapper=self.wrap(student)
        wrapper.propose_chart_reasoning(self.obs)
        self.assertEqual(wrapper.latest["status"],"SOURCE_UNAVAILABLE_OR_IDENTITY_REJECTED")
        self.assertEqual(student.passages,[])

    def test_exact_matching_chunk_not_unrelated_first_fragment_is_studied(self):
        from williams_ai.study import fingerprint
        irrelevant="This first fragment contains general introductory material only."
        relevant="Сопровождение стоп перенос: stop trailing pyramid. Проверить завершённые бары. "*2
        with sqlite3.connect(self.books) as db:
            db.execute("DELETE FROM excerpts WHERE pdf_page=161")
            db.execute("INSERT INTO excerpts VALUES('TC2',161,?,'UNVERIFIED_EXTRACTION')",(irrelevant,))
            db.execute("INSERT INTO excerpts VALUES('TC2',161,?,'UNVERIFIED_EXTRACTION')",(relevant,))
            db.execute("INSERT INTO excerpts VALUES('TC2',162,?,'UNVERIFIED_EXTRACTION')",(relevant+" Extra.",))
        student=Student()
        model=self.wrap(student)
        model.propose_chart_reasoning(self.obs)
        self.assertEqual(set(model.latest["source_pages"]),{"TC2: PDF 161","TC2: PDF 131"})
        self.assertNotIn(irrelevant,[text for text,_ in student.passages])
        with sqlite3.connect(self.notes) as db:
            rows=db.execute("SELECT page,chunk_sha FROM study_notes").fetchall()
            self.assertIn((161,fingerprint(relevant)),rows)
            self.assertTrue(any(page==131 for page,_ in rows))

    def test_historical_entry_point_invokes_learning_before_final_audit(self):
        student=Student()
        result=evaluate_historical_thought("DOGEUSDT",self.market["bars"],
            tick_size=self.market["tick_size"],bar_index=923,model=student,
            include_topdown=False,book_index=self.books,study_memory=self.notes,
            model_id="gguf-student")
        self.assertTrue(result["deliberate_book_learning"]["reconsideration_performed"])
        self.assertFalse(result["execution_allowed"])
        self.assertEqual(student.calls[-1][0]["as_of"],self.rows[-1]["close_time"])
        self.assertEqual(result["audit"]["hard_failures"],[])

    def test_cli_automatically_creates_model_scoped_memory_and_runs_consultation(self):
        from williams_ai.reasoning_trial import main
        student=Student()
        weights=self.root/"fixture.gguf"
        weights.write_bytes(b"TEST_RUNTIME_INJECTED_NOT_A_REAL_MODEL")
        argv=["reasoning_trial","--symbol","DOGEUSDT","--bar-index","923",
            "--local-model",str(weights),"--book-index",str(self.books),"--no-topdown"]
        output=io.StringIO()
        with patch("sys.argv",argv),patch("williams_ai.local_model.LocalWilliamsModel",
                return_value=student) as constructor,redirect_stdout(output):
            main()
        result=json.loads(output.getvalue())
        self.assertTrue(result["deliberate_book_learning"]["reconsideration_performed"])
        self.assertTrue((self.root/"Williams_AI_Decision_Study.sqlite3").is_file())
        self.assertEqual(constructor.call_args.kwargs["learning_mode"],"auto")
        self.assertTrue(constructor.call_args.kwargs["use_decision_schema"])


if __name__=="__main__":
    unittest.main()
