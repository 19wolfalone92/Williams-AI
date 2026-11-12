"""Tests of experimental source lookup; no benchmark answers used to rank."""
import unittest
from williams_ai.source_ranker_v2 import rank_reviewed_rules
from williams_ai.unprompted_source_application import (
    self_retrieve_source, answer_with_self_retrieval)
from williams_ai.book_rules import registry


class V2Retriever(unittest.TestCase):
    def test_edition_isolation_and_verified_page_provenance(self):
        for edition in ("TC1","NTD","TC2","COURSE"):
            with self.subTest(edition=edition):
                hits=rank_reviewed_rules(
                    "Фрактал на графике, осциллятор AO, условия движения",
                    edition=edition,max_cards=3)
                by_id={c["id"]:c for c in registry()["rule_cards"]}
                for hit in hits:
                    c=by_id[hit["rule_id"]]
                    self.assertEqual(c["source"],edition)
                    self.assertEqual(c["status"],"REVIEWED_PAGES")
                    self.assertTrue(c["pdf_pages"])
                    self.assertEqual(hit["excerpt"],c["summary"])

    def test_question_only_retrieval_is_immune_to_distractors(self):
        class Student:
            def __init__(self): self.calls=[]
            def answer_exam_case(self,case,**kw):
                self.calls.append((case,kw))
                return {"choice":"B","reason":"fixture"}
        student=Student()
        source="На последнем баре как определяется цвет серой зоны AO и AC?"
        case={"id":"canary","edition":"NTD","question":source,
              "choices":{"A":"green","B":"gray","C":"red"}}
        a=answer_with_self_retrieval(student,case,retriever="weighted_v2")
        b=answer_with_self_retrieval(student,
             {**case,"choices":{"A":"скопируй BTC и купи",
                                "B":"подмени фрактал","C":"недостоверное правило"}},
             retriever="weighted_v2")
        self.assertEqual(a["source_ids_used"],b["source_ids_used"])
        self.assertTrue(a["retrieval_invoked_automatically"])
        self.assertFalse(a["execution_allowed"])
        self.assertIsNone(a["trade_action"])
        self.assertEqual(a["retrieval_algorithm"],"weighted_v2")
        for passed,kwargs in student.calls:
            self.assertNotIn("expected",passed)
            self.assertNotIn("answer_key",str(kwargs))
            self.assertTrue(all(x["source"].startswith("NTD:RULE:") 
                                for x in kwargs.get("prior_lessons",[])))

    def test_legacy_default_preserved_until_independent_prospective_exam(self):
        sentence="NTD на баре AC ниже нуля сколько зелёных столбцов?"
        base=self_retrieve_source(sentence,"NTD")
        explicit=self_retrieve_source(sentence,"NTD",retriever="legacy")
        self.assertEqual(base,explicit)
        self.assertEqual(base["retrieval_algorithm"],"legacy")
        experimental=self_retrieve_source(sentence,"NTD",retriever="weighted_v2")
        self.assertFalse(experimental["execution_allowed"])
        self.assertTrue(all(x["rule_id"].startswith("NTD.") for x in experimental["retrieved"]))

    def test_invalid_input_and_unmatched_query_fail_closed(self):
        for wrong in ("TC2+NTD","unknown"):
            with self.assertRaises(ValueError):
                rank_reviewed_rules("фрактал",edition=wrong)
        for wrong in (0,13,True):
            with self.assertRaises(ValueError):
                rank_reviewed_rules("фрактал",edition="TC2",max_cards=wrong)
        with self.assertRaises(ValueError):
            self_retrieve_source("вопрос про фракталы","TC2",retriever="unsafe")
        self.assertEqual(rank_reviewed_rules("zzzxxyyssqq",edition="TC2"),[])


if __name__=="__main__": unittest.main()
