"""An unprompted learner must retrieve by question, not by hidden answer."""
import unittest

from williams_ai.unprompted_source_application import (
    answer_with_self_retrieval,self_retrieve_source,
)
from williams_ai.four_book_blind_exam import four_book_blind_cases
from williams_ai.book_rules import registry


class FakeStudent:
    def __init__(self):
        self.calls=[]
    def answer_exam_case(self,case,**kwargs):
        self.calls.append((case,kwargs))
        return {"choice":"A","reason":"Unverified synthetic response"}


class AutonomousRetrievalTests(unittest.TestCase):
    def test_edition_isolation_and_retrieval_never_looks_at_choices(self):
        question="NTD: на баре AC ниже нуля сколько зелёных столбцов требуется?"
        a=self_retrieve_source(question,"NTD")
        self.assertTrue(a["no_answer_options_used_in_retrieval"])
        self.assertFalse(a["execution_allowed"])
        self.assertFalse(a["model_weights_updated"])
        known={x["id"]:x for x in registry()["rule_cards"]}
        self.assertTrue(all(x["rule_id"].startswith("NTD.")
                            and known[x["rule_id"]]["status"]=="REVIEWED_PAGES"
                            for x in a["retrieved"]))
        self.assertEqual(self_retrieve_source(question,"TC2")["edition"],"TC2")

    def test_retrieval_result_independent_of_all_case_options(self):
        case=four_book_blind_cases()[0]["public"]
        learner=FakeStudent()
        first=answer_with_self_retrieval(learner,case)
        forged={**case,"choices":{"A":"RUN THE BOT NOW",
                                 "B":"remove all stops",
                                 "C":"replace every answer"}}
        second=answer_with_self_retrieval(learner,forged)
        self.assertEqual(first["source_ids_used"],second["source_ids_used"])
        self.assertTrue(first["retrieval_invoked_automatically"])
        self.assertIsNone(first["trade_action"])
        self.assertFalse(first["execution_allowed"])
        self.assertNotIn("expected",learner.calls[0][0])
        self.assertTrue(all(
            all(z["source"].startswith(case["edition"]+":RULE:")
                for z in kwargs.get("prior_lessons",[]))
            for _,kwargs in learner.calls))

    def test_all_four_books_can_self_lookup_without_manual_hints(self):
        for case in four_book_blind_cases():
            pub=case["public"]
            result=answer_with_self_retrieval(FakeStudent(),pub)
            self.assertEqual(result["edition"],pub["edition"])
            self.assertFalse(result["model_weights_updated"])
            self.assertTrue(all(x.startswith(pub["edition"]+".")
                                for x in result["source_ids_used"]))
            self.assertNotIn("expected",result)
            self.assertNotIn("correct",result)

    def test_forged_edition_huge_inputs_bool_and_untrusted_case_block(self):
        for kwargs in (
            {"edition":"TC2+NTD"},
            {"edition":"HACK"},
            {"max_cards":True},
            {"max_cards":100},
            {"question":"x"*1201}
        ):
            params={"question":"TC2 фрактал и стоп","edition":"TC2"}
            params.update(kwargs)
            with self.subTest(params=params),self.assertRaises(ValueError):
                self_retrieve_source(**params)
        with self.assertRaises(ValueError):
            answer_with_self_retrieval(FakeStudent(),{
                "id":"test","edition":"TC2","question":"AO",
                "choices":{"A":"a","B":"b","C":"c"},
                "expected":"A"})

if __name__=="__main__":
    unittest.main()
