import json
import unittest
from williams_ai.v6_objective_practice import audit_v5, curriculum, case, FAMILIES


class ObjectivePracticeTests(unittest.TestCase):
    def test_frozen_v5_shortcuts_confirmed_not_changed(self):
        x=audit_v5()
        self.assertEqual(x["splits"]["train"]["rows"],300)
        self.assertEqual(x["splits"]["train"]["unique_rules"],25)
        for s,n in (("train",100),("validation",16),("test",28)):
            p=x["splits"][s]
            self.assertEqual(p["mcq_rows"],n)
            self.assertEqual(p["longest_choice_correct"],n)
            self.assertEqual(p["generic_distractor_shortcut_correct"],n)
            self.assertEqual(p["boundary_answer_evidence_in_prompt"],n)
        print("V5_SHORTCUT_AUDIT",json.dumps(x,sort_keys=True))

    def test_objective_only_distinct_cases_and_no_false_final_claim(self):
        rows,m=curriculum()
        self.assertEqual(m["split_rows"],{"train":64,"validation":32,"development":32})
        self.assertEqual(len({r["case_id"] for r in rows}),128)
        self.assertEqual(len({r["question"] for r in rows}),128)
        self.assertTrue(all(not r["market_expert_gold"] and not r["execution_allowed"] for r in rows))
        self.assertFalse(m["independent_final_exam"])
        self.assertFalse(m["independent_expert_review"])
        self.assertEqual(m["candidate_labels_used"],0)
        for i,s in enumerate(("train","validation","development")):
            self.assertTrue(all(1000+1000*i<=r["numeric_anchor"]<2000+1000*i
                                for r in rows if r["split"]==s))
        self.assertEqual(curriculum(),(rows,m))
        print("V6_OBJECTIVE_PILOT_MANIFEST",json.dumps(m,sort_keys=True))

    def test_positive_negative_and_mirrored_conditions(self):
        def answer(family,i):
            return json.loads(case(family,"train",i)["answer"])["answer"]
        for family in ("wm1_shape","wm2_direction","wm3_activation"):
            self.assertEqual([answer(family,i) for i in range(4)],["YES","YES","NO","NO"])
        self.assertEqual([answer("fractal",i) for i in range(4)],["UP","DOWN","NONE","NONE"])
        self.assertEqual([answer("zone",i) for i in range(4)],["GREEN","RED","GRAY","GRAY"])
        self.assertEqual([answer("course_momentum",i) for i in range(2)],["UP","DOWN"])
        self.assertGreater(answer("ao_difference",0),0)
        self.assertLess(answer("ao_difference",1),0)
