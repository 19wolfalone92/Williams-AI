import unittest
from collections import Counter
from williams_ai.v5_objective_diagnostic import ITEMS,build,grade_choice


class ObjectiveDiagnosticTests(unittest.TestCase):
    def test_balanced_fifteen_competencies_and_hidden_keys(self):
        rows=build(42)
        self.assertEqual(len(rows),30)
        self.assertEqual(Counter(r["competency"] for r in rows),{i[0]:2 for i in ITEMS})
        self.assertEqual(Counter(r["expected"] for r in rows),dict.fromkeys("ABC",10))
        self.assertEqual({r["public"]["edition"] for r in rows},{"TC1","NTD","TC2","COURSE"})
        for r in rows:
            p=r["public"]
            self.assertEqual(set(p),{"id","edition","question","choices"})
            i=int(p["id"].split("-")[1])
            self.assertEqual(p["choices"][r["expected"]],ITEMS[i][3])
            self.assertEqual(len(set(p["choices"].values())),3)

    def test_seed_reproducible_not_fixed_answer_position(self):
        self.assertEqual(build(42),build(42))
        self.assertNotEqual(build(42),build(43))

    def test_technical_failures_are_not_analytical_abstention(self):
        self.assertEqual(grade_choice("A","A","length"),"TECHNICAL_TRUNCATION")
        self.assertEqual(grade_choice("","A","stop"),"TECHNICAL_OR_UNPARSED")
        self.assertEqual(grade_choice("B","A","stop"),"INCORRECT")
        self.assertEqual(grade_choice("A","A","stop"),"CORRECT")


if __name__=="__main__":
    unittest.main()
