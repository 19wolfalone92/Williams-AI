import unittest
from datetime import datetime,timezone,timedelta
from williams_ai.wave_evidence import corrective_alternatives


def p(kind, value, i):
    return {"type":kind,"price":value,"index":i,"confirmed_at_index":i+2,
            "time":(datetime(2026,1,1,tzinfo=timezone.utc)+timedelta(hours=i)).isoformat()}


class CorrectionShapeTests(unittest.TestCase):
    def test_zigzag_and_flat_are_only_causal_hypotheses(self):
        zig=[p("L",100,1),p("H",120,25),p("L",109,48),p("H",130,78)]
        flat=[p("L",100,1),p("H",120,25),p("L",101,48),p("H",120,78)]
        a=corrective_alternatives(zig)
        b=corrective_alternatives(flat)
        self.assertEqual(a[0]["type"],"ZIGZAG_SHAPE_CANDIDATE")
        self.assertEqual(b[0]["type"],"FLAT_SHAPE_CANDIDATE")
        self.assertEqual(a[0]["confirmation_index"],80)
        self.assertTrue(all(x["status"].startswith("SHAPE_ONLY") for x in a+b))
        self.assertNotIn("execution_allowed",a[0])

    def test_contraction_is_not_claimed_as_complete_elliott_triangle(self):
        series=[p(k,v,i) for k,v,i in [
            ("H",120,1),("L",100,20),("H",117,40),
            ("L",103,60),("H",114,80),("L",106,100)]]
        options=corrective_alternatives(series)
        self.assertTrue(any(x["type"]=="CONTRACTING_TRIANGLE_SHAPE_CANDIDATE" for x in options))
        self.assertTrue(all(x["status"].startswith("SHAPE_ONLY") for x in options))

    def test_reject_non_alternating_and_incomplete(self):
        self.assertEqual(corrective_alternatives([p("H",100,1),p("H",110,3),p("L",108,5),p("H",114,8)]),[])
        self.assertEqual(corrective_alternatives([p("L",100,1),p("H",120,2),p("L",95,3)]),[])

    def test_failed_zigzag_retracement_not_upgraded(self):
        invalid=[p("L",100,1),p("H",120,20),p("L",80,50),p("H",130,80)]
        self.assertFalse(any(x["type"]=="ZIGZAG_SHAPE_CANDIDATE" for x in corrective_alternatives(invalid)))

if __name__=="__main__": unittest.main()
