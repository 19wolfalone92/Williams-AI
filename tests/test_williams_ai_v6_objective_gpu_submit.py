import unittest
from types import SimpleNamespace
from scripts.ai_submit_v6_objective_smoke import worker_source,status_absent,SLUG


class SubmitContracts(unittest.TestCase):
    def test_worker_is_pinned_private_and_stoppable(self):
        code=worker_source("a"*40)
        compile(code,"worker.py","exec")
        self.assertIn("a"*40,code)
        self.assertIn(SLUG,code)
        self.assertIn("os.killpg",code)
        self.assertNotIn("KAGGLE_API_TOKEN",code)
        with self.assertRaises(ValueError):worker_source("main")

    def test_absence_fail_closed(self):
        def fail(code):
            def f(_):
                e=RuntimeError("test")
                e.status=code
                raise e
            return SimpleNamespace(kernels_status=f)
        self.assertTrue(status_absent(fail(404),"owner/slug"))
        for code in (401,403,500,None):
            with self.assertRaisesRegex(RuntimeError,"ABSENCE_NOT_VERIFIED"):
                status_absent(fail(code),"owner/slug")
        self.assertFalse(status_absent(SimpleNamespace(kernels_status=lambda _:object()),"owner/slug"))
