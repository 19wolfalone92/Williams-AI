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

    def test_denied_status_requires_complete_own_inventory(self):
        def denied(kernel):
            raise ValueError(f"Cannot access kernel '{kernel}' (Permission 'kernels.get' was denied).")
        witness=SimpleNamespace(ref="owner/williams-ai-v5-book-sft-20261111")
        target=SimpleNamespace(ref="owner/slug")
        def inventory(items,token=None):
            return SimpleNamespace(kernels=items,next_page_token=token)
        def api(pages):
            it=iter(pages)
            return SimpleNamespace(kernels_status=denied,kernels_list_with_response=lambda **kw:next(it))
        self.assertTrue(status_absent(api([inventory([witness],"next"),inventory([])]),"owner/slug"))
        self.assertFalse(status_absent(api([inventory([witness],"next"),inventory([target])]),"owner/slug"))
        for pages in ([inventory([])],[inventory([witness],"again"),inventory([],"again")],
                      [inventory([witness],"next")],[inventory([SimpleNamespace(ref="other/slug")])]):
            with self.assertRaisesRegex(RuntimeError,"ABSENCE_NOT_VERIFIED"):
                status_absent(api(pages),"owner/slug")

    def test_owner_scoped_metadata_slug_resolves_missing_ref(self):
        def denied(kernel):
            raise ValueError(f"Cannot access kernel '{kernel}' (Permission 'kernels.get' was denied).")
        witness=SimpleNamespace(ref="OWNER/williams-ai-v5-book-sft-20261111")
        def api(item):
            def inventory(**kw):
                self.assertEqual(kw["user"],"owner")
                return SimpleNamespace(kernels=[witness,item],next_page_token=None)
            return SimpleNamespace(kernels_status=denied,kernels_list_with_response=inventory)
        self.assertTrue(status_absent(api(SimpleNamespace(ref="",slug="other-notebook",id=123)),"owner/slug"))
        self.assertFalse(status_absent(api(SimpleNamespace(ref="",slug="slug",id=123)),"owner/slug"))
        with self.assertRaisesRegex(RuntimeError,"ABSENCE_NOT_VERIFIED"):
            status_absent(api(SimpleNamespace(ref="",slug="",id=123)),"owner/slug")
