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
        witness=SimpleNamespace(ref="owner/williams-ai-v5-book-sft-20261111",id=1)
        target=SimpleNamespace(ref="owner/valid-slug",id=2)
        def inventory(items,token=None):
            return SimpleNamespace(kernels=items,next_page_token=token)
        def api(pages):
            it=iter(pages)
            return SimpleNamespace(kernels_status=denied,kernels_list_with_response=lambda **kw:next(it))
        self.assertTrue(status_absent(api([inventory([witness],"next"),inventory([])]),"owner/valid-slug"))
        self.assertFalse(status_absent(api([inventory([witness],"next"),inventory([target])]),"owner/valid-slug"))
        for pages in ([inventory([])],[inventory([witness],"again"),inventory([],"again")],
                      [inventory([witness],"next")],[inventory([SimpleNamespace(ref="other/slug")])]):
            with self.assertRaisesRegex(RuntimeError,"ABSENCE_NOT_VERIFIED"):
                status_absent(api(pages),"owner/slug")

    def test_owner_scoped_metadata_slug_resolves_missing_ref(self):
        def denied(kernel):
            raise ValueError(f"Cannot access kernel '{kernel}' (Permission 'kernels.get' was denied).")
        witness=SimpleNamespace(ref="OWNER/williams-ai-v5-book-sft-20261111",id=1)
        def api(item):
            def inventory(**kw):
                self.assertEqual(kw["user"],"owner")
                return SimpleNamespace(kernels=[witness,item],next_page_token=None)
            return SimpleNamespace(kernels_status=denied,kernels_list_with_response=inventory)
        self.assertTrue(status_absent(api(SimpleNamespace(ref="",slug="other-notebook",id=123,title="Other Notebook")),"owner/valid-slug"))
        self.assertFalse(status_absent(api(SimpleNamespace(ref="",slug="valid-slug",id=123,title="Valid Slug")),"owner/valid-slug"))
        with self.assertRaisesRegex(RuntimeError,"ABSENCE_NOT_VERIFIED"):
            status_absent(api(SimpleNamespace(ref="",slug="",id=123,title="")),"owner/valid-slug")


    def test_title_only_inventory_rows_use_kaggle_slug_normalization(self):
        def denied(kernel):
            raise ValueError(f"Cannot access kernel '{kernel}' (Permission 'kernels.get' was denied).")
        witness=SimpleNamespace(ref="owner/williams-ai-v5-book-sft-20261111",id=1)
        item=SimpleNamespace(ref="",slug="",title="Legacy Notebook Title",id=2)
        def api(items):
            def inventory(**kw):
                self.assertEqual(kw["mine"],True)
                self.assertEqual(kw["user"],"owner")
                self.assertEqual(kw["page_size"],100)
                return SimpleNamespace(kernels=[witness,*items],next_page_token=None)
            return SimpleNamespace(kernels_status=denied,kernels_list_with_response=inventory)
        self.assertTrue(status_absent(api([item]),"owner/new-kernel-target"))
        self.assertFalse(status_absent(api([item]),"owner/legacy-notebook-title"))

    def test_title_only_inventory_rows_can_resolve_when_numeric_id_is_omitted(self):
        def denied(kernel):
            raise ValueError(f"Cannot access kernel '{kernel}' (Permission 'kernels.get' was denied).")
        witness=SimpleNamespace(ref="owner/williams-ai-v5-book-sft-20261111",id=1)
        item=SimpleNamespace(ref="",slug="",title="Legacy Notebook Title",id=0)
        def inventory(**kw):
            return SimpleNamespace(kernels=[witness,item],next_page_token=None)
        api=SimpleNamespace(kernels_status=denied,kernels_list_with_response=inventory)
        self.assertFalse(status_absent(api,"owner/legacy-notebook-title"))

    def test_duplicate_title_slug_keeps_absence_blocked(self):
        def denied(kernel):
            raise ValueError(f"Cannot access kernel '{kernel}' (Permission 'kernels.get' was denied).")
        witness=SimpleNamespace(ref="owner/williams-ai-v5-book-sft-20261111",id=1)
        first=SimpleNamespace(ref="",slug="",title="Legacy Notebook Title",id=2)
        second=SimpleNamespace(ref="",slug="",title="Legacy Notebook Title",id=3)
        def inventory(**kw):
            return SimpleNamespace(kernels=[witness,first,second],next_page_token=None)
        api=SimpleNamespace(kernels_status=denied,kernels_list_with_response=inventory)
        with self.assertRaisesRegex(RuntimeError,"ABSENCE_NOT_VERIFIED"):
            status_absent(api,"owner/new-kernel-target")


    def test_duplicate_same_positive_id_is_deduplicated_as_one_kernel(self):
        def denied(kernel):
            raise ValueError(f"Cannot access kernel '{kernel}' (Permission 'kernels.get' was denied).")
        witness=SimpleNamespace(ref="owner/williams-ai-v5-book-sft-20261111",id=1)
        duplicate=SimpleNamespace(ref="OWNER/williams-ai-v5-book-sft-20261111",id=1)
        def inventory(**kw):
            return SimpleNamespace(kernels=[witness,duplicate],next_page_token=None)
        api=SimpleNamespace(kernels_status=denied,kernels_list_with_response=inventory)
        self.assertTrue(status_absent(api,"owner/new-kernel-target"))


    def test_duplicate_same_slug_without_numeric_ids_is_one_owner_kernel(self):
        def denied(kernel):
            raise ValueError(f"Cannot access kernel '{kernel}' (Permission 'kernels.get' was denied).")
        witness=SimpleNamespace(ref="owner/williams-ai-v5-book-sft-20261111",id=1)
        first=SimpleNamespace(ref="",slug="",title="Legacy Notebook Title",id=0)
        second=SimpleNamespace(ref="",slug="",title="Legacy Notebook Title",id=0)
        def inventory(**kw):
            return SimpleNamespace(kernels=[witness,first,second],next_page_token=None)
        api=SimpleNamespace(kernels_status=denied,kernels_list_with_response=inventory)
        self.assertTrue(status_absent(api,"owner/new-kernel-target"))


    def test_inventory_completes_after_legacy_100_page_cap(self):
        def denied(kernel):
            raise ValueError(f"Cannot access kernel '{kernel}' (Permission 'kernels.get' was denied).")
        witness=SimpleNamespace(ref="owner/williams-ai-v5-book-sft-20261111",id=1)
        calls=0
        def inventory(**kw):
            nonlocal calls
            calls+=1
            if calls==1:
                return SimpleNamespace(kernels=[witness],next_page_token="page-1")
            if calls<=101:
                return SimpleNamespace(kernels=[],next_page_token=f"page-{calls}")
            return SimpleNamespace(kernels=[],next_page_token=None)
        api=SimpleNamespace(kernels_status=denied,kernels_list_with_response=inventory)
        self.assertTrue(status_absent(api,"owner/new-kernel-target"))
        self.assertEqual(calls,102)

    def test_inventory_page_cap_fails_closed(self):
        def denied(kernel):
            raise ValueError(f"Cannot access kernel '{kernel}' (Permission 'kernels.get' was denied).")
        witness=SimpleNamespace(ref="owner/williams-ai-v5-book-sft-20261111",id=1)
        calls=0
        def inventory(**kw):
            nonlocal calls
            calls+=1
            return SimpleNamespace(kernels=[witness] if calls==1 else [],
                                   next_page_token=f"page-{calls}")
        api=SimpleNamespace(kernels_status=denied,kernels_list_with_response=inventory)
        with self.assertRaisesRegex(RuntimeError,"ABSENCE_NOT_VERIFIED"):
            status_absent(api,"owner/new-kernel-target")
        self.assertEqual(calls,1000)
