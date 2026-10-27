from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from williams_ai.book_index import index_pages, search_pages, _preferred_source_pages
from williams_ai.book_rules import registry
from williams_ai.knowledge import query_terms


class PrivateSourceRetrievalTests(unittest.TestCase):
    def test_russian_zero_and_minimum_reach_english_source_terms(self):
        words = query_terms("четвёртая волна ноль минимальные требования")
        self.assertIn("zero", words)
        self.assertIn("minimum", words)
        self.assertIn("requirement", words)

    def test_specific_zero_evidence_beats_repetitive_generic_wave_page(self):
        with TemporaryDirectory() as d:
            p=Path(d)/"books.sqlite3"
            book=next(b for b in registry()["books"] if b["id"]=="COURSE")
            pages=[""]*book["pages"]
            pages[30]="General wave observation. "*80
            pages[43]="Synthetic source fixture: minimum requirements for fourth wave: oscillator crosses zero. This is not its completion."
            index_pages(p,"COURSE",book["sha256"],pages,declared_pages=book["pages"])
            hits=search_pages(p,"четвёртая волна ноль минимальные требования",edition="COURSE",limit=2)
            self.assertEqual(hits[0]["page"],44)
            self.assertEqual(len({h["page"] for h in hits}),2)
            self.assertIn("zero", hits[0]["excerpt"])
            self.assertTrue(all(h["status"]=="PRIVATE_PAGE_TEXT_UNVERIFIED" for h in hits))

    def test_different_pages_use_the_evidence_budget_and_missing_evidence_is_not_invented(self):
        with TemporaryDirectory() as d:
            p=Path(d)/"books.sqlite3"
            book=next(b for b in registry()["books"] if b["id"]=="COURSE")
            pages=[""]*book["pages"]
            pages[68]="Inside bars and thumb evidence. "*100
            pages[70]="Thumb summary and inside bars."
            index_pages(p,"COURSE",book["sha256"],pages,declared_pages=book["pages"])
            hits=search_pages(p,"thumb inside bars",edition="COURSE",limit=2)
            self.assertEqual({h["page"] for h in hits},{69,71})
            self.assertEqual(search_pages(p,"extraterrestrial volcanic geology",edition="COURSE"),[])

    def test_specific_bookmark_does_not_cross_edition_or_invent_generic_wave_citations(self):
        self.assertIn(("COURSE",44),_preferred_source_pages("четвёртая волна ноль","COURSE"))
        self.assertEqual(_preferred_source_pages("четвёртая волна ноль","TC2"),set())
        self.assertEqual(_preferred_source_pages("расскажи про волны","COURSE"),set())
