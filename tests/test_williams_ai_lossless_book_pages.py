"""Source indexing may never quietly discard the end of a long PDF page."""
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from williams_ai.book_index import _chunks,index_pages
from williams_ai.book_rules import registry

class LosslessSourceIndexTests(unittest.TestCase):
    def test_index_all_text_beyond_old_35000_character_cutoff(self):
        book=next(v for v in registry()["books"] if v["id"]=="TC2")
        sentence="Фрактал проверяется только после подтверждения. "
        raw=sentence*830+"ОСОБОЕ_КОНЕЧНОЕ_ПРАВИЛО"
        self.assertGreater(len(raw),35000)
        self.assertEqual("".join(_chunks(raw)),raw.strip())
        with TemporaryDirectory() as d:
            db=Path(d)/"private_books.db"
            pages=[""]*book["pages"]
            pages[10]=raw
            status=index_pages(db,"TC2",book["sha256"],pages,
                               declared_pages=book["pages"])
            self.assertEqual(status["nonempty_pages"],1)
            with sqlite3.connect(db) as cn:
                reconstructed="".join(v[0] for v in cn.execute(
                    "SELECT content FROM excerpts WHERE book=? AND pdf_page=11 ORDER BY rowid",
                    ("TC2",)))
            self.assertEqual(reconstructed,raw.strip())
            self.assertTrue(reconstructed.endswith("ОСОБОЕ_КОНЕЧНОЕ_ПРАВИЛО"))

    def test_oversize_page_rejected_not_partially_indexed(self):
        with self.assertRaisesRegex(ValueError,"SOURCE_PAGE_TOO_LARGE"):
            _chunks("x"*250001)

if __name__=="__main__":
    unittest.main()
