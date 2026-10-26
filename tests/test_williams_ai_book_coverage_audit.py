"""Source page coverage is verified on synthetic private indexes only."""
import sqlite3
import tempfile
from pathlib import Path
import unittest
from williams_ai.book_coverage_audit import page_coverage


class Coverage(unittest.TestCase):
    def test_missing_scanned_pages_remain_gaps_even_when_other_pages_reviewed(self):
        reg={"books":[{"id":ed,"pages":3,"sha256":ed*4} for ed in
                      ("TC1","NTD","TC2","COURSE")],
             "rule_cards":[{"source":"TC2","id":"TC2.WM3",
                            "status":"REVIEWED_PAGES","pdf_pages":[1]}]}
        with tempfile.TemporaryDirectory() as scratch:
            path=Path(scratch)/"private_books.db"
            with sqlite3.connect(path) as db:
                db.execute("CREATE TABLE volumes(book TEXT,sha256 TEXT,total_pages INTEGER,"
                           "extracted_pages INTEGER,missing_pages INTEGER,source_type TEXT)")
                db.execute("CREATE TABLE excerpts(book TEXT,pdf_page INTEGER,content TEXT,"
                           "evidence_status TEXT)")
                for ed in ("TC1","NTD","TC2","COURSE"):
                    db.execute("INSERT INTO volumes VALUES (?,?,?,?,?,?)",
                               (ed,ed*4,3,1,2,"PDF_TEXT"))
                    db.execute("INSERT INTO excerpts VALUES (?,?,?,?)",
                               (ed,1,"private synthetic", "TEST"))
                db.commit()
            result=page_coverage(path,registry_data=reg)
            self.assertEqual(result["books"]["TC2"]["pages_without_text"],2)
            self.assertEqual(result["books"]["TC2"]["bookmarks"][0]["pdf_page"],2)
            self.assertIsNone(result["book_mastery_percent"])
            self.assertFalse(result["execution_allowed"])

    def test_unverified_identity_does_not_count_any_page(self):
        reg={"books":[{"id":ed,"pages":1,"sha256":ed*4}
                      for ed in ("TC1","NTD","TC2","COURSE")],"rule_cards":[]}
        with tempfile.TemporaryDirectory() as scratch:
            path=Path(scratch)/"private_books.db"
            with sqlite3.connect(path) as db:
                db.execute("CREATE TABLE volumes(book TEXT,sha256 TEXT,total_pages INTEGER,"
                           "extracted_pages INTEGER,missing_pages INTEGER,source_type TEXT)")
                db.execute("CREATE TABLE excerpts(book TEXT,pdf_page INTEGER,content TEXT,"
                           "evidence_status TEXT)")
                for ed in ("TC1","NTD","TC2","COURSE"):
                    db.execute("INSERT INTO volumes VALUES (?,?,?,?,?,?)",
                               (ed,"mismatch",1,1,0,"PDF_TEXT"))
                db.commit()
            out=page_coverage(path,registry_data=reg)
            self.assertFalse(out["books"]["TC1"]["verified_local_source_identity"])
            self.assertEqual(out["books"]["TC1"]["pages_with_text"],0)


if __name__=="__main__":unittest.main()
