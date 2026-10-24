import unittest
from pathlib import Path
from williams_ai.book_rules import registry,coverage

ROOT=Path(__file__).resolve().parents[1]

class BookSafetyContracts(unittest.TestCase):
    def test_four_source_hashes_and_unknown_accuracy(self):
        self.assertEqual(len(registry()["books"]),4)
        for book in registry()["books"]:
            self.assertRegex(book["sha256"],r"^[a-f0-9]{64}$")
            self.assertIsNone(coverage()[book["id"]]["accuracy_percent"])

    def test_original_course_scanned_nested_wave_diagram_has_separate_source_card(self):
        notes=[card for card in registry()["rule_cards"]
               if card["id"]=="COURSE.VISUAL.NESTED_ABC"]
        self.assertEqual(len(notes),1)
        self.assertEqual(notes[0]["pdf_pages"],[56])
        self.assertEqual(notes[0]["source"],"COURSE")
        self.assertIn("не доказывает вход",notes[0]["summary"])
        self.assertNotIn("100%",notes[0]["summary"])

    def test_public_git_has_no_full_book_pdf(self):
        for folder in ("knowledge","williams_ai"):
            self.assertFalse(list((ROOT/folder).glob("*.pdf")))

    def test_chart_fixture_is_explicitly_synthetic(self):
        self.assertIn("SYNTHETIC EXAMPLE",
            (ROOT/"examples/ai_chart_example.json").read_text(encoding="utf-8"))

    def test_private_book_index_not_committed(self):
        self.assertFalse((ROOT/"knowledge/Williams_AI_Knowledge_Private.sqlite3").exists())

if __name__=="__main__":
    unittest.main()
