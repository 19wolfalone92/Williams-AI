"""Never launder auto-generated book advice into approved weight updates."""
import json
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from williams_ai.private_review_queue import propose_private_book_review_queue
from williams_ai.private_lora import checked_private_training_data


class QueueTests(unittest.TestCase):
    def test_four_books_split_whole_rules_and_block_training_until_reviewed(self):
        with TemporaryDirectory() as d:
            file=Path(d)/"queue.jsonl"
            result=propose_private_book_review_queue(file)
            self.assertEqual(result["proposals"],148)  # 37 reviewed source cards x 4 questions
            self.assertEqual(set(result["source_editions"]),
                             {"TC1","NTD","TC2","COURSE"})
            self.assertFalse(result["all_examples_human_reviewed"])
            lines=[json.loads(line) for line in file.read_text().splitlines()]
            self.assertEqual(len(lines),148)
            by_rule=defaultdict(set)
            for row in lines:
                by_rule[row["rule_id"]].add(row["split"])
                self.assertEqual(row["review_status"],"PENDING_INDEPENDENT_REVIEW")
                self.assertNotEqual(row["split"],"future")
            self.assertTrue(all(len(parts)==1 for parts in by_rule.values()))
            self.assertTrue(all(result["splits"][k]>=16
                for k in ("train","validation","test")))
            with self.assertRaisesRegex(ValueError,"UNREVIEWED_SOURCE_LABEL"):
                checked_private_training_data(file)
            with self.assertRaisesRegex(ValueError,"NEW_PRIVATE_JSONL_DESTINATION"):
                propose_private_book_review_queue(file)

if __name__=="__main__":
    unittest.main()
