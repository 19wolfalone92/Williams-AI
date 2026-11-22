"""V5 data and actual GPU trainer contracts; no fake training claims."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import sqlite3
import unittest

from williams_ai.knowledge_curriculum_v5 import (
    build,export_database,SCHEMA,SPLITS,EDITIONS,EXCLUDE,digest)
from williams_ai.knowledge_training_v5 import tokenize


class ByteTokenizerStub:
    """Only verifies Qwen chat-template answer masking contract, not LM skill."""
    def apply_chat_template(self,msgs,*,tokenize,add_generation_prompt,
                            enable_thinking):
        if not tokenize or enable_thinking:
            raise AssertionError("QWEN_NONTHINKING_ANSWER_TEMPLATE_REQUIRED")
        content="".join(m["role"]+":"+m["content"]+"\n" for m in msgs)
        if add_generation_prompt:
            content+="assistant:"
        return list(content.encode("utf-8"))


class KnowledgeCurriculumV5Tests(unittest.TestCase):
    def setUp(self):
        self.rows,self.manifest=build()

    def test_all_books_and_strict_frozen_rule_disjoint_splits(self):
        owners={}
        counts=Counter()
        for item in self.rows:
            key=(item["edition"],item["rule_id"])
            self.assertEqual(owners.setdefault(key,item["split"]),item["split"])
            counts[item["split"],item["edition"]]+=1
            self.assertEqual(item["schema"],SCHEMA)
            self.assertFalse(item["chart_expert_gold"])
            self.assertFalse(item["execution_allowed"])
            self.assertFalse(item["visual_semantics_independently_verified"])
            self.assertEqual(item["eligible_for_sft"],item["split"]=="train")
        for split in SPLITS:
            for book in EDITIONS:
                self.assertGreater(counts[split,book],0)
        self.assertEqual(len(self.rows),12*len(owners))
        self.assertEqual(sum(self.manifest["split_rows"].values()),len(self.rows))
        self.assertEqual(self.manifest["approved_market_labels"],0)

    def test_visual_only_scanned_course_is_not_teacher_or_gold(self):
        ids={item["rule_id"] for item in self.rows}
        self.assertTrue(EXCLUDE.isdisjoint(ids))
        self.assertEqual(self.manifest["independently_reviewed_diagrams"],0)
        self.assertFalse(self.manifest["source_pdf_text_or_images_in_dataset"])
        self.assertFalse(self.manifest["live_trading_allowed"])

    def test_compiled_prompts_unique_no_validation_or_test_as_gpu_train(self):
        hashes=[digest(item["prompt"]) for item in self.rows]
        self.assertEqual(len(hashes),len(set(hashes)))
        test_cases=[x for x in self.rows if x["split"]=="test"]
        with self.assertRaisesRegex(ValueError,"NEVER_PASS_TEST"):
            tokenize(test_cases,ByteTokenizerStub())
        train=[x for x in self.rows if x["split"]=="train"][:2]
        tensors=tokenize(train,ByteTokenizerStub(),max_tokens=2048)
        self.assertEqual(len(tensors),len(train))
        for row in tensors:
            self.assertEqual(len(row["input_ids"]),len(row["labels"]))
            self.assertEqual(len(row["input_ids"]),len(row["attention_mask"]))
            self.assertTrue(any(x==-100 for x in row["labels"]))
            self.assertTrue(any(x!=-100 for x in row["labels"]))

    def test_sft_mcq_targets_are_only_correct_labels_and_reason(self):
        for item in self.rows:
            answer=json.loads(item["answer"])
            if item["task_type"]=="MCQ":
                self.assertEqual(set(answer),{"choice","reason"})
                self.assertIn(answer["choice"],("A","B","C"))
                self.assertTrue(answer["reason"])
                self.assertIn("choices",item["prompt"])
                self.assertEqual(set(item["prompt"]["choices"]),{"A","B","C"})
            elif item["task_type"]=="BOUNDARY":
                self.assertEqual(answer["status"],"INSUFFICIENT_MARKET_CONFIRMATION")
                self.assertFalse(answer["execution_allowed"])
            else:
                self.assertEqual(answer["rule_id"],item["rule_id"])
                self.assertFalse(answer["execution_allowed"])

    def test_sqlite_output_is_actual_private_database_with_provenance(self):
        with TemporaryDirectory() as tmp:
            dbfile=Path(tmp)/"curriculum_v5.sqlite3"
            receipt=export_database(dbfile)
            self.assertEqual(receipt,self.manifest)
            self.assertEqual(dbfile.stat().st_mode & 0o077,0)
            with sqlite3.connect(dbfile) as db:
                cards=db.execute("SELECT count(*) FROM rules").fetchone()[0]
                examples=db.execute("SELECT count(*) FROM curriculum").fetchone()[0]
                eligible=db.execute("SELECT count(*) FROM curriculum "
                    "WHERE eligible_for_sft=1").fetchone()[0]
                bad=db.execute("SELECT count(*) FROM curriculum "
                    "WHERE expert_market_gold!=0").fetchone()[0]
                self.assertEqual(cards,len(self.rows)//12)
                self.assertEqual(examples,len(self.rows))
                self.assertEqual(eligible,self.manifest["split_rows"]["train"])
                self.assertEqual(bad,0)
                self.assertEqual(json.loads(db.execute(
                    "SELECT value_json FROM manifest WHERE key='excluded_rules'"
                    ).fetchone()[0]),["COURSE.VISUAL.NESTED_ABC"])
            with self.assertRaises(FileExistsError):
                export_database(dbfile)

    def test_source_rules_snapshot_deterministic(self):
        rows,manifest=build()
        self.assertEqual(rows,self.rows)
        self.assertEqual(manifest,self.manifest)
        self.assertEqual(len({x["case_id"] for x in rows}),len(rows))


if __name__=="__main__":
    unittest.main()
