"""Root/propagation audit: late study, coach and research notes MUST NOT leak into old charts."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import sqlite3
import unittest
from tempfile import TemporaryDirectory

from williams_ai.adaptive_coach import notebook, recall_model_corrections
from williams_ai.book_rules import registry
from williams_ai.study import memory, recall
from williams_ai.unified_memory import UnifiedMemory


class RootCausalMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db=Path(self.temp.name)/"root_legacy_causal.sqlite3"
        self.model_id="gguf-root-causal-test"

    def test_retained_research_material_cannot_be_backdated(self):
        brain=UnifiedMemory(self.db)
        original_date="2026-01-01T00:00:00Z"
        brain.retain("Фрактал на закрытом баре: стоп не является исполненным ордером.",
                     source="user-test-fixture",title="Backdated user note",
                     edition="TC2",available_at=original_date)
        historical=brain.retrieve("фрактал",edition="TC2",
                                  as_of="2026-01-02T00:00:00Z")
        self.assertEqual(historical,[])
        now=(datetime.now(timezone.utc)+timedelta(seconds=5)).isoformat()
        self.assertEqual(len(brain.retrieve("фрактал",edition="TC2",as_of=now)),1)
        with sqlite3.connect(brain.path) as db:
            ingestions=db.execute(
                "SELECT ingested_at FROM retained_material_ingested_v4").fetchall()
        self.assertEqual(len(ingestions),1)

    def test_old_checkpoint_material_migrates_at_real_recording_time(self):
        # Older DB schemas did not store ingested_at; do not silently infer
        # original ingestion from user-controlled available_at.
        with sqlite3.connect(self.db) as db:
            db.execute("""CREATE TABLE retained_materials(
                document_sha256 TEXT PRIMARY KEY,edition TEXT,source TEXT,title TEXT,
                available_at TEXT,text TEXT,status TEXT)""")
            fields=["TC2","old-note","legacy","2025-01-01T00:00:00+00:00",
                    "Old unknown provenance note concerning fractal stop."]
            import hashlib
            sha=hashlib.sha256(json.dumps(fields,ensure_ascii=False,
                  separators=(",",":")).encode()).hexdigest()
            db.execute("INSERT INTO retained_materials VALUES(?,?,?,?,?,?,?)",
                 (sha,"TC2","old-note","legacy","2025-01-01T00:00:00+00:00",
                  fields[-1],"UNVERIFIED_RESEARCH_NOT_BOOK_AUTHORITY"))
        brain=UnifiedMemory(self.db)
        self.assertEqual(brain.retrieve("fractal",edition="TC2",
             as_of="2025-06-01T00:00:00Z"),[])
        later=(datetime.now(timezone.utc)+timedelta(seconds=5)).isoformat()
        self.assertEqual(len(brain.retrieve("fractal",edition="TC2",as_of=later)),1)

    def test_coach_correction_from_future_is_not_in_earlier_chart(self):
        with notebook(self.db) as db:
            db.execute("""INSERT INTO private_remediation VALUES (?,?,?,?,?,?,?)""",
                       (self.model_id,"TC2","TC2.WM3",
                        "TC2:RULE:TC2.WM3",
                        "Fractal confirmation was missing at previous cutoff.",
                        3,"2026-11-05T12:00:00+00:00"))
        before=recall_model_corrections(self.db,self.model_id,"TC2",
                 as_of="2026-11-01T12:00:00Z")
        self.assertEqual(recall_model_corrections(self.db,self.model_id,"TC2",
                 as_of="2026-11-06T12:00:00Z"),[])
        after=recall_model_corrections(self.db,self.model_id,"TC2",
                 as_of=(datetime.now(timezone.utc)+timedelta(seconds=5)).isoformat())
        self.assertEqual(before,[])
        self.assertEqual(len(after),1)
        self.assertIn("Fractal",after[0]["excerpt"])
        self.assertEqual(len(recall_model_corrections(
            self.db,self.model_id,"TC2")),1)

    def test_book_study_lesson_from_future_is_not_in_past(self):
        known=next(b for b in registry()["books"] if b["id"]=="TC2")
        result={"summary":"Фрактал является только гипотезой до подтверждения.",
                "claims":[],"questions":[],"status":"SOURCE_DERIVED_UNVERIFIED"}
        with memory(self.db) as db:
            db.execute("""INSERT INTO study_notes VALUES (?,?,?,?,?,?,?)""",
                ("TC2",known["sha256"],25,"a"*64,self.model_id,
                 json.dumps(result,ensure_ascii=False),
                 "2026-11-05T12:00:00+00:00"))
        before=recall(self.db,"фрактал",edition="TC2",
               model_id=self.model_id,as_of="2026-11-01T12:00:00Z")
        self.assertEqual(recall(self.db,"фрактал",edition="TC2",
               model_id=self.model_id,as_of="2026-11-06T12:00:00Z"),[])
        after=recall(self.db,"фрактал",edition="TC2",
               model_id=self.model_id,
               as_of=(datetime.now(timezone.utc)+timedelta(seconds=5)).isoformat())
        self.assertEqual(before,[])
        self.assertEqual(len(after),1)

    def test_non_utc_time_is_rejected_in_causal_recall(self):
        with notebook(self.db):
            pass
        with self.assertRaisesRegex(ValueError,"CAUSAL_COACH"):
            recall_model_corrections(self.db,self.model_id,"TC2",
                  as_of="2026-11-01 12:00:00")
        with self.assertRaisesRegex(ValueError,"CAUSAL_STUDY"):
            recall(self.db,"фрактал",edition="TC2",
                   model_id=self.model_id,as_of="2026-11-01 12:00:00")


if __name__=="__main__":
    unittest.main()
