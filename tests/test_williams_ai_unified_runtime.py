"""Adversarial integration tests. Fixture GGUF bytes are NOT trained weights."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from williams_ai.book_index import index_pages
from williams_ai.book_rules import registry
from williams_ai.causal_paper_lab import frozen_markets
from williams_ai.chart_reader import analyze_chart
from williams_ai.local_model import LocalWilliamsModel
from williams_ai.model_bundle import FORMAT, file_sha256, runtime_identity
from williams_ai.study import memory, recall
from williams_ai.study_cli import _model_identity
from williams_ai.unified_ai import WilliamsAI
from williams_ai.unified_memory import UnifiedMemory


def fake_bundle(root):
    """Synthetic provenance receipt exercises wiring, not scientific truth."""
    base=root/"base.gguf";base.write_bytes(b"GGUF FAKE BASE FOR INJECTED TEST RUNTIME")
    adapter=root/"adapter.gguf";adapter.write_bytes(b"GGUF FAKE ADAPTER NOT REAL TRAINED WEIGHTS")
    receipt=root/"research_manifest.json"
    receipt.write_text(json.dumps({"mode":"LOCAL_RESEARCH_LORA_NOT_TRADING_READY",
        "optimizer_steps":1,"source_dataset_sha256":"c"*64,
        "adapter_weights_sha256":"a"*64,"adapter_config_sha256":"b"*64,
        "baseline_validation_loss":2.0,"trained_validation_loss":1.0,
        "test_labels_not_supplied_to_trainer":True,"execution_allowed":False,
        "trade_action":None,"source_only_research":False}))
    manifest=root/"runtime_manifest.json"
    manifest.write_text(json.dumps({"format":FORMAT,"base_sha256":file_sha256(base),
        "adapter_sha256":file_sha256(adapter),"scale":1.0,
        "training_receipt_sha256":file_sha256(receipt)}))
    return base,adapter,manifest


class Runtime:
    instances=[]
    metadata={"general.architecture":"qwen3"}
    def __init__(self,**kwargs):
        self.options=kwargs;self.calls=[];self.instances.append(self)
    def create_chat_completion(self,**kwargs):
        self.calls.append(kwargs)
        payload=json.loads(kwargs["messages"][1]["content"])
        if "passage" in payload:
            result={"summary":"Фрактал и стоп требуют проверки исходной страницы.",
                "claims":[{"statement":"Нужны закрытые свечи.","support":payload["passage"][:45]}],
                "questions":[]}
        else:
            chart=payload["chart"]
            result={"symbol":chart["symbol"],"as_of":chart["as_of"],"edition":payload["edition"],
                "setup":"NO_SETUP","side":"NONE","entry_state":"NO_SETUP",
                "exit_basis":"NONE","rule_ids":[],"learning_need":"NONE",
                "reason":"A fixture abstention, not a real neural result.",
                "execution_allowed":False,"trade_action":None}
        return {"choices":[{"finish_reason":"stop","message":{"content":json.dumps(result)}}]}


class UnifiedRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.base,self.adapter,self.manifest=fake_bundle(self.root)
        self.db=self.root/"mind.sqlite3"
        self.books=self.root/"books.sqlite3"
        for ed in registry()["books"]:
            pages=[""]*ed["pages"]
            pages[9]="Фрактал и защитный стоп проверяются по закрытым свечам. "*3
            pages[130 if ed["id"]=="TC2" else 10]="Сопровождение стопа не подтверждает биржевую заявку. "*3
            index_pages(self.books,ed["id"],ed["sha256"],pages,declared_pages=ed["pages"])
        Runtime.instances=[]

    def brain(self,adapted=True,**kwargs):
        return WilliamsAI(local_model=self.base,memory=self.db,book_index=self.books,
            lora_adapter=self.adapter if adapted else None,
            lora_manifest=self.manifest if adapted else None,loader=Runtime,**kwargs)

    def test_base_identity_preserves_old_checkpoints_and_adapter_gets_its_own_identity(self):
        plain=runtime_identity(self.base)
        bound=runtime_identity(self.base,adapter=self.adapter,manifest=self.manifest)
        self.assertEqual(plain["model_id"],"gguf-"+file_sha256(self.base)[:24])
        self.assertEqual(plain["model_id"],_model_identity(self.base))
        self.assertNotEqual(plain["model_id"],bound["model_id"])
        self.assertFalse(bound["adapter_loaded"])
        self.assertFalse(bound["training_origin_independently_verified"])

    def test_study_and_chart_use_one_effective_model_and_one_durable_memory(self):
        brain=self.brain()
        brain.memory.retain("Стоп и фрактал требуют проверки закрытых свечей.",source="user-note",
            title="Памятка",edition="TC2",available_at="2026-01-01T00:00:00Z")
        study=brain.study(editions=("TC2",),budget_per_book=1)
        self.assertEqual(study["new_study_chunks"],1)
        market=frozen_markets()["ETHUSDT"]
        result=brain.analyze("ETHUSDT",market["bars"],tick_size=market["tick_size"],
            bar_index=299,include_topdown=False)
        self.assertEqual(len(Runtime.instances),1)
        runtime=Runtime.instances[0]
        self.assertEqual(runtime.options["lora_path"],str(self.adapter))
        self.assertEqual(runtime.options["lora_scale"],1.0)
        self.assertTrue(result["unified_ai"]["effective_model"]["adapter_loaded"])
        self.assertEqual(result["unified_ai"]["effective_model"]["model_id"],study["model_fingerprint_id"])
        self.assertTrue(result["private_research_material_refs_used"])
        self.assertEqual(result["deliberate_book_learning"]["past_study_sources_reread"],["TC2: PDF 10"])
        prompt=json.loads(runtime.calls[-1]["messages"][1]["content"])
        self.assertTrue(prompt["retained_research_materials_not_authoritative_rules"])
        self.assertNotIn("Непроверенный учебный вывод",json.dumps(prompt,ensure_ascii=False))
        self.assertEqual(brain.status()["memory_counts"]["study_notes"],1)
        resumed=self.brain()
        self.assertFalse(resumed.status()["runtime_loaded"])
        self.assertEqual(resumed.status()["memory_counts"]["study_notes"],1)
        self.assertFalse(result["execution_allowed"])
        self.assertIsNone(result["trade_action"])

    def test_another_adapter_cannot_claim_previous_student_notes(self):
        brain=self.brain()
        brain.study(editions=("TC2",),budget_per_book=1)
        old=brain.identity["model_id"]
        self.adapter.write_bytes(b"GGUF DIFFERENT SYNTHETIC ADAPTER")
        data=json.loads(self.manifest.read_text());data["adapter_sha256"]=file_sha256(self.adapter)
        self.manifest.write_text(json.dumps(data))
        newer=self.brain()
        self.assertNotEqual(old,newer.identity["model_id"])
        self.assertEqual(recall(self.db,"фрактал стоп",edition="TC2",model_id=newer.identity["model_id"],
            book_index=self.books),[])
        self.assertEqual(newer.status()["memory_counts"]["study_notes"],1)
        self.assertEqual(newer.status()["book_study"]["books"][2]["studied_chunks"],0)

    def test_old_study_pointer_reloads_original_and_rejects_changed_ocr(self):
        from williams_ai.study import recalled_source_pages
        brain=self.brain();brain.study(editions=("TC2",),budget_per_book=1)
        pages=recalled_source_pages(self.books,self.db,"фрактал стоп",edition="TC2",model_id=brain.identity["model_id"])
        self.assertEqual([x["source"] for x in pages],["TC2: PDF 10"])
        self.assertIn("Фрактал и защитный стоп",pages[0]["excerpt"])
        self.assertNotIn("требуют проверки исходной страницы",pages[0]["excerpt"])
        with sqlite3.connect(self.books) as db:
            db.execute("UPDATE excerpts SET content='Исправленный OCR фрактал стоп' WHERE book='TC2' AND pdf_page=10")
        self.assertEqual(recalled_source_pages(self.books,self.db,"фрактал стоп",edition="TC2",
            model_id=brain.identity["model_id"]),[])

    def test_ntd_zone_lesson_can_select_its_original_in_unified_chart_analysis(self):
        brain=self.brain()
        text="Zone colors green red gray depend on Awesome Oscillator and Acceleration."
        with sqlite3.connect(self.books) as db:
            db.execute("INSERT INTO excerpts VALUES('NTD',11,?,'UNVERIFIED_EXTRACTION')",(text,))
        ed=next(x for x in registry()["books"] if x["id"]=="NTD")
        with memory(self.db) as db:
            result={"summary":"Zone colors green red gray.","claims":[],"questions":[],
                    "status":"SOURCE_DERIVED_UNVERIFIED","trade_action":None}
            db.execute("INSERT INTO study_notes VALUES(?,?,?,?,?,?,?)",('NTD',ed['sha256'],11,
                hashlib.sha256(text.encode()).hexdigest(),brain.identity['model_id'],json.dumps(result),'2026-01-01'))
        market=frozen_markets()["ETHUSDT"]
        result=brain.analyze('ETHUSDT',market['bars'],tick_size=market['tick_size'],bar_index=299,
            edition='NTD',include_topdown=False)
        self.assertEqual(result['deliberate_book_learning']['past_study_sources_reread'],['NTD: PDF 11'])

    def test_missing_mixed_or_changed_adapter_stops_before_native_loader(self):
        for key,value in (("base_sha256","0"*64),("adapter_sha256","1"*64),
                          ("training_receipt_sha256","2"*64),("scale",True)):
            data=json.loads(self.manifest.read_text());changed={**data,key:value}
            self.manifest.write_text(json.dumps(changed))
            with self.subTest(key=key),self.assertRaises(ValueError):self.brain()
            self.manifest.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            runtime_identity(self.base,adapter=self.adapter)
        self.assertEqual(Runtime.instances,[])

    def test_incomplete_training_receipt_is_never_activated(self):
        receipt=self.root/"research_manifest.json"
        obj=json.loads(receipt.read_text());obj["optimizer_steps"]=0;receipt.write_text(json.dumps(obj))
        m=json.loads(self.manifest.read_text());m["training_receipt_sha256"]=file_sha256(receipt)
        self.manifest.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,"COMPLETED_RESEARCH"):
            self.brain()
        self.assertEqual(Runtime.instances,[])

    def test_native_adapter_failure_is_not_silently_retried_without_lora(self):
        calls=[]
        def broken(**kwargs):calls.append(kwargs);raise RuntimeError("ADAPTER_LOAD_FAILED")
        with self.assertRaisesRegex(RuntimeError,"ADAPTER_LOAD_FAILED"):
            LocalWilliamsModel(self.base,loader=broken,lora_adapter=self.adapter,lora_manifest=self.manifest)
        self.assertEqual(len(calls),1)
        self.assertIn("lora_path",calls[0])

    def test_effective_weights_change_between_configuration_and_load_is_rejected(self):
        brain=self.brain(adapted=False)
        self.base.write_bytes(b"GGUF CHANGED BASE BEFORE LOAD")
        with self.assertRaisesRegex(ValueError,"EFFECTIVE_MODEL_CHANGED"):_=brain.model
        self.assertTrue(all(not x.calls for x in Runtime.instances))

    def test_retained_materials_keep_the_end_of_long_documents_and_exclude_future_and_other_editions(self):
        memory=UnifiedMemory(self.db)
        text="Introduction without relevant terms. "*100+"Сопровождение: стоп и фрактал проверяются по закрытым свечам."
        key=memory.retain(text,source="private-wave-guide",title="Guide",edition="TC2",
            available_at="2026-01-01T00:00:00Z")
        memory.retain("Сопровождение стоп фрактал "*3,source="future",title="Future",edition="TC2",
            available_at="2026-10-22T00:00:00+03:00")
        memory.retain("Сопровождение стоп фрактал "*3,source="NTD",title="Another",edition="NTD",
            available_at="2026-01-01T00:00:00Z")
        rows=memory.retrieve("сопровождение стоп фрактал",edition="TC2",as_of="2026-10-20T00:00:00Z")
        self.assertEqual([x["source"] for x in rows],["MATERIAL:"+key])
        self.assertIn("Сопровождение",rows[0]["excerpt"])
        with memory.connect() as db:
            self.assertEqual(db.execute("SELECT text FROM retained_materials WHERE document_sha256=?",(key,)).fetchone()[0],text)

    def test_future_or_mixed_edition_material_is_rejected_by_actual_prompt_builder(self):
        market=frozen_markets()["ETHUSDT"];rows=market["bars"][:300]
        obs=analyze_chart("ETHUSDT","H1",rows,as_of=rows[-1]["close_time"],tick_size=market["tick_size"])
        model=LocalWilliamsModel(self.base,loader=Runtime)
        good={"source":"MATERIAL:"+"a"*64,"excerpt":"Стоп по закрытым свечам.","edition":"TC2",
              "available_at":"2026-01-01T00:00:00Z","status":"UNVERIFIED_RESEARCH_NOT_BOOK_AUTHORITY"}
        for item in ({**good,"edition":"NTD"},{**good,"available_at":"2027-01-01T00:00:00Z"}):
            with self.assertRaises(ValueError):model.propose_chart_reasoning(obs,research_materials=[item])
        self.assertEqual(model._model.calls,[])

    def test_tampered_material_never_enters_context(self):
        memory=UnifiedMemory(self.db)
        memory.retain("Фрактал стоп",source="notes",title="Guide",edition="TC2",available_at="2026-01-01T00:00:00Z")
        with memory.connect() as db:db.execute("UPDATE retained_materials SET text='Revised фрактал стоп'")
        with self.assertRaisesRegex(ValueError,"MATERIAL_CHANGED"):
            memory.retrieve("стоп",edition="TC2",as_of="2026-02-01T00:00:00Z")

    def test_checkpoint_import_is_repeatable_preserves_receipts_and_does_not_change_source(self):
        source=self.root/"legacy.sqlite3";legacy=UnifiedMemory(source)
        with legacy.connect() as db:
            payload=json.dumps({"result":"historical diagnostic"})
            db.execute("INSERT INTO actual_run_receipts VALUES(?,?,?)",("run",payload,hashlib.sha256(payload.encode()).hexdigest()))
        before=file_sha256(source)
        memory=UnifiedMemory(self.db)
        first=memory.import_checkpoint(source);second=memory.import_checkpoint(source)
        self.assertEqual(first["imported_rows"]["actual_run_receipts"],1)
        self.assertEqual(second["imported_rows"]["actual_run_receipts"],0)
        self.assertEqual(file_sha256(source),before)
        self.assertFalse(first["unverified_notes_promoted"])

    def test_checkpoint_import_rolls_back_all_tables_on_conflicting_old_memory(self):
        source=self.root/"legacy.sqlite3";legacy=UnifiedMemory(source);memory=UnifiedMemory(self.db)
        for store,text in ((legacy,"old"),(memory,"current")):
            with store.connect() as db:
                db.execute("INSERT INTO private_remediation VALUES(?,?,?,?,?,?,?)",
                    ("gguf-test","TC2","TC2.CAMPAIGN","TC2:RULE:TC2.CAMPAIGN",text,1,"2026-01-01"))
        with legacy.connect() as db:
            db.execute("INSERT INTO study_errors VALUES('chunk','gguf-test',1,'TEST_ERROR')")
        before=memory.inventory()
        with self.assertRaisesRegex(ValueError,"IMPORT_CONFLICT"):
            memory.import_checkpoint(source)
        self.assertEqual(memory.inventory(),before)


if __name__=="__main__":
    unittest.main()
