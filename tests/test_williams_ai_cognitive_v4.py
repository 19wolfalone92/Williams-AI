"""Cognitive v4 integration contract tests: synthetic only; no actual Qwen or gold."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from williams_ai.cognitive_memory_v4 import CognitiveMemory
from williams_ai.cognitive_engine_v4 import (
    CognitiveResearchEngine, LocalQwenResearchBackend,
)


def now():
    return datetime.now(timezone.utc).isoformat()


def earlier():
    return (datetime.now(timezone.utc)-timedelta(hours=2)).isoformat()


def toy_embedding(value):
    text=value.casefold()
    # Explicit deterministic test double; not a trained semantic encoder.
    if "слабе" in text or "momentum fade" in text:
        return [1.,0.,0.]
    if "correction" in text or "коррекц" in text:
        return [0.,1.,0.]
    return [0.,0.,1.]


class FakeQwen:
    def __init__(self,*,conflict=False):
        self.calls=[];self.conflict=conflict
    def __call__(self,stage,packet):
        self.calls.append(stage)
        if stage=="hypothesis":
            return {"claim":"Closing fractal offers an unconfirmed research idea.",
                    "alternative":"The apparent pattern is a continuing correction.",
                    "evidence_ids":["H1_AO"],
                    "counterevidence_ids":["H4_RANGE"],
                    "rule_ids":["TC2.WM3"],"confidence":0.44,
                    "invalidation":"Closed H4 weakness invalidates the idea."}
        if stage=="skeptic":
            return {"counterhypothesis":"The apparent reversal lacks a second confirmation.",
                    "challenged_fact_ids":["H1_AO"] if self.conflict else [],
                    "missing_information":["No order acknowledgements or positions."],
                    "rule_conflicts":[],
                    "suggestion":"WAIT" if self.conflict else "HOLD"}
        if stage=="arbiter":
            return {"stance":"RESEARCH_HYPOTHESIS","direction":"UP",
                    "supported_fact_ids":["H1_AO"],
                    "reason":"This remains a hypothetical closed-bar research scenario.",
                    "confidence":0.3}
        raise AssertionError("unexpected stage")


def facts():
    return [
      {"fact_id":"H1_AO","closed_at":earlier(),"source":"DERIVED:AO-H1",
       "observation":"AO rose across a confirmed completed candle.",
       "closed_bar_verified":True},
      {"fact_id":"H4_RANGE","closed_at":earlier(),"source":"CLOSED_OHLCV:H4",
       "observation":"H4 price range contradicts the unconfirmed reversal.",
       "closed_bar_verified":True},
    ]


class CognitiveV4Tests(unittest.TestCase):
    def setUp(self):
        self.folder=TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.memory=CognitiveMemory(Path(self.folder.name)/"private_research.sqlite3")
        self.model_id="gguf-model-parent-001"
        self.run_id="validation-replay-001"

    def test_memory_is_versioned_append_only_and_no_past_ingestion(self):
        key=self.memory.put(kind="RESEARCH_NOTE",edition="TC2",
                source_ref="USER:note-12345",
                text="The impulse was weakening in the last confirmed close.",
                observed_at=earlier())
        self.assertEqual(self.memory.retrieve(
                "impulse",edition="TC2",as_of=earlier()),[])
        found=self.memory.retrieve(
                "impulse",edition="TC2",as_of=now())
        self.assertEqual(found[0]["entry_id"],key)
        self.assertFalse(found[0]["expert_gold"])
        self.assertTrue(found[0]["lexical_match"])
        self.assertFalse(found[0]["model_scoped"])
        self.assertEqual(self.memory.stats()["entries"],1)

    def test_real_coaching_mistakes_recalled_but_not_retroactively(self):
        from williams_ai.adaptive_coach import notebook
        with notebook(self.memory.path) as db:
            db.execute("INSERT INTO private_remediation VALUES(?,?,?,?,?,?,?)",
                (self.model_id,"TC2","TC2.WM3","TC2:RULE:TC2.WM3",
                 "A previous fractal confirmation error needs review.",2,
                 "2026-01-10T00:00:00+00:00"))
        report=self.memory.import_previous_coaching_errors(
            model_id=self.model_id,run_id=self.run_id)
        self.assertEqual(report["eligible_scoped_lessons"],1)
        self.assertFalse(report["imported_as_expert_gold"])
        self.assertEqual(self.memory.retrieve(
            "fractal confirmation",edition="TC2",
            as_of="2026-02-01T00:00:00Z",
            model_id=self.model_id,run_id=self.run_id),[])
        results=self.memory.retrieve(
            "fractal confirmation",edition="TC2",
            as_of=(datetime.now(timezone.utc)+timedelta(seconds=5)).isoformat(),
            model_id=self.model_id,run_id=self.run_id)
        self.assertEqual(len(results),1)
        self.assertEqual(results[0]["status"],
                         "UNVERIFIED_RESEARCH_NOT_BOOK_AUTHORITY")
        self.memory.import_previous_coaching_errors(
            model_id=self.model_id,run_id=self.run_id)
        self.assertEqual(self.memory.stats()["entries"],1)
        self.assertEqual(self.memory.retrieve(
            "fractal",edition="TC2",
            as_of=(datetime.now(timezone.utc)+timedelta(seconds=5)).isoformat(),
            model_id="gguf-other-model-001",run_id=self.run_id),[])

    def test_russian_inflection_fallback_is_scoped_and_explicitly_weak(self):
        self.memory.put(kind="HYPOTHESIS",edition="TC2",
            source_ref="RESEARCH:past-fractal-setup",
            text="Неподтверждённая геометрия фракталов на закрытых H1 свечах.",
            observed_at=earlier(),model_id=self.model_id,run_id=self.run_id)
        old_cutoff=self.memory.retrieve(
            "фрактала",edition="TC2",as_of=earlier(),
            model_id=self.model_id,run_id=self.run_id)
        self.assertEqual(old_cutoff,[])
        now_cutoff=(datetime.now(timezone.utc)+timedelta(seconds=5)).isoformat()
        result=self.memory.retrieve(
            "фрактала",edition="TC2",as_of=now_cutoff,
            model_id=self.model_id,run_id=self.run_id)
        self.assertEqual(len(result),1)
        self.assertTrue(result[0]["lexical_match"])
        self.assertEqual(result[0]["lexical_match_type"],
                         "LOW_CONFIDENCE_INFLECTION_PREFIX")
        self.assertFalse(result[0]["semantic_match"])
        self.assertFalse(result[0]["expert_gold"])
        wrong_model=self.memory.retrieve(
            "фрактала",edition="TC2",as_of=now_cutoff,
            model_id="gguf-another-model-100",run_id=self.run_id)
        self.assertEqual(wrong_model,[])

    def test_exact_matches_beats_prefix_only_when_both_recall(self):
        self.memory.put(kind="RESEARCH_NOTE",edition="TC2",
            source_ref="RESEARCH:exact-spell",
            text="Fractals may be candidate structures without confirmed fills.",
            observed_at=earlier())
        self.memory.put(kind="RESEARCH_NOTE",edition="TC2",
            source_ref="RESEARCH:plural-prefix",
            text="Fractal observations also need historical validation.",
            observed_at=earlier())
        results=self.memory.retrieve("fractals",edition="TC2",
            as_of=(datetime.now(timezone.utc)+timedelta(seconds=5)).isoformat(),
            limit=5)
        self.assertEqual(len(results),2)
        self.assertEqual(results[0]["lexical_match_type"],"EXACT_FTS_TERM")
        self.assertEqual(results[1]["lexical_match_type"],
                         "LOW_CONFIDENCE_INFLECTION_PREFIX")

    def test_semantic_recall_finds_paraphrase_no_word_overlap(self):
        self.memory.put(kind="RESEARCH_NOTE",edition="TC2",
            source_ref="RESEARCH:learning-0001",
            text="Ослабевание импульса на закрытой свече, без подтверждения входа.",
            observed_at=earlier(),
            embedding_encoder=toy_embedding,
            embedding_model_id="local-test-embedder-v1")
        lexical=self.memory.retrieve("momentum fade",edition="TC2",as_of=now())
        self.assertEqual(lexical,[])
        semantic=self.memory.retrieve(
            "momentum fade",edition="TC2",as_of=now(),
            embedding_encoder=toy_embedding,
            embedding_model_id="local-test-embedder-v1")
        self.assertEqual(len(semantic),1)
        self.assertTrue(semantic[0]["semantic_match"])
        self.assertFalse(semantic[0]["expert_gold"])
        with self.assertRaisesRegex(ValueError,"EMBEDDING_FINGERPRINT"):
            self.memory.retrieve("momentum fade",edition="TC2",as_of=now(),
                embedding_encoder=toy_embedding)

    def test_model_and_run_isolation_and_graph_edge(self):
        parent=self.memory.put(kind="HYPOTHESIS",edition="TC2",
            source_ref="RESEARCH:study-parent",
            text="First hypothesis from this specific local model.",
            observed_at=earlier(),model_id=self.model_id,run_id=self.run_id)
        another=self.memory.put(kind="HYPOTHESIS",edition="TC2",
            source_ref="RESEARCH:study-parent-2",
            text="Follow up hypothesis invalidating earlier forecast.",
            observed_at=earlier(),model_id=self.model_id,run_id=self.run_id)
        link=self.memory.link(parent,another,relation="CONTRADICTS",
                 justification="Subsequent evidence contradicts the older hypothesis.")
        self.assertEqual(len(link),64)
        self.assertEqual(self.memory.related(parent,as_of=now(),
                model_id=self.model_id,run_id=self.run_id)[0]["relation"],
                         "CONTRADICTS")
        with self.assertRaisesRegex(ValueError,"ANCHOR_NOT_VISIBLE"):
            self.memory.related(parent,as_of=now(),
                model_id="gguf-other-model-001",run_id=self.run_id)
        self.assertEqual(self.memory.retrieve("hypothesis",edition="TC2",
                as_of=now(),model_id="gguf-other-model-001",
                run_id=self.run_id),[])

    def test_no_cross_model_evidence_link(self):
        a=self.memory.put(kind="REVIEWED_ERROR",edition="TC2",
            source_ref="REVIEW:case-first",
            text="One documented error category with declared reviewers.",
            observed_at=earlier(),model_id=self.model_id,run_id=self.run_id)
        b=self.memory.put(kind="REVIEWED_ERROR",edition="TC2",
            source_ref="REVIEW:case-other",
            text="Another model's error must never contaminate this memory.",
            observed_at=earlier(),model_id="gguf-model-child-002",run_id=self.run_id)
        with self.assertRaisesRegex(ValueError,"CROSS_MODEL"):
            self.memory.link(a,b,relation="CORRECTS",
                justification="Unauthorized correction across competing model experiments.")

    def test_book_pointer_not_source_authentication(self):
        with self.assertRaisesRegex(ValueError,"BOOK_REFERENCE_MUST_CITE_EDITION"):
            self.memory.put(kind="BOOK_REFERENCE",edition="COURSE",
                source_ref="TC2:P56",
                text="A local OCR summary that cannot certify source meaning.",
                observed_at=earlier())
        entry=self.memory.put(kind="BOOK_REFERENCE",edition="COURSE",
                source_ref="COURSE:P56",
                text="OCR and diagram labels need independent source validation.",
                observed_at=earlier())
        self.assertEqual(self.memory.retrieve("diagram",edition="COURSE",
                as_of=now())[0]["entry_id"],entry)
        self.assertEqual(self.memory.retrieve("diagram",edition="TC2",
                as_of=now()),[])

    def test_three_stage_same_qwen_research_never_trades(self):
        brain=FakeQwen()
        engine=CognitiveResearchEngine(memory=self.memory,backend=brain,
                model_id=self.model_id,run_id=self.run_id)
        result=engine.reason("фрактал",edition="TC2",as_of=now(),
                             closed_facts=facts())
        self.assertEqual(brain.calls,["hypothesis","skeptic","arbiter"])
        self.assertEqual(result["effective_stance"],"RESEARCH_HYPOTHESIS")
        self.assertFalse(result["execution_allowed"])
        self.assertIsNone(result["trade_action"])
        self.assertTrue(result["books_are_pointers_not_certified_understanding"])
        self.assertEqual(result["skeptic_independence"],
                         "SAME_MODEL_ROLE_NOT_INDEPENDENT_EXPERT")

    def test_critic_veto_unsupported_rule_and_future_data(self):
        brain=FakeQwen(conflict=True)
        engine=CognitiveResearchEngine(memory=self.memory,backend=brain,
                model_id=self.model_id,run_id=self.run_id)
        r=engine.reason("фрактал",edition="TC2",as_of=now(),
                        closed_facts=facts())
        self.assertEqual(r["effective_stance"],"WAIT")
        self.assertEqual(r["effective_direction"],"UNDETERMINED")
        self.assertIn("CHALLENGED_CORE_EVIDENCE",r["veto_reasons"])
        future=facts()
        future[0]["closed_at"]=(datetime.now(timezone.utc)+timedelta(days=1)).isoformat()
        with self.assertRaisesRegex(ValueError,"FUTURE_MARKET_FACT"):
            engine.reason("фрактал",edition="TC2",as_of=now(),
                          closed_facts=future)

    def test_cannot_fabricate_book_rule_or_market_fact(self):
        class Liar(FakeQwen):
            def __call__(self,stage,packet):
                out=super().__call__(stage,packet)
                if stage=="hypothesis":
                    out["evidence_ids"]=["FAKE_FUTURE_PRICE"]
                return out
        engine=CognitiveResearchEngine(memory=self.memory,backend=Liar(),
                model_id=self.model_id,run_id=self.run_id)
        with self.assertRaisesRegex(ValueError,"FABRICATED_HYPOTHESIS"):
            engine.reason("фрактал",edition="TC2",as_of=now(),
                          closed_facts=facts())

    def test_cognitive_model_adapter_uses_existing_runtime_not_loads(self):
        class Local:
            def __init__(self):
                self._model=self
                self.loads=1
            def create_chat_completion(self,**kwargs):
                self.last=kwargs
                return {"choices":[{"finish_reason":"stop",
                          "message":{"content":json.dumps(
                              {"stance":"WAIT","direction":"UNDETERMINED",
                               "supported_fact_ids":[],
                               "reason":"Research insufficient, cannot propose an order.",
                               "confidence":0.2})}}]}
        local=Local()
        adapter=LocalQwenResearchBackend(local)
        out=adapter("arbiter",{"test_only":True})
        self.assertEqual(out["stance"],"WAIT")
        self.assertEqual(local.loads,1)
        self.assertIn("system",local.last["messages"][0]["role"])


if __name__=="__main__":
    unittest.main()
