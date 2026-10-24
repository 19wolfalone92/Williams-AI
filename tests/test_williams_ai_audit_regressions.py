"""Adversarial audit cases; mocks never certify real model trading skill."""
import copy
from datetime import datetime,timedelta,timezone
import json
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from test_williams_ai_public_forward_capture import make_fetch,SERVER,LAST_OPEN,HOUR_MS
from test_williams_ai_prospective_qualification import rows,freeze
from test_williams_ai_private_lora import candidates
from williams_ai.public_forward_capture import capture_latest_closed_h1
from williams_ai.prospective_qualification import settle_prospective_call,summarize_sealed_results
from williams_ai.private_lora import checked_private_training_data,train_private_lora
from williams_ai.competency_exam import CASES,grade_exam
from williams_ai.adaptive_coach import coach_practice,learning_cases,notebook
from williams_ai.local_model import LocalWilliamsModel
from williams_ai.chart_reader import analyze_chart
from williams_ai.causal_paper_lab import frozen_markets
from test_williams_ai_demo_pilot import public_fixture,CLOCK
from williams_ai.demo_pilot_readiness import check_public_demo,_read_only_demo_json,DemoSafetyError
from williams_ai.binance_wave_history import _market_snapshot
from williams_ai.demo_futures import DemoHttp,DemoFutures,TradeProposal
from test_williams_ai_demo import FakeVenue


class AuditForecastEvidence(unittest.TestCase):
    def test_full_futures_catalog_selects_requested_contract_not_first_row(self):
        fetch,_,_=make_fetch()
        urls=[]
        def full_catalog(url):
            urls.append(url)
            if "/exchangeInfo" in url:
                target=fetch("https://fapi.binance.com/fapi/v1/exchangeInfo?symbol=BTCUSDT")["symbols"][0]
                return {"symbols":[{"symbol":"SOLUSDT","status":"BREAK","filters":[]},target]}
            return fetch(url)
        result=capture_latest_closed_h1("BTCUSDT",venue="FUTURES",fetch_json=full_catalog,clock_ms=SERVER)
        self.assertEqual(result["market"]["tick_size"],0.01)
        self.assertIn("https://fapi.binance.com/fapi/v1/exchangeInfo",urls)
        self.assertFalse(any("exchangeInfo?" in x for x in urls))

    def test_missing_most_recent_safe_closed_hour_is_rejected(self):
        _,original,_=make_fetch()
        fetch,_,_=make_fetch(rows=original[:-2])
        with self.assertRaisesRegex(ValueError,"NOT_LATEST"):
            capture_latest_closed_h1("BTCUSDT",venue="SPOT",fetch_json=fetch,clock_ms=SERVER)

    def test_five_second_close_guard_is_exact_at_hour_boundary(self):
        for elapsed,expected in ((0,LAST_OPEN-HOUR_MS-1),
                                 (4999,LAST_OPEN-HOUR_MS-1),
                                 (5000,LAST_OPEN-1),
                                 (5001,LAST_OPEN-1),
                                 (420000,LAST_OPEN-1)):
            clock=LAST_OPEN+elapsed
            fetch,_,_=make_fetch(clock=clock)
            result=capture_latest_closed_h1("BTCUSDT",venue="SPOT",fetch_json=fetch,clock_ms=clock)
            self.assertEqual(result["source"]["as_of"],
                             datetime.fromtimestamp(expected/1000,timezone.utc).isoformat())

    def test_inconsistent_cached_accuracy_flags_cannot_inflate_summary(self):
        full=rows(260)
        original=settle_prospective_call(freeze(full[:230],"DOWN"),full)
        for key,value in (("correct",True),("correct","False"),("called","yes"),
                          ("non_neutral",1),("baseline_correct_on_selected",False)):
            altered={**original,key:value}
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                summarize_sealed_results([altered])

    def test_abstention_cannot_be_converted_to_success_by_cached_flags(self):
        full=rows(260)
        out=settle_prospective_call(freeze(full[:230],"ABSTAIN"),full)
        out.update(called=True,correct=True,baseline_correct_on_selected=True)
        with self.assertRaises(ValueError):summarize_sealed_results([out])

    def test_timezone_alias_is_the_same_observation_not_an_extra_trial(self):
        full=rows(260)
        original=settle_prospective_call(freeze(full[:230]),full)
        alias=dict(original)
        alias["as_of"]=datetime.fromisoformat(original["as_of"]).astimezone(
            timezone(timedelta(hours=3))).isoformat()
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            summarize_sealed_results([original,alias])


class AuditReviewedTrainingData(unittest.TestCase):
    def test_dataset_digest_describes_the_frozen_rows_not_a_later_replacement(self):
        with TemporaryDirectory() as d:
            path=Path(d)/"reviewed.jsonl"
            original="\n".join(json.dumps(x) for x in candidates()).encode()
            path.write_bytes(original)
            real_hash=hashlib.sha256
            changed=False
            def hash_while_operator_replaces_file(data=b""):
                nonlocal changed
                if not changed:
                    changed=True
                    path.write_text("new incomplete operator edit")
                return real_hash(data)
            with patch("williams_ai.private_lora.hashlib.sha256",side_effect=hash_while_operator_replaces_file):
                result=checked_private_training_data(path)
            self.assertEqual(result["counts"]["train"],64)
            self.assertEqual(result["source_dataset_sha256"],real_hash(original).hexdigest())

    def test_non_qwen_base_is_rejected_before_loading_training_runtime(self):
        with TemporaryDirectory() as d:
            root=Path(d);data=root/"reviewed.jsonl"
            data.write_text("\n".join(json.dumps(x) for x in candidates()))
            base=root/"wrong-family";base.mkdir()
            (base/"config.json").write_text('{"model_type":"llama"}')
            with self.assertRaisesRegex(ValueError,"CAUSAL_QWEN_MODEL_REQUIRED"):
                train_private_lora(data=data,model_dir=base,output_dir=root/"adapter",
                                   source_only_research=True)
            self.assertFalse((root/"adapter").exists())

    def test_every_partition_must_cover_all_four_books(self):
        with TemporaryDirectory() as d:
            path=Path(d)/"reviewed.jsonl"
            for split in ("train","validation","test"):
                data=candidates()
                base=next(x for x in data if x["edition"]=="TC2" and x["split"]==split)
                for item in data:
                    if item["split"]==split:
                        for key in ("edition","rule_id","pdf_page","source_sha256"):
                            item[key]=base[key]
                        item["question"]=f"TC2 unique question {item['case_id']}: explain the cited closed-candle condition."
                path.write_text("\n".join(json.dumps(x) for x in data))
                with self.subTest(split=split),self.assertRaisesRegex(ValueError,"FOUR_BOOKS_IN_EACH_SPLIT"):
                    checked_private_training_data(path)

    def test_whitespace_only_paraphrase_cannot_cross_train_holdout(self):
        with TemporaryDirectory() as d:
            path=Path(d)/"reviewed.jsonl"
            data=candidates()
            data[64]["question"]="  \n"+data[0]["question"].replace(" ","  ")+"\n "
            path.write_text("\n".join(json.dumps(x) for x in data))
            with self.assertRaisesRegex(ValueError,"DUPLICATE_QUESTION"):
                checked_private_training_data(path)


class AuditDemoPublicMetadata(unittest.TestCase):
    def test_durable_history_resolves_symbol_from_full_futures_catalog(self):
        def fetch(venue,path,args):
            if path=="/fapi/v1/time":return {"serverTime":SERVER}
            self.assertEqual(path,"/fapi/v1/exchangeInfo")
            self.assertIsNone(args)
            return {"symbols":[{"symbol":"OTHERUSDT","status":"BREAK"},
                    {"symbol":"BTCUSDT","status":"TRADING","filters":[
                     {"filterType":"PRICE_FILTER","tickSize":"0.01"}]}]}
        stamp,tick=_market_snapshot("PUBLIC_USDM","BTCUSDT",fetch,now_ms=SERVER)
        self.assertEqual((stamp,tick),(SERVER,"0.01"))

    def test_manual_demo_filter_check_accepts_catalog_without_sending_orders(self):
        venue=FakeVenue()
        def transport(method,path,body,headers):
            response=venue(method,path,body,headers)
            if path=="/fapi/v1/exchangeInfo":
                response["symbols"].insert(0,{"symbol":"BTCUSDT","status":"BREAK"})
            return response
        demo=DemoFutures(DemoHttp("FAKE_TEST_KEY","FAKE_TEST_SECRET",transport=transport))
        proposal=TradeProposal(ticket="test",symbol="ETHUSDT",side="BUY",quantity="0.1",
                               stop_price="95",expires_at=100,risk_usdt="0.5",notional_usdt="10")
        demo._check_contract_filters(proposal)
        self.assertEqual(len(venue.calls),1)
        self.assertEqual(venue.calls[0][:3],("GET","/fapi/v1/exchangeInfo",{}))

    def test_demo_accepts_full_catalog_and_rejects_duplicate_requested_contract(self):
        fetch,calls=public_fixture()
        target=fetch("/fapi/v1/exchangeInfo",None)["symbols"][0]
        def full_catalog(path,args):
            if path=="/fapi/v1/exchangeInfo":
                self.assertIsNone(args)
                return {"symbols":[{"symbol":"BTCUSDT","status":"BREAK","filters":[]},target]}
            return fetch(path,args)
        state=check_public_demo(fetch=full_catalog,now_ms=CLOCK)
        self.assertEqual(state.symbol,"ETHUSDT")
        self.assertFalse(state.execution_allowed)
        def duplicate(path,args):
            if path=="/fapi/v1/exchangeInfo":return {"symbols":[target,target]}
            return fetch(path,args)
        with self.assertRaises(DemoSafetyError):check_public_demo(fetch=duplicate,now_ms=CLOCK)

    def test_large_catalog_has_a_bounded_larger_budget_than_clock_endpoint(self):
        payload=json.dumps({"symbols":[],"padding":"x"*200000}).encode()
        class Response:
            status=200
            def read(self,size):return payload[:size]
        class Connection:
            def __init__(self,*args,**kwargs):pass
            def request(self,*args,**kwargs):pass
            def getresponse(self):return Response()
            def close(self):pass
        with patch("williams_ai.demo_pilot_readiness.http.client.HTTPSConnection",Connection):
            result=_read_only_demo_json("/fapi/v1/exchangeInfo")
            self.assertEqual(len(result["padding"]),200000)
            with self.assertRaises(DemoSafetyError):_read_only_demo_json("/fapi/v1/time")
            self.assertEqual(len(DemoHttp._raw("GET","/fapi/v1/exchangeInfo","",{})["padding"]),200000)
            with self.assertRaises(DemoSafetyError):DemoHttp._raw("GET","/fapi/v1/time","",{})


class AuditModelFailureIsolation(unittest.TestCase):
    def test_one_runtime_error_does_not_abort_remaining_exam(self):
        calls=[]
        def answer(case):
            calls.append(case)
            if len(calls)==1:raise ValueError("PRIVATE_FILE_OR_SOURCE_MUST_NOT_APPEAR")
            return {"choice":CASES[1]["correct"],"reason":"Reviewed mock response"}
        result=grade_exam(answer,CASES[:2])
        self.assertEqual(result["correct_cases"],1)
        self.assertEqual(result["results"][0]["answer_error"],"ValueError")
        self.assertNotIn("PRIVATE_FILE_OR_SOURCE",json.dumps(result))
        self.assertEqual(len(calls),2)

    def test_malformed_choice_or_missing_reason_is_failed_not_an_exception(self):
        for response in ({"choice":[],"reason":"bad"},
                         {"choice":CASES[0]["correct"],"reason":None},
                         {"choice":CASES[0]["correct"],"reason":"ok","trade_action":"BUY"}):
            with self.subTest(response=response):
                result=grade_exam(lambda _:response,CASES[:1])
                self.assertEqual(result["correct_cases"],0)
                self.assertEqual(result["results"][0]["answer_error"],"MODEL_MALFORMED_ANSWER")

    def test_coaching_runtime_failure_is_not_a_book_misconception(self):
        cases=[x for x in learning_cases() if x["edition"]=="TC1"][:2]
        key={x["id"]:x["correct"] for x in cases}
        class Student:
            def __init__(self):self.calls=0
            def answer_exam_case(self,case,**kwargs):
                self.calls+=1
                if self.calls==1:raise RuntimeError("PRIVATE_MODEL_PATH")
                return {"choice":key[case["id"]],"reason":"Valid mock answer"}
        with TemporaryDirectory() as d:
            path=Path(d)/"study.db"
            result=coach_practice(Student(),path,model_id="gguf-audit",editions=("TC1",),
                                  max_cases=2,repeat_after_feedback=False)
            self.assertEqual(result["initial_pass"],1)
            self.assertEqual(result["private_misconception_topics"],0)
            self.assertEqual(result["failed_model_calls"],1)
            with notebook(path) as db:
                self.assertEqual(db.execute("SELECT count(*) FROM practice_attempts").fetchone()[0],2)

    def test_finished_attempt_and_remediation_survive_interrupted_retry(self):
        case=next(x for x in learning_cases() if x["edition"]=="TC1")
        wrong=next(x for x in "ABC" if x!=case["correct"])
        class Interrupted:
            def __init__(self):self.calls=0
            def answer_exam_case(self,case,**kwargs):
                self.calls+=1
                if self.calls==2:raise KeyboardInterrupt()
                return {"choice":wrong,"reason":"Mock conceptual mistake"}
        with TemporaryDirectory() as d:
            path=Path(d)/"study.db"
            with self.assertRaises(KeyboardInterrupt):
                coach_practice(Interrupted(),path,model_id="gguf-audit",editions=("TC1",),max_cases=1)
            with notebook(path) as db:
                self.assertEqual(db.execute("SELECT count(*) FROM practice_attempts").fetchone()[0],1)
                self.assertEqual(db.execute("SELECT count(*) FROM private_remediation").fetchone()[0],1)


class PromptSpy:
    def __init__(self,**kwargs):self.calls=[]
    def create_chat_completion(self,**kwargs):
        self.calls.append(kwargs)
        chart=json.loads(kwargs["messages"][1]["content"])["chart"]
        answer={"symbol":chart["symbol"],"as_of":chart["as_of"],"edition":"TC2",
                "setup":"NO_SETUP","side":"NONE","reason":"No setup",
                "entry_state":"NO_SETUP","exit_basis":"NONE","rule_ids":[],
                "trade_action":None,"execution_allowed":False}
        return {"choices":[{"message":{"content":json.dumps(answer)}}]}


class AuditPromptClock(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        weights=Path(self.tmp.name)/"test.gguf";weights.write_bytes(b"NOT_REAL_AI")
        self.model=LocalWilliamsModel(weights,loader=PromptSpy)
        market=frozen_markets()["BTCUSDT"];bars=market["bars"][:350]
        self.obs=analyze_chart("BTCUSDT","H1",bars,as_of=bars[-1]["close_time"],tick_size=market["tick_size"])

    def frame(self,at):
        return {"close":at,"trend":"UP","ao":17,"ac":9,"zone":"GREEN"}

    def test_timezone_formatted_future_native_bar_fails_before_model(self):
        at=datetime.fromisoformat(self.obs["as_of"])
        future=(at.replace(tzinfo=timezone(timedelta(hours=-9)))-timedelta(hours=1)).isoformat()
        self.obs["all_binance_timeframes_compact"]={"1m":self.frame(future)}
        with self.assertRaisesRegex(ValueError,"FUTURE"):
            self.model.propose_chart_reasoning(self.obs)
        self.assertEqual(self.model._model.calls,[])

    def test_same_closed_time_in_minsk_offset_is_not_discarded(self):
        alias=datetime.fromisoformat(self.obs["as_of"]).astimezone(timezone(timedelta(hours=3))).isoformat()
        self.obs["all_binance_timeframes_compact"]={"1m":self.frame(alias)}
        self.model.propose_chart_reasoning(self.obs)
        payload=json.loads(self.model._model.calls[-1]["messages"][1]["content"])
        self.assertIn("1m",payload["chart"]["native_binance_timeframe_evidence"])

    def test_unstamped_native_bar_is_rejected(self):
        self.obs["all_binance_timeframes_compact"]={"1m":self.frame("2026-01-01T00:00:00")}
        with self.assertRaisesRegex(ValueError,"TIME"):
            self.model.propose_chart_reasoning(self.obs)

    def test_future_higher_timeframe_is_rejected_before_model(self):
        future=(datetime.fromisoformat(self.obs["as_of"])+timedelta(hours=4)).isoformat()
        self.obs["higher_timeframe_context"]={"H4":{"status":"OBSERVED_CLOSED_BARS",
                                                       "closed_through":future,"ao":999}}
        with self.assertRaisesRegex(ValueError,"FUTURE"):
            self.model.propose_chart_reasoning(self.obs)
        self.assertEqual(self.model._model.calls,[])


if __name__=="__main__":unittest.main()
