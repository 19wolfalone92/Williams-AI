"""Offline contracts for a new 2025 real market source and temporal evaluation.

Fixtures are synthetic ONLY to verify transport, clocks and safety; live
GitHub Actions obtains independent public monthly Binance archives.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import unittest
import zipfile
from datetime import datetime,timedelta,timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from williams_ai.chart_reader import analyze_chart
from williams_ai.causal_timeframes import aggregate_closed_h1
from williams_ai.temporal_wave_revision import temporal_revision
from williams_ai.unseen_2025_cohort import (
    SYMBOLS,MONTHS,fetch_month,make_never_used_cohort,
)
from williams_ai.unseen_2025_wave_exam import (
    load_archives,audit_new_cohort,
)


class SyntheticBinanceArchiveServer:
    """Offline data generator, NOT an external historical test result."""
    def __init__(self,corrupt=False):
        self.data={}
        self.corrupt=corrupt

    def __call__(self,url):
        name=url.split("/")[-1].removesuffix(".CHECKSUM")
        key=name.removesuffix(".zip")
        if key not in self.data:
            symbol,_,year,month=key.split("-")
            start=datetime(int(year),int(month),1,tzinfo=timezone.utc)
            stop=(datetime(int(year),int(month)+1,1,tzinfo=timezone.utc)
                  if int(month)<12 else datetime(int(year)+1,1,1,tzinfo=timezone.utc))
            n=int((stop-start).total_seconds()/3600)
            payload=io.StringIO()
            writer=csv.writer(payload)
            for i in range(n):
                op=100+0.04*i+2*math.sin(i/13)
                cl=op+0.03
                ms=int((start+timedelta(hours=i)).timestamp()*1000)
                writer.writerow([ms,f"{op:.9f}",f"{op+.4:.9f}",f"{op-.4:.9f}",
                                f"{cl:.9f}",1000+i%11,ms+3_599_999,10,20,1,1,0])
            out=io.BytesIO()
            with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_DEFLATED) as z:
                z.writestr(key+".csv",payload.getvalue())
            blob=out.getvalue()
            self.data[key]=(blob,hashlib.sha256(blob).hexdigest())
        blob,digest=self.data[key]
        if url.endswith(".CHECKSUM"):
            if self.corrupt:digest="0"*64
            return f"{digest}  {name}\n".encode("ascii")
        return blob


class NoModelWeightFake:
    def propose_chart_reasoning(self,chart,*,edition):
        return {
            "symbol":chart["symbol"],"as_of":chart["as_of"],
            "edition":edition,"setup":"NO_SETUP","side":"NONE",
            "rule_ids":[],"entry_state":"NO_SETUP","exit_basis":"NONE",
            "reason":"No source-confirmed current trigger; hypothesis only.",
            "trade_action":None,"execution_allowed":False,
        }
    def propose_24h_research_forecast(self,chart):
        return {"symbol":chart["symbol"],"as_of":chart["as_of"],
                "direction":"ABSTAIN","trade_action":None,
                "execution_allowed":False}


class FreshUnseen2025Safety(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cohort=make_never_used_cohort(
            downloader=SyntheticBinanceArchiveServer())
        cls.market=cls.cohort["markets"]["BTCUSDT"]

    def test_monthly_zip_sha256_15_symbols_and_complete_2025_H1(self):
        data=self.cohort
        self.assertEqual(len(data["markets"]),15)
        self.assertEqual(len(data["sources"]),30)
        self.assertEqual(data["intentionally_predeclared_dates"],
                         ["2025-03","2025-04"])
        self.assertEqual(len(self.market["bars"]),1464)
        self.assertEqual(len({x["archive_sha256"] for x in data["sources"]}),30)
        self.assertFalse(data["execution_allowed"])
        self.assertTrue(data["source_volumes_are_NOT_TC1_tick_count"])
        self.assertNotIn("tick_volume",self.market["bars"][0])

    def test_checksum_tampering_never_silently_uses_old_2026_history(self):
        with self.assertRaisesRegex(ValueError,"SHA256_MISMATCH"):
            fetch_month("BTCUSDT","2025-03",
                        downloader=SyntheticBinanceArchiveServer(corrupt=True))
        with self.assertRaises(ValueError):
            fetch_month("BTCUSDT","2026-09",downloader=SyntheticBinanceArchiveServer())
        with self.assertRaises(ValueError):
            fetch_month("FAKEUSDT","2025-03",downloader=SyntheticBinanceArchiveServer())

    def test_source_manifest_immutable_and_never_predefined_trade(self):
        with TemporaryDirectory() as tmp:
            file=Path(tmp)/"market.json"
            file.write_text(json.dumps(self.cohort))
            valid=load_archives(file)
            self.assertEqual(valid["source_manifest_sha256"],
                             self.cohort["source_manifest_sha256"])
            tampered=json.loads(file.read_text())
            tampered["markets"]["BTCUSDT"]["bars"][0]["close"]=42
            file.write_text(json.dumps(tampered))
            with self.assertRaisesRegex(ValueError,"Unpinned"):
                load_archives(file)

    def test_time_state_cannot_retroactively_revise_wave_proof(self):
        rows=self.market["bars"]
        def view(t):
            prefix=rows[:t+1]
            return analyze_chart(
                "BTCUSDT","H1",prefix,
                as_of=prefix[-1]["close_time"],tick_size=None)
        a=view(847);b=view(1039)
        out=temporal_revision(a,b)
        self.assertLess(out["earlier_as_of"],out["later_as_of"])
        self.assertTrue(out["smaller_wave_completion_not_equal_to_larger_wave_completion"])
        self.assertTrue(out["ao_near_zero_not_proof_of_W4"])
        self.assertFalse(out["exact_wave_degree_verified"])
        self.assertTrue(out["tick_volume_missing_prevents_TC1_squat_confirmation"])
        self.assertFalse(out["execution_allowed"])
        self.assertIsNone(out["trade_action"])
        with self.assertRaisesRegex(ValueError,"genuine forward"):
            temporal_revision(b,a)
        with self.assertRaisesRegex(ValueError,"genuine forward"):
            temporal_revision(a,a)
        # A large alteration to AFTER t0 cannot retroactively change a.
        changed=json.loads(json.dumps(rows))
        for x in changed[848:]:
            x["open"]*=2;x["close"]*=2;x["high"]*=2;x["low"]*=2
        old=analyze_chart(
            "BTCUSDT","H1",changed[:848],
            as_of=rows[847]["close_time"],tick_size=None)
        self.assertEqual(a["principle_evidence"],old["principle_evidence"])

    def test_true_gguf_exam_interface_abstention_is_not_marked_as_mastery(self):
        # Contract-only fake. A real model is evaluated separately on
        # the GitHub CPU workflow with fingerprint-checked GGUF bytes.
        report=audit_new_cohort(
            self.cohort,model=NoModelWeightFake(),
            model_hash="a"*64,code_id="offline-unit-test-ONLY")
        self.assertEqual(report["book_situations"],8)
        self.assertEqual(report["source_consistent_total_including_abstentions"],8)
        self.assertEqual(report["source_consistent_substantive_setups"],0)
        self.assertEqual(report["directional_outcomes"],5)
        self.assertTrue(all(x["objective_temporal_revision_supplied"]==(x["index"]==1039)
                            for x in report["book_results"]))
        self.assertEqual(len(report["temporal_evidence_revisions"]),4)
        self.assertFalse(report["trading_qualified"])
        self.assertTrue(report["safe_abstention_is_not_wave_comprehension"])


if __name__=="__main__":
    unittest.main()
