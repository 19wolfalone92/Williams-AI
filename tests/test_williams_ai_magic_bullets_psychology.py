"""Source-grounded TC1 five magic bullets and model discipline.

These tests check recognition at ANY supported TF, correctness of book
meanings, no future wave confidence, no fake 'squat' from Binance traded
base volume, plus 15 live-market HISTORICAL prefixes (read-only).
"""
from __future__ import annotations
import unittest
from datetime import datetime,timezone,timedelta

from williams_ai.chart_reader import (
    analyze_chart, validate_closed_bars, TF_SECONDS,
)
from williams_ai.five_magic_bullets import five_magic_bullets
from williams_ai.psychology_discipline import psychology_discipline_audit
from williams_ai.causal_paper_lab import frozen_markets
from williams_ai.book_rules import registry


PIVOTS=[(0,110.),(10,100.),(30,120.),(50,110.),
        (70,145.),(90,130.),(110,160.),(125,141.)]


def maker(*,tf="H1",n=125,tick=True):
    seconds=TF_SECONDS[tf]
    when=datetime(2026,1,1,0,0,0,tzinfo=timezone.utc)
    result=[]
    for i in range(n):
        for k in range(len(PIVOTS)-1):
            t0,y0=PIVOTS[k]
            t1,y1=PIVOTS[k+1]
            if t0<=i<=t1:
                price=y0+(y1-y0)*(i-t0)/(t1-t0)
                break
        base={
            "close_time":(when+timedelta(seconds=(i+1)*seconds)
                          -timedelta(milliseconds=1)).isoformat(),
            "open":price,"high":price+.3,"low":price-.3,
            "close":price,"volume":25,"closed":True,
        }
        if tick:
            base["tick_volume"]=2000 if i==110 else 1000
        result.append(base)
    return result


def ao_scenario(n):
    x=[2.]*n
    for i in range(62,min(n,72)):x[i]=10.
    for i in range(91,min(n,111)):x[i]=5.
    if n>111:x[111]=1.
    if n>112:x[112]=1.
    return x


def five(rows,tf="H1"):
    bars,gaps=validate_closed_bars(rows,tf,rows[-1]["close_time"])
    assert gaps==0
    return five_magic_bullets(bars,ao_scenario(len(rows)),
                              data=rows,timeframe=tf)


class FiveMagicBulletsTests(unittest.TestCase):
    def test_actual_five_bullet_confluence_any_timeframe_without_chart_matching(self):
        for tf in ("M5","M15","H1","H4","D1"):
            with self.subTest(timeframe=tf):
                rows=maker(tf=tf)
                report=five(rows,tf)
                self.assertEqual(report["wave_candidate_count"],1)
                candidate=report["last_n_wave_candidates"][-1]
                self.assertEqual(candidate["observed_bullets"],5)
                self.assertEqual(candidate["unknown_bullets"],0)
                self.assertTrue(candidate["all_five_observed"])
                self.assertEqual(candidate["direction_of_prior_impulse"],"BULL")
                self.assertEqual(
                    candidate["classification"],
                    "FIVE_OBSERVED_ZERO_POINT_HYPOTHESIS_NOT_REVERSAL",
                )
                self.assertFalse(report["execution_allowed"])
                self.assertIsNone(report["trade_action"])
                self.assertEqual(
                    set(candidate["bullets"]),
                    {"DIVERGENCE_3_VS_5","FIBONACCI_TARGET_ZONE",
                     "TERMINAL_FRACTAL","SQUAT_MFI_VOLUME",
                     "PROFITUNITY_5345_MOMENTUM_CHANGE"},
                )
                for c in candidate["bullets"].values():
                    self.assertEqual(c["status"],"OBSERVED")

    def test_bearish_five_bullets_are_mirror_of_bullish_on_all_timeframes(self):
        for tf in ("M5","M15","H1","H4","D1"):
            rows=maker(tf=tf)
            for item in rows:
                high,low=item["high"],item["low"]
                item["high"],item["low"]=300-low,300-high
                item["open"]=300-item["open"]
                item["close"]=300-item["close"]
            candles,gaps=validate_closed_bars(rows,tf,rows[-1]["close_time"])
            self.assertEqual(gaps,0)
            oscillator=[-x for x in ao_scenario(len(rows))]
            info=five_magic_bullets(candles,oscillator,
                                    data=rows,timeframe=tf)
            self.assertEqual(info["wave_candidate_count"],1)
            c=info["last_n_wave_candidates"][-1]
            self.assertEqual(c["direction_of_prior_impulse"],"BEAR")
            self.assertEqual(c["observed_bullets"],5)
            self.assertTrue(c["all_five_observed"])
            self.assertFalse(info["execution_allowed"])

    def test_missing_actual_tick_count_blocks_full_five(self):
        rows=maker(tick=False)
        r=five(rows)
        self.assertEqual(r["wave_candidate_count"],1)
        x=r["last_n_wave_candidates"][-1]
        self.assertEqual(x["bullets"]["SQUAT_MFI_VOLUME"]["status"],"UNKNOWN")
        self.assertEqual(x["observed_bullets"],4)
        self.assertEqual(x["unknown_bullets"],1)
        self.assertFalse(x["all_five_observed"])
        self.assertTrue(r["tick_volume_missing_does_not_fabricate_squat"])

    def test_unconfirmed_fractal_cannot_be_used_to_call_wave_finished(self):
        rows=maker(n=112)
        r=five(rows)
        self.assertEqual(r["wave_candidate_count"],0)
        self.assertEqual(r["last_n_wave_candidates"],[])
        self.assertFalse(r["execution_allowed"])

    def test_not_every_squat_means_end_of_trend(self):
        rows=maker()
        rows[110]["tick_volume"]=700
        rows[111]["tick_volume"]=700
        # Otherwise the NEXT candle's rebound from 700 to 1000 ticks
        # creates an independent valid Squat near the price extreme.
        r=five(rows)
        candidate=r["last_n_wave_candidates"][-1]
        self.assertEqual(candidate["bullets"]["SQUAT_MFI_VOLUME"]["status"],
                         "NOT_OBSERVED")
        self.assertFalse(candidate["all_five_observed"])

    def test_dont_substitute_wrong_indicator_or_fake_divergence(self):
        rows=maker()
        from williams_ai.chart_reader import validate_closed_bars
        candles,_=validate_closed_bars(rows,"H1",rows[-1]["close_time"])
        ao=ao_scenario(len(rows))
        # Fifth price extreme has stronger histogram amplitude than W3.
        for k in range(95,111):ao[k]=14.
        r=five_magic_bullets(candles,ao,data=rows,timeframe="H1")
        x=r["last_n_wave_candidates"][-1]
        self.assertEqual(x["bullets"]["DIVERGENCE_3_VS_5"]["status"],
                         "NOT_OBSERVED")
        self.assertFalse(x["all_five_observed"])

    def test_all_15_actual_binance_markets_are_read_only_even_without_tick_volume(self):
        markets=frozen_markets()
        self.assertEqual(len(markets),15)
        for sym,info in markets.items():
            with self.subTest(symbol=sym):
                rows=info["bars"]
                chart=analyze_chart(sym,"H1",rows,
                                    as_of=rows[-1]["close_time"],
                                    tick_size=info["tick_size"])
                obs=chart["tc1_five_magic_bullets"]
                self.assertEqual(obs["timeframe"],"H1")
                self.assertTrue(obs["no_matching_picture_required"])
                self.assertIsNone(obs["trade_action"])
                self.assertFalse(obs["execution_allowed"])
                for item in obs["last_n_wave_candidates"]:
                    self.assertFalse(item["trade_action"])
                    self.assertFalse(item["all_five_observed"])
                    self.assertEqual(item["bullets"]["SQUAT_MFI_VOLUME"]["status"],
                                     "UNKNOWN")

    def test_15_real_binance_pairs_scanned_on_45_complete_timeframe_passports(self):
        from williams_ai.five_bullets_cli import scan_fifteen
        report=scan_fifteen()
        self.assertEqual(report["symbols"],15)
        self.assertEqual(report["timeframe_symbol_scans"],45)
        self.assertTrue(report["without_observed_M5_M15_data"])
        self.assertFalse(report["profitability_checked"])
        self.assertEqual(report["summary"]["complete_bullet_sets"],0)
        self.assertEqual({x["timeframe"] for x in report["by_pair_timeframe"]},
                         {"H1","H4","D1"})
        self.assertTrue(all(x["execution_allowed"] is False
                            and x["trade_action"] is None
                            for x in report["by_pair_timeframe"]))

    def test_llm_must_name_tc1_not_ntu_and_cannot_trade_on_five_bullets(self):
        from williams_ai.algorithm_guard import audit_reasoning
        rows=maker()
        test=five(rows)
        obs={
            "symbol":"BTCUSDT",
            "as_of":rows[-1]["close_time"],
            "execution_allowed":False,
            "candidates":[],
            "tc1_five_magic_bullets":test,
        }
        ask={
            "symbol":"BTCUSDT",
            "as_of":rows[-1]["close_time"],
            "edition":"TC1",
            "setup":"TC1_FIVE_MAGIC_BULLETS","side":"NONE",
            "rule_ids":["TC1.MAGIC.DIVERGENCE","TC1.MAGIC.SQUAT"],
            "trade_action":None,"execution_allowed":False,
            "entry_state":"CLOSED_CANDIDATE","exit_basis":"NONE",
        }
        ok=audit_reasoning(obs,ask)
        self.assertEqual(ok["verdict"],"PROVISIONAL_REASONING_ONLY")
        self.assertEqual(ok["hard_failures"],[])
        self.assertIn("ZERO_POINT_EVIDENCE_NOT_AUTOMATIC_EXECUTED_REVERSAL",
                      ok["uncertainties"])
        self.assertIsNone(ok["trade_action"])
        wrong=audit_reasoning(obs,{**ask,"edition":"NTD","rule_ids":["NTD.AO"]})
        self.assertIn("MAGIC_BULLETS_ARE_TC1_NOT_NTD_FIVE_DIMENSIONS",
                      wrong["hard_failures"])
        forbidden=audit_reasoning(obs,{**ask,"trade_action":"BUY",
                                       "execution_allowed":True})
        self.assertIn("EXECUTION_REQUEST_FORBIDDEN",forbidden["hard_failures"])

    def test_unconfirmed_binance_squat_cannot_be_asserted_as_all_five(self):
        from williams_ai.algorithm_guard import audit_reasoning
        from williams_ai.local_model import LocalWilliamsModel
        from williams_ai.causal_paper_lab import frozen_markets
        import tempfile
        from pathlib import Path
        rows=maker(tick=False)
        observation={
            "symbol":"BTCUSDT","as_of":rows[-1]["close_time"],
            "execution_allowed":False,"candidates":[],
            "tc1_five_magic_bullets":five(rows),
        }
        claim={
            "symbol":"BTCUSDT","as_of":rows[-1]["close_time"],
            "edition":"TC1","setup":"TC1_FIVE_MAGIC_BULLETS",
            "side":"NONE","rule_ids":["TC1.MAGIC.SQUAT"],
            "entry_state":"CLOSED_CANDIDATE","exit_basis":"NONE",
            "trade_action":None,"execution_allowed":False,
        }
        verdict=audit_reasoning(observation,claim)
        self.assertIn("FIVE_MAGIC_BULLETS_NOT_ALL_OBSERVED",
                      verdict["hard_failures"])
        market=frozen_markets()["BTCUSDT"]
        complete=market["bars"][:900]
        chart=analyze_chart("BTCUSDT","H1",complete,
                            as_of=complete[-1]["close_time"],
                            tick_size=market["tick_size"])
        class Dummy:
            def __init__(self,**_):self.calls=[]
            def create_chat_completion(self,**args):
                self.calls.append(args)
                content=__import__("json").loads(args["messages"][1]["content"])
                reply={"symbol":content["chart"]["symbol"],"as_of":content["chart"]["as_of"],
                       "edition":content["edition"],"setup":"NO_SETUP","side":"NONE",
                       "rule_ids":[],"trade_action":None,"execution_allowed":False}
                return {"choices":[{"message":{"content":__import__("json").dumps(reply)}}]}
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"mock.gguf"
            p.write_bytes(b"mock-but-only-unit-test")
            model=LocalWilliamsModel(p,loader=Dummy,
                                     use_decision_schema=True)
            model.propose_chart_reasoning(chart,edition="TC1")
            schema=model._model.calls[0]["response_format"]["schema"]
            self.assertNotIn("TC1_FIVE_MAGIC_BULLETS",
                             schema["properties"]["setup"]["enum"])
            self.assertIn("NO_SETUP",schema["properties"]["setup"]["enum"])

    def test_tenets_added_to_source_registry_are_not_mislabeled_five_dimensions(self):
        pack=registry()["rule_cards"]
        matching=[x for x in pack if x["id"].startswith("TC1.MAGIC.")]
        self.assertEqual(len(matching),5)
        self.assertTrue(all(x["status"]=="REVIEWED_PAGES" for x in matching))
        self.assertEqual(len([x for x in pack if x["id"].startswith("TC1.PSY.")]),4)


class PsychologyDisciplineTests(unittest.TestCase):
    def baseline(self):
        return {
            "edition":"TC2","rule_ids":["TC2.WM3"],
            "risk_limit_verified":True,"account_identity_verified":True,
            "data_fresh":True,"price_trigger_confirmed":True,
            "exchange_protective_stop_ack":True,
        }

    def test_human_emotions_not_projected_onto_ai(self):
        x=psychology_discipline_audit(self.baseline())
        self.assertEqual(x["pretrade_gate"],
                         "ELIGIBLE_FOR_SEPARATE_DETERMINISTIC_REVIEW")
        self.assertFalse(x["fear_or_greed_in_AI_diagnosed"])
        self.assertFalse(x["execution_allowed"])

    def test_forbid_revenge_averaging_fomo_stop_widening_and_hallucinated_certainty(self):
        bad={**self.baseline(),
             "increase_risk_because_recent_loss":True,
             "average_losing_position":True,
             "cancel_or_widen_stop_to_avoid_loss":True,
             "opposite_signal_only_requests_exit":True,
             "daily_loss_limit_reached":True,
             "market_direction_certain":True,
             "exchange_protective_stop_ack":False,
             "price_trigger_confirmed":False,
             "execution_policy_override_requested":True}
        x=psychology_discipline_audit(bad)
        reasons=x["blocking_reasons"]
        self.assertEqual(x["pretrade_gate"],"BLOCK")
        for rule in (
            "REVENGE_SIZE_INCREASE_AFTER_LOSS",
            "UNPLANNED_AVERAGING_DOWN",
            "FEAR_BASED_STOP_CANCELLATION_OR_WIDENING",
            "OPPOSITE_UNEXECUTED_SIGNAL_NOT_A_FILLED_EXIT",
            "LOSS_LIMIT_REACHED_NO_REVENGE_TRADING",
            "OVERCONFIDENT_CERTAINTY_ABOUT_CHAOTIC_MARKET",
            "PROTECTIVE_STOP_NOT_ACKNOWLEDGED",
            "FOMO_PRICE_CHASE_NO_CONFIRMED_TRIGGER",
            "MODEL_CANNOT_OVERRIDE_EXECUTION_DOOR",
        ):
            self.assertIn(rule,reasons)

    def test_owning_strategy_keeps_campaign_and_protection(self):
        plan={**self.baseline(),"new_campaign":True,
              "manage_existing_campaign":True}
        pos={"active_campaign":True,"unprotected_position":True,
             "owning_edition":"TC1"}
        result=psychology_discipline_audit(plan,position=pos)
        self.assertIn("NON_OWNER_ATTEMPTS_TO_MANAGE_EXISTING_CAMPAIGN",
                      result["blocking_reasons"])
        self.assertIn("UNPROTECTED_POSITION_REQUIRES_EXCHANGE_RECOVERY_FIRST",
                      result["blocking_reasons"])
        self.assertIn("ONE_CAMPAIGN_PER_SYMBOL_ALREADY_ACTIVE",
                      result["blocking_reasons"])
        self.assertIsNone(result["trade_action"])


if __name__=="__main__":
    unittest.main()
