"""Permanently sealed REAL-API-shaped forward trial: never re-evaluate model."""
import copy
import unittest
from datetime import datetime, timezone

from williams_ai.public_forward_trial import (
    prepare_forward_cohort, settle_forward_cohort,
)
from williams_ai.public_forward_capture import HOUR_MS

OPEN=int(datetime(2026,10,1,tzinfo=timezone.utc).timestamp()*1000)
NOW=OPEN+269*HOUR_MS+7*60_000
LATER=OPEN+294*HOUR_MS+7*60_000


def rows(count):
    out=[]
    for i in range(count):
        start=OPEN+i*HOUR_MS
        px=100+i*.012
        out.append([start,str(px),str(px+.5),str(px-.5),str(px),
                    "25",start+HOUR_MS-1,"0",1,"0","0","0"])
    return out


def public_get(venue,entries,stamp):
    host=("https://fapi.binance.com/fapi/v1" if venue=="FUTURES"
          else "https://data-api.binance.vision/api/v3")
    def get(url):
        if not url.startswith(host+"/"):
            raise AssertionError("Mixed up venue during settlement")
        if url.endswith("/time"):
            return {"serverTime":stamp}
        if "/exchangeInfo" in url:
            return {"symbols":[{
                "symbol":"BTCUSDT","status":"TRADING",
                "filters":[{"filterType":"PRICE_FILTER","tickSize":"0.01"}],
            }]}
        if "/klines?" in url:
            return entries
        raise AssertionError("unexpected public endpoint")
    return get


def predictor(obs):
    return {
        "symbol":obs["symbol"],"as_of":obs["as_of"],
        "direction":"UP","trade_action":None,"execution_allowed":False,
    }


def preparation(venue="FUTURES"):
    return prepare_forward_cohort(
        venue=venue,symbols=("BTCUSDT",),
        model_id="pinned-real-model-ident",code_id="git-exact-head-src-commit",
        predictor=predictor,
        fetch_json=public_get(venue,rows(270),NOW),clock_ms=NOW,
    )


class PublicForwardCohortAcceptanceTests(unittest.TestCase):
    def test_new_market_sealed_and_future_scored_without_model_recall(self):
        cohort=preparation()
        self.assertEqual(len(cohort["trials"]),1)
        self.assertEqual(cohort["venue"],"FUTURES")
        self.assertEqual(cohort["trials"][0]["source"]["venue"],"FUTURES")
        self.assertEqual(cohort["trials"][0]["sealed_forecast"]["direction"],"UP")
        self.assertEqual(cohort["trials"][0]["sealed_forecast"]["horizon_hours"],24)
        self.assertFalse(cohort["execution_allowed"])
        self.assertIsNone(cohort["trade_action"])
        result=settle_forward_cohort(
            cohort,fetch_json=public_get("FUTURES",rows(295),LATER),
            clock_ms=LATER,
        )
        self.assertEqual(result["commitment_manifest_sha256"],
                         cohort["manifest_sha256"])
        self.assertEqual(result["summary"]["total_cases"],1)
        self.assertEqual(result["summary"]["called_accuracy"],1.0)
        self.assertEqual(result["summary"]["live_trade_qualification"],"NOT_QUALIFIED")
        self.assertFalse(result["execution_allowed"])
        self.assertFalse(result["verified_real_account_PnL"])

    def test_future_has_not_yet_passed_no_early_settlement(self):
        cohort=preparation()
        with self.assertRaisesRegex(ValueError,"Future observation not mature"):
            settle_forward_cohort(
                cohort,fetch_json=public_get("FUTURES",rows(270),NOW),
                clock_ms=NOW,
            )

    def test_changed_public_historical_price_is_rejected(self):
        cohort=preparation()
        altered=rows(295)
        altered[100][4]=str(float(altered[100][4])+.001)
        with self.assertRaises(ValueError):
            settle_forward_cohort(
                cohort,fetch_json=public_get("FUTURES",altered,LATER),
                clock_ms=LATER,
            )

    def test_spot_and_futures_must_never_be_confused(self):
        cohort=preparation("SPOT")
        with self.assertRaises(AssertionError):
            settle_forward_cohort(
                cohort,fetch_json=public_get("FUTURES",rows(295),LATER),
                clock_ms=LATER,
            )

    def test_retroactive_market_venue_swap_or_decision_change_fails(self):
        cohort=preparation()
        modified=copy.deepcopy(cohort)
        modified["venue"]="SPOT"
        with self.assertRaisesRegex(ValueError,"Cohort seal mutated"):
            settle_forward_cohort(
                modified,fetch_json=public_get("FUTURES",rows(295),LATER),
                clock_ms=LATER,
            )
        modified=copy.deepcopy(cohort)
        modified["trials"][0]["sealed_forecast"]["direction"]="DOWN"
        with self.assertRaisesRegex(ValueError,"Cohort seal mutated"):
            settle_forward_cohort(
                modified,fetch_json=public_get("FUTURES",rows(295),LATER),
                clock_ms=LATER,
            )


if __name__=="__main__":
    unittest.main()
