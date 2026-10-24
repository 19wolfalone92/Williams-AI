"""A reviewer gets exact closed prices, never the engine answer or later bars."""
import unittest
from datetime import datetime,timedelta,timezone

from williams_ai.blind_review_packet import build_blind_review_packet,_digest


def capture(symbol="BTCUSDT",venue="SPOT"):
    start=datetime(2026,6,1,tzinfo=timezone.utc)
    bars=[{"close_time":(start+timedelta(hours=i)).isoformat(),"open":100.,
           "high":102.,"low":99.,"close":101.,"volume":12.,"closed":True} for i in range(500)]
    market={"symbol":symbol,"tick_size":.01,"bars":bars}
    return {"market":market,"source":{"venue":venue,"as_of":bars[-1]["close_time"],
        "observed_server_time":(start+timedelta(hours=500)).isoformat(),
        "market_sha256":_digest(market),"execution_allowed":False,"trade_action":None,
        "clock_checked":True},"model_prediction":"HIDDEN_ANSWER"}


class BlindHistoricalPacketTests(unittest.TestCase):
    def cohort(self):return [capture(s) for s in ("BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT")]

    def test_forty_prefixes_eight_days_include_prices_without_later_bars_or_predictions(self):
        inputs=self.cohort();p=build_blind_review_packet(inputs)
        self.assertEqual(p["cases"],40);self.assertEqual(len(p["utc_days"]),8)
        self.assertEqual(p["labels_completed"],0)
        self.assertNotIn("HIDDEN_ANSWER",str(p))
        self.assertEqual(len({t["case_id"] for t in p["tasks"]}),40)
        for task in p["tasks"]:
            bars=task["market_prefix"]["bars"]
            self.assertEqual(_digest(bars),task["visible_prefix_sha256"])
            self.assertEqual(bars[-1]["close_time"],task["cutoff"])
            self.assertTrue(all(b["close_time"]<=task["cutoff"] for b in bars))
            self.assertLess(task["cutoff"],inputs[0]["source"]["as_of"])
        digest=p.pop("packet_sha256");self.assertEqual(digest,_digest(p))

    def test_changed_price_cannot_keep_old_capture_seal(self):
        inputs=self.cohort();inputs[0]["market"]["bars"][5]["high"]=1000.
        with self.assertRaisesRegex(ValueError,"TAMPERED"):build_blind_review_packet(inputs)

    def test_unclosed_future_and_discontinuous_rows_fail_after_resealing(self):
        for mutate in (lambda c:c["market"]["bars"][-1].update(closed=False),
                       lambda c:c["source"].update(as_of=c["market"]["bars"][-2]["close_time"]),
                       lambda c:c["market"]["bars"].pop(70)):
            inputs=self.cohort();mutate(inputs[0])
            inputs[0]["source"]["market_sha256"]=_digest(inputs[0]["market"])
            with self.assertRaises(ValueError):build_blind_review_packet(inputs)

    def test_mixed_venues_duplicate_symbols_and_insufficient_days_fail(self):
        for inputs,kwargs in (([capture(),capture()],{}),
                              ([capture(),capture("ETHUSDT","FUTURES")],{"minimum_cases":1}),
                              (self.cohort(),{"utc_days":30}),
                              ([capture()],{})):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                build_blind_review_packet(inputs,**kwargs)

    def test_naive_capture_cutoff_cannot_authorize_a_timezone_guess(self):
        inputs=self.cohort();inputs[0]["source"]["as_of"]=inputs[0]["source"]["as_of"][:19]
        with self.assertRaises(ValueError):build_blind_review_packet(inputs)

    def test_parameters_and_invalid_tick_are_rejected(self):
        for kwargs in ({"utc_days":True},{"minimum_cases":False},{"minimum_cases":451}):
            with self.assertRaises(ValueError):build_blind_review_packet(self.cohort(),**kwargs)
        c=self.cohort();c[0]["market"]["tick_size"]=0
        c[0]["source"]["market_sha256"]=_digest(c[0]["market"])
        with self.assertRaisesRegex(ValueError,"TICK_SIZE"):build_blind_review_packet(c)


if __name__=="__main__":unittest.main()
