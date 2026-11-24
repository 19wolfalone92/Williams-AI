from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import patch
from williams_ai.independent_exam_v6 import (
    audit_episode_windows, validate_final_cases, digest,
)


def window(i, symbol="BTCUSDT", offset=0, hours=1024):
    start=datetime(2025,1,1,tzinfo=timezone.utc)+timedelta(hours=offset)
    return {"id":str(i),"symbol":symbol,
            "start_utc":start.isoformat().replace("+00:00","Z"),
            "end_utc":(start+timedelta(hours=hours)).isoformat().replace("+00:00","Z")}


class FinalEpisodeTests(unittest.TestCase):
    def test_reproduce_frozen_32_daily_pilot_overlap(self):
        windows=[window(f"{s}-{d}",s,d*24) for s in
                 ("BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT") for d in range(8)]
        audit=audit_episode_windows(windows)
        self.assertEqual(len(audit["same_symbol_conflicts"]),112)
        self.assertFalse(audit["same_symbol_independence_gate"])

    def test_adjacent_and_known_prior_exposure_are_rejected(self):
        a=window(1)
        b=window(2,offset=1024+167)
        c=window(3,offset=1024+168)
        self.assertFalse(audit_episode_windows([a,b])["same_symbol_independence_gate"])
        self.assertTrue(audit_episode_windows([a,c])["same_symbol_independence_gate"])
        self.assertEqual(audit_episode_windows([b],[a])["known_exposure_conflicts"],[("2","1")])
        self.assertFalse(audit_episode_windows([a,c])["cross_symbol_statistical_independence_verified"])

    def test_final_gate_rejects_32_case_pilot_and_changed_inventory(self):
        prior=[window("old")]
        with patch("williams_ai.independent_exam_v6.validate_cases",return_value={"case_count":32}):
            with self.assertRaisesRegex(ValueError,"FINAL_MIN_100_CHARTS"):
                validate_final_cases({}, ".",prior_exposure=prior,exposure_sha256=digest(prior))
        with self.assertRaisesRegex(ValueError,"EXPOSURE_INVENTORY_MISMATCH"):
            validate_final_cases({}, ".",prior_exposure=prior,exposure_sha256="0"*64)

    def test_duplicate_episode_and_invalid_interval(self):
        with self.assertRaisesRegex(ValueError,"DUPLICATE_EPISODE_ID"):
            audit_episode_windows([window(1),window(1)])
        with self.assertRaisesRegex(ValueError,"BAD_EPISODE_INTERVAL"):
            audit_episode_windows([window(2,hours=0)])


if __name__=="__main__":
    unittest.main()
