"""Private local 2-stage CLI never accepts credentials or overwrites seals."""
import json
import tempfile
import unittest
from pathlib import Path

from williams_ai.prospective_cli import _market_file, _write_once_private


class PrivateProspectiveCLITests(unittest.TestCase):
    def test_market_snapshot_rejects_any_unrecognized_key(self):
        with tempfile.TemporaryDirectory() as d:
            dest=Path(d)/"h1.json"
            dest.write_text(json.dumps({
                "symbol":"BTCUSDT","tick_size":.01,
                "bars":[],"api_key":"NOT_A_REAL_SECRET_TEST_VALUE",
            }),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"no credentials"):
                _market_file(dest)

    def test_sealed_json_is_exclusive_create_not_mutable_in_place(self):
        with tempfile.TemporaryDirectory() as d:
            dest=Path(d)/"sealed.json"
            _write_once_private(dest,{"direction":"ABSTAIN","trade_action":None})
            self.assertTrue(dest.is_file())
            original=dest.read_bytes()
            with self.assertRaises(FileExistsError):
                _write_once_private(dest,{"direction":"UP"})
            self.assertEqual(original,dest.read_bytes())


if __name__=="__main__":
    unittest.main()
