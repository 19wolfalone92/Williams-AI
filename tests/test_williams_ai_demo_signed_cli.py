"""Integration regression: CLI itself enforces signed preflight, not optional."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from williams_ai import demo_pilot_cli
from williams_ai.demo_futures import DemoSafetyError
from test_williams_ai_demo_pilot import small_preview
from test_williams_ai_demo_account_recovery import SignedDemoReadOnlyPreflightTests


class SupervisedPilotSignedGateTests(unittest.TestCase):
    def test_pending_eth_order_blocks_before_engine_or_order_submission(self):
        with tempfile.TemporaryDirectory() as d:
            journal=Path(d)/"one.sqlite3"
            preview,_=small_preview()
            fake=SignedDemoReadOnlyPreflightTests.Account(
                open_orders=[{"symbol":"ETHUSDT","orderId":1001}])
            argv=[
                "demo_pilot_cli","supervised-send",
                "--side","BUY","--quantity","0.1","--stop","95",
                "--journal",str(journal)
            ]
            with (
                patch.object(demo_pilot_cli.sys,"argv",argv),
                patch.object(demo_pilot_cli.sys,"stdin",
                             MagicMock(isatty=lambda:True)),
                patch.object(demo_pilot_cli,"check_public_demo",return_value=None),
                patch.object(demo_pilot_cli,"validate_small_pilot",
                             return_value=preview),
                patch.object(demo_pilot_cli,"getpass",side_effect=[
                    "DEMO_ONLY_KEY","DEMO_ONLY_SECRET"]),
                patch("builtins.input",return_value="DEMO_ACCOUNT_ONLY"),
                patch.object(demo_pilot_cli,"DemoHttp",return_value=fake),
                patch.object(demo_pilot_cli,"DemoFutures") as engine
            ):
                with self.assertRaisesRegex(DemoSafetyError,
                                            "EXISTING_ORDERS_ON_PAIR"):
                    demo_pilot_cli.main()
                engine.assert_not_called()
            self.assertTrue(fake.calls)
            self.assertTrue(all(method=="GET" and signed
                                for method,path,args,signed in fake.calls))


if __name__=="__main__":
    unittest.main()
