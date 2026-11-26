"""Optimizer accounting guard: fixtures, not evidence of GPU training."""
from types import SimpleNamespace
import unittest

from williams_ai.lora_training import require_actual_optimizer_step


class OptimizerAccountingTests(unittest.TestCase):
    def test_successful_update_and_resumed_counter(self):
        good=SimpleNamespace(optimizer_step_was_skipped=False)
        self.assertEqual(require_actual_optimizer_step(good, previous_step=0, reported_step=1),1)
        self.assertEqual(require_actual_optimizer_step(good, previous_step=10, reported_step=11),11)

    def test_skipped_or_missing_flag_never_counts(self):
        for value in (True,None,0,"false"):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,"SKIPPED_OR_UNOBSERVABLE"):
                require_actual_optimizer_step(SimpleNamespace(optimizer_step_was_skipped=value),
                                              previous_step=0,reported_step=1)
        with self.assertRaisesRegex(ValueError,"SKIPPED_OR_UNOBSERVABLE"):
            require_actual_optimizer_step(SimpleNamespace(),previous_step=0,reported_step=1)

    def test_counter_must_be_consecutive(self):
        good=SimpleNamespace(optimizer_step_was_skipped=False)
        for before,after in ((3,3),(3,5),(3,2),(-1,0),(False,1),(0,True)):
            with self.subTest(before=before,after=after),self.assertRaisesRegex(ValueError,"DISCONTINUITY"):
                require_actual_optimizer_step(good,previous_step=before,reported_step=after)
