import unittest
from williams_ai.v71_guided_json_diagnostics import ENFORCER_COMMIT
from scripts.ai_v71_existing_kernel_status import _safe_kernel_failure_diagnostic

class KaggleLogSafetyTests(unittest.TestCase):
 def test_failure_classification_without_log_text_leak(self):
  secret="SECRET_private_random_token_1234"
  log="Traceback AttributeError: _decode_function in lmformatenforcer\n"+secret
  out=_safe_kernel_failure_diagnostic(log)
  self.assertIn("AttributeError",out["exception_types"])
  self.assertIn("UPSTREAM_ENFORCER",out["evidence_flags"])
  self.assertNotIn(secret,str(out))
  self.assertNotIn("log_text",out)
 def test_known_public_missing_dependencies_only(self):
  log=("ModuleNotFoundError: No module named 'interegular'\\n"
       "ModuleNotFoundError: No module named 'veryprivate_secret_43'\\n")
  out=_safe_kernel_failure_diagnostic(log)
  self.assertEqual(out["missing_module_names_allowlisted"],["interegular"])
  self.assertTrue(out["missing_module_unclassified"])
  self.assertNotIn("veryprivate_secret_43",str(out))
 def test_empty_log_is_safe(self):
  out=_safe_kernel_failure_diagnostic("")
  self.assertEqual(out["log_length"],0)
  self.assertEqual(out["exception_types"],[])
 def test_upstream_identity_is_unchanged(self):
  self.assertEqual(ENFORCER_COMMIT,"d4d9b7cc20937bcd3a643be7a8dafb7b1ca20dfb")

if __name__=="__main__":unittest.main()
