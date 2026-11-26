import unittest
from scripts.ai_kaggle_kernel_diagnostic import redact

class KaggleDiagnosticRedactionTests(unittest.TestCase):
 def test_known_secrets_are_removed(self):
  self.assertNotIn("secret-value",redact("KAGGLE_API_TOKEN=secret-value",["secret-value"]))
 def test_key_value_patterns_are_redacted(self):
  out=redact("KAGGLE_KEY: abc123 KAGGLE_USERNAME=wolf",[])
  self.assertNotIn("abc123",out);self.assertNotIn("wolf",out)
 def test_multiple_secrets_are_removed(self):
  out=redact("first second",["first","second"])
  self.assertEqual(out,"[REDACTED] [REDACTED]")

if __name__=="__main__":unittest.main()
