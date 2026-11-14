"""Verify 100-case CPU shard layout without downloading weights or private data."""
import unittest
from scripts.ai_v6_historical100_shard_cpu import assigned_shard


class ShardRegression(unittest.TestCase):
    def test_ten_nonoverlapping_equal_shards(self):
        cases = [({"id": f"HIST100-{i // 4 + 1:02d}-{i % 4 + 1}"}, None, None, "sha")
                 for i in range(100)]
        groups = [assigned_shard(cases, i, 10) for i in range(10)]
        self.assertTrue(all(len(g) == 10 for g in groups))
        ids = [case[0]["id"] for group in groups for case in group]
        self.assertEqual(len(set(ids)), 100)

    def test_bad_index_rejected(self):
        cases = [({"id": f"{i}"}, None, None, "sha") for i in range(100)]
        with self.assertRaisesRegex(ValueError, "BAD_SHARD_SPEC"):
            assigned_shard(cases, 10, 10)
        with self.assertRaisesRegex(ValueError, "BAD_SHARD_SPEC"):
            assigned_shard(cases, 0, 5)


if __name__ == "__main__":
    unittest.main()
