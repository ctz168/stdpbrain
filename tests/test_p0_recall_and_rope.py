import unittest
import torch

from core.qwen_narrow_band_patch import SparseAttentionCompressor
from hippocampus.ca3_memory import CA3EpisodicMemory


class TestRopeAndRecall(unittest.TestCase):
    def test_window_keys_not_double_rotated(self):
        torch.manual_seed(42)
        batch, heads, seq_len, dim = 1, 2, 8, 4
        window_size = 4

        key_states = torch.randn(batch, heads, seq_len, dim)
        value_states = torch.randn(batch, heads, seq_len, dim)
        window_keys_before = key_states[:, :, -window_size:, :].clone()

        cos = torch.randn(seq_len + 10, dim)
        sin = torch.randn(seq_len + 10, dim)
        anchors = [{"key_features": torch.ones(dim), "value_features": torch.ones(dim)}]

        compressed_keys, _ = SparseAttentionCompressor.compress_kv(
            key_states=key_states,
            value_states=value_states,
            anchors=anchors,
            window_size=window_size,
            num_heads=heads,
            head_dim=dim,
            cos=cos,
            sin=sin,
        )

        window_after = compressed_keys[:, :, -window_size:, :]
        self.assertTrue(torch.allclose(window_after, window_keys_before, atol=1e-6))

    def test_ca3_recall_trace_uses_true_indices(self):
        mem = CA3EpisodicMemory(max_capacity=10, feature_dim=4)

        mem.store(
            memory_id="m0",
            timestamp=10,
            semantic_pointer="wrong semantic",
            dg_features=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            content="A",
        )
        mem.store(
            memory_id="m1",
            timestamp=20,
            semantic_pointer="押金 500",
            dg_features=torch.tensor([0.0, 1.0, 0.0, 0.0]),
            content="B",
        )

        query = torch.tensor([0.0, 1.0, 0.0, 0.0])
        mem.recall(query_features=query, query_semantic="押金 500", topk=1)
        trace = mem.get_last_recall_trace()

        self.assertGreaterEqual(len(trace), 1)
        self.assertEqual(trace[0]["memory_id"], "m1")
        self.assertGreater(trace[0]["semantic_score"], 0.0)
        self.assertTrue(trace[0]["selected"])


if __name__ == "__main__":
    unittest.main()
