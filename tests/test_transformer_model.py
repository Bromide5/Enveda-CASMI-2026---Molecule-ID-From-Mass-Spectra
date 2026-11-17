"""编码器形状、双向信息流、梯度和权重恢复检查。"""

import unittest

import torch
from torch.nn import functional as F

from algorithms.transformer.model import SpectrumEncoder, SpectrumEncoderConfig


class EncoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)
        self.config = SpectrumEncoderConfig(max_mz=16, d_model=16, nhead=4,
                                            dim_feedforward=32, embedding_dim=8)
        self.model = SpectrumEncoder(self.config).eval()
        self.x = torch.zeros(3, 16)
        self.x[:, [2, 9, 14]] = torch.rand(3, 3) + 0.1

    def test_shapes_norms_and_batch_independence(self):
        with torch.no_grad():
            self.assertEqual(self.model.input_embedding(self.x).shape, (3, 16, 16))
            result = self.model(self.x)
            self.assertEqual(result.shape, (3, 8))
            torch.testing.assert_close(result.norm(dim=1), torch.ones(3))
            torch.testing.assert_close(result, self.model(self.x))
            torch.testing.assert_close(result, torch.cat([self.model(row[None]) for row in self.x]))

    def test_mass_encoding_and_bidirectional_attention(self):
        embedded = self.model.input_embedding(torch.ones(1, 16))
        self.assertFalse(torch.allclose(embedded[:, 2], embedded[:, 14]))
        changed = self.x.clone()
        changed[:, 14] += 2
        # 后方的峰应影响前方 token；不存在自回归的因果屏蔽。
        before = self.model.encode_tokens(self.x)[:, 2]
        after = self.model.encode_tokens(changed)[:, 2]
        self.assertFalse(torch.allclose(before, after))

    def test_empty_bins_are_excluded(self):
        before = self.model(self.x)
        with torch.no_grad():
            self.model.input_embedding.mass_encoding[:, 0] += 1000
        torch.testing.assert_close(before, self.model(self.x))

    def test_gradients_and_independent_layers(self):
        result = self.model(self.x)
        F.mse_loss(result, F.normalize(torch.randn_like(result), dim=-1)).backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for parameter in [self.model.input_embedding.intensity_projection.weight,
                          self.model.output_projection.weight,
                          *[layer.self_attn.in_proj_weight for layer in self.model.layers]]:
            self.assertGreater(parameter.grad.norm().item(), 0)
        self.assertFalse(torch.equal(self.model.layers[0].self_attn.in_proj_weight,
                                     self.model.layers[1].self_attn.in_proj_weight))

    def test_weight_roundtrip(self):
        restored = SpectrumEncoder(self.config).eval()
        restored.load_state_dict(self.model.state_dict())
        torch.testing.assert_close(self.model(self.x), restored(self.x))

    def test_invalid_inputs_and_config(self):
        for x in [torch.zeros(1, 16), torch.ones(1, 15), torch.empty(0, 16),
                  torch.full((1, 16), float("nan")), -torch.ones(1, 16),
                  torch.ones(1, 16, dtype=torch.int64)]:
            with self.assertRaises(ValueError):
                self.model(x)
        with self.assertRaises(ValueError):
            SpectrumEncoderConfig(d_model=15, nhead=4)
        with self.assertRaises(ValueError):
            SpectrumEncoderConfig(dropout=1)
        odd = SpectrumEncoder(SpectrumEncoderConfig(max_mz=16, d_model=15, nhead=3))
        self.assertEqual(odd(self.x).shape, (3, 128))


if __name__ == "__main__":
    unittest.main()
