"""验证多正样本目标、真实参数更新和检查点恢复。"""

import io
import math
import unittest

import torch

from algorithms.transformer.losses import supervised_contrastive_loss
from algorithms.transformer.model import SpectrumEncoder, SpectrumEncoderConfig
from algorithms.transformer.train import train_step


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        torch.set_num_threads(2)

    def test_uniform_scores_exclude_self(self):
        vectors = torch.ones(6, 4, requires_grad=True)
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        loss = supervised_contrastive_loss(vectors, labels)
        self.assertAlmostEqual(loss.item(), math.log(5), places=5)
        loss.backward()
        self.assertTrue(torch.isfinite(vectors.grad).all())

    def test_matches_explicit_multiple_positive_formula(self):
        vectors = torch.nn.functional.normalize(torch.randn(6, 5), dim=1)
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        scores = vectors @ vectors.T / 0.3
        expected = []
        for i in range(6):
            denominator = sum(torch.exp(scores[i, j]) for j in range(6) if j != i)
            terms = [-torch.log(torch.exp(scores[i, j]) / denominator)
                     for j in range(6) if j != i and labels[j] == labels[i]]
            expected.append(torch.stack(terms).mean())
        torch.testing.assert_close(supervised_contrastive_loss(vectors, labels, 0.3),
                                   torch.stack(expected).mean())
        permutation = torch.tensor([5, 0, 3, 2, 1, 4])
        torch.testing.assert_close(supervised_contrastive_loss(vectors, labels),
                                   supervised_contrastive_loss(vectors[permutation], labels[permutation]))

    def test_correct_pairs_have_lower_loss(self):
        vectors = torch.tensor([[1., 0], [1., 0], [0., 1], [0., 1]])
        good = supervised_contrastive_loss(vectors, torch.tensor([0, 0, 1, 1]))
        bad = supervised_contrastive_loss(vectors, torch.tensor([0, 1, 0, 1]))
        self.assertLess(good.item(), bad.item())

    def test_invalid_batches(self):
        for labels in ([0, 0, 0, 0], [0, 0, 1, 2]):
            with self.assertRaises(ValueError):
                supervised_contrastive_loss(torch.ones(4, 3), torch.tensor(labels))
        with self.assertRaises(ValueError):
            supervised_contrastive_loss(torch.ones(4, 3), torch.tensor([0, 0, 1, 1]), 0)

    def test_update_overfit_and_checkpoint_restore(self):
        config = SpectrumEncoderConfig(max_mz=16, d_model=16, nhead=4,
                                       num_layers=1, dim_feedforward=32, embedding_dim=8, dropout=0)
        model = SpectrumEncoder(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
        x = torch.zeros(4, 16)
        x[0, [1, 3]] = torch.tensor([1., 0.5])
        x[1, [1, 3]] = torch.tensor([0.9, 0.6])
        x[2, [10, 13]] = torch.tensor([1., 0.5])
        x[3, [10, 13]] = torch.tensor([0.9, 0.6])
        labels = torch.tensor([0, 0, 1, 1])
        initial = model.input_embedding.intensity_projection.weight.detach().clone()
        before = supervised_contrastive_loss(model(x), labels).item()
        for _ in range(12):
            train_step(model, optimizer, x, labels)
        after = supervised_contrastive_loss(model(x), labels).item()
        self.assertLess(after, before)
        self.assertFalse(torch.equal(initial, model.input_embedding.intensity_projection.weight))
        stream = io.BytesIO()
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict()), stream)
        stream.seek(0)
        saved = torch.load(stream, weights_only=True)
        restored = SpectrumEncoder(config)
        restored.load_state_dict(saved["model"])
        restored_optimizer = torch.optim.AdamW(restored.parameters())
        restored_optimizer.load_state_dict(saved["optimizer"])
        train_step(model, optimizer, x, labels)
        train_step(restored, restored_optimizer, x, labels)
        torch.testing.assert_close(model(x), restored(x))


if __name__ == "__main__":
    unittest.main()
