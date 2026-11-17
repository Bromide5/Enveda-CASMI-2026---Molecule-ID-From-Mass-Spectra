"""生成分支因果性、冻结、合法性及检索门控的关键检查。"""

import unittest
from types import SimpleNamespace
import pandas as pd
import torch

from algorithms.transformer.model import SpectrumEncoder, SpectrumEncoderConfig
from algorithms.transformer.generation import SpectrumGenerator, SmilesTokenizer, canonical_smiles
from algorithms.transformer.hybrid import accepts, retrieve
from evaluation.calibrate_hybrid import calibrate


class HybridTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        torch.set_num_threads(2)
        self.tokenizer = SmilesTokenizer("CON()=12")
        encoder = SpectrumEncoder(SpectrumEncoderConfig(max_mz=8, d_model=8, nhead=2,
                                  num_layers=1, dim_feedforward=16, dropout=0, embedding_dim=8))
        self.model = SpectrumGenerator(encoder, self.tokenizer, num_layers=1, max_length=12)
        self.x = torch.ones(2, 8)

    def test_tokenizer_and_validity(self):
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode("CC(=O)N")), "CC(=O)N")
        self.assertEqual(canonical_smiles("OCC"), "CCO")
        self.assertIsNone(canonical_smiles("C("))
        self.assertIsNone(canonical_smiles(""))

    def test_causal_mask_and_spectral_conditioning(self):
        self.model.eval()
        ids = torch.tensor([[1, 3, 4, 5], [1, 3, 4, 5]])
        before = self.model(self.x, ids)
        changed = ids.clone()
        changed[:, -1] = 6
        torch.testing.assert_close(before[:, :-1], self.model(self.x, changed)[:, :-1])
        other = self.x.clone()
        other[:, :4] = 0
        self.assertFalse(torch.allclose(before, self.model(other, ids)))

    def test_decoder_update_preserves_encoder(self):
        self.model.train()
        self.assertFalse(self.model.encoder.training)
        original = {k: v.clone() for k, v in self.model.encoder.state_dict().items()}
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=0.01)
        ids = torch.tensor([self.tokenizer.encode("CCO"), self.tokenizer.encode("CCN")])
        loss = torch.nn.functional.cross_entropy(self.model(self.x, ids[:, :-1]).flatten(0,1), ids[:,1:].flatten())
        loss.backward()
        self.assertGreater(self.model.output.weight.grad.norm().item(), 0)
        optimizer.step()
        for key, value in self.model.encoder.state_dict().items():
            torch.testing.assert_close(original[key], value, rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in self.model.encoder.parameters()))

    def test_generation_eos_and_invalid_rejection(self):
        c_token = self.tokenizer.lookup["C"]
        def scripted(tokens, memory, padding):
            logits = torch.full((len(tokens), tokens.shape[1], len(self.tokenizer)), -100.)
            logits[:, -1, c_token if tokens.shape[1] == 1 else self.tokenizer.eos] = 100
            return logits
        self.model.decode_logits = scripted
        result = self.model.generate(self.x[:1], beam_size=1)
        self.assertEqual(result[0][0], "C")
        def invalid(tokens, memory, padding):
            logits = torch.full((len(tokens), tokens.shape[1], len(self.tokenizer)), -100.)
            logits[:, -1, self.tokenizer.eos] = 100
            return logits
        self.model.decode_logits = invalid
        self.assertEqual(self.model.generate(self.x[:1], beam_size=1), [])

    def test_gate_branches(self):
        row = SimpleNamespace(candidate_structures=2, top_score=.95, margin=.2)
        gate = dict(min_score=.9, min_margin=.1)
        self.assertTrue(accepts(row, gate))
        self.assertFalse(accepts(row, dict(reject_all=True)))
        row.candidate_structures = 1
        self.assertFalse(accepts(row, gate))
        row.candidate_structures, row.margin = 2, .01
        self.assertFalse(accepts(row, gate))

    def test_retrieval_distinct_structures_strict_filter_and_no_label_use(self):
        common = dict(ms2_mzs=[1., 3.], ms2_normalized_intensities=[1., .5],
                      precursor_mz=100., adduct="[M+H]+", ionization_mode="positive")
        reference = pd.DataFrame([dict(common, inchikey14="A", normalized_smiles="CC"),
                                  dict(common, inchikey14="A", normalized_smiles="CC"),
                                  dict(common, inchikey14="B", normalized_smiles="CO")])
        queries = pd.DataFrame([dict(common, molecule_id="q", inchikey14="NOT_READ"),
                                dict(common, molecule_id="q", inchikey14="WRONG"),
                                dict(common, molecule_id="empty", adduct="[M+Na]+")])
        ranked, diagnostics, _ = retrieve(self.model.encoder, reference, queries)
        self.assertEqual(ranked["q"], ["CC", "CO"])
        self.assertEqual(ranked["empty"], [])
        self.assertEqual(diagnostics.iloc[0].candidate_structures, 2)
        self.assertEqual(diagnostics.iloc[0].margin, 0)

    def test_calibration_rejects_unknown_and_fails_closed(self):
        routing = pd.DataFrame(dict(molecule_id=["a","b","c","d"], top_score=[.99,.98,.8,.7],
                                    margin=[.3,.2,.1,.05], candidate_structures=[2]*4,
                                    top_key=["A","B","A","B"]))
        labels = pd.DataFrame(dict(molecule_id=["a","b","c","d"], inchikey14=["A","B","C","D"],
                                   in_library=[True,True,False,False]))
        gate = calibrate(routing, labels, min_accepted=2)
        self.assertFalse(gate["reject_all"])
        self.assertEqual(gate["unknown_accept_rate"], 0)
        self.assertTrue(calibrate(routing, labels, min_accepted=3)["reject_all"])


if __name__ == "__main__":
    unittest.main()
