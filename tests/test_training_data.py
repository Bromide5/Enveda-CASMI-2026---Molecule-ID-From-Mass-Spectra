"""验证结构隔离、重复谱处理及对比学习批次，使用合成数据避免依赖原始数据文件。"""

import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from algorithms.transformer.dataset import PKBatchSampler, SpectrumDataset
from evaluation.build_training_split import build_training_split


def example_frame(structures=12, spectra=3):
    rows = []
    for key in range(structures):
        for number in range(spectra):
            rows.append(dict(
                inchikey14=f"KEY{key:011d}", normalized_smiles="C" * (key + 1),
                ms2_mzs=[10.25 + number, 40.5 + key],
                ms2_normalized_intensities=[0.25, 1.0],
                precursor_mz=200.0 + key, adduct="[M+H]+", ionization_mode="positive",
                source_row=len(rows),
            ))
    return pd.DataFrame(rows)


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.scratch = Path(__file__).resolve().parents[1] / "tmp"
        self.scratch.mkdir(exist_ok=True)
        self.root = self.scratch / f"training-data-test-{uuid.uuid4().hex}"
        self.root.mkdir()

    def tearDown(self):
        self.assertTrue(self.root.resolve().is_relative_to(self.scratch.resolve()))
        shutil.rmtree(self.root)

    def test_disjoint_reproducible_multispectrum_queries(self):
        frame = example_frame()
        duplicate = frame.iloc[[0]].copy()
        duplicate.at[0, "ms2_mzs"] = list(reversed(duplicate.iloc[0].ms2_mzs))
        duplicate.at[0, "ms2_normalized_intensities"] = list(
            reversed(duplicate.iloc[0].ms2_normalized_intensities))
        frame = pd.concat([frame, duplicate], ignore_index=True)
        report = build_training_split(frame, self.root / "a", queries_per_structure=2)
        build_training_split(frame, self.root / "b", queries_per_structure=2)
        self.assertEqual(report["dropped_exact_duplicates"], 1)
        manifest = pd.read_csv(self.root / "a/manifest.csv")
        pd.testing.assert_frame_equal(manifest, pd.read_csv(self.root / "b/manifest.csv"))
        self.assertTrue(manifest.signature.is_unique)
        self.assertEqual(manifest.groupby("inchikey14").split.nunique().max(), 1)
        for split in ("validation", "holdout"):
            queries = pd.read_parquet(self.root / "a" / split / "queries.parquet")
            reference = pd.read_parquet(self.root / "a" / split / "reference.parquet")
            labels = pd.read_csv(self.root / "a" / split / "labels.csv")
            self.assertNotIn("inchikey14", queries)
            self.assertNotIn("normalized_smiles", queries)
            self.assertTrue(labels.molecule_id.is_unique)
            self.assertTrue(queries.groupby("molecule_id").size().eq(2).all())
            self.assertFalse(set(queries.input_row) & set(reference.input_row))
            self.assertTrue(set(labels.inchikey14).issubset(set(reference.inchikey14)))

    def test_conflicting_labels_and_invalid_spectra_removed(self):
        frame = example_frame()
        conflict = frame.iloc[[0]].copy()
        conflict.at[0, "inchikey14"] = "DIFFERENTKEY000"
        invalid = frame.iloc[[1]].copy()
        invalid.at[1, "ms2_normalized_intensities"] = [0.25]
        empty = frame.iloc[[2]].copy()
        empty.at[2, "ms2_normalized_intensities"] = [0.0, 0.0]
        frame = pd.concat([frame, conflict, invalid, empty], ignore_index=True)
        report = build_training_split(frame, self.root / "split")
        self.assertEqual(report["dropped_conflicting_spectra"], 2)
        self.assertEqual(report["dropped_invalid_spectra"], 2)
        manifest = pd.read_csv(self.root / "split/manifest.csv")
        self.assertNotIn(0, set(manifest.input_row))
        self.assertTrue(manifest.signature.is_unique)

    def test_batch_has_distinct_positive_examples_and_negative_structures(self):
        path = self.root / "train.parquet"
        example_frame().to_parquet(path, index=False)
        dataset = SpectrumDataset(path)
        sampler = PKBatchSampler(dataset.labels, 3, 2, batches_per_epoch=4, seed=12)
        first_epoch = list(sampler)
        self.assertEqual(first_epoch, list(sampler))
        for indices in first_epoch:
            self.assertEqual(len(set(indices)), 6)
            batch = [dataset[i] for i in indices]
            x = np.stack([item["spectrum"] for item in batch])
            labels = np.array([item["label"] for item in batch])
            self.assertEqual(x.shape, (6, 1500))
            self.assertEqual(x.dtype, np.float32)
            self.assertEqual(labels.dtype, np.int64)
            self.assertEqual(np.unique(labels, return_counts=True)[1].tolist(), [2, 2, 2])
            self.assertTrue(np.isfinite(x).all())
            np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1.0, atol=1e-6)
        sampler.set_epoch(1)
        self.assertNotEqual(first_epoch, list(sampler))

    def test_insufficient_data_and_existing_outputs_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "至少需要 6"):
            build_training_split(example_frame(3), self.root / "too_small")
        self.assertFalse((self.root / "too_small").exists())
        build_training_split(example_frame(), self.root / "existing")
        with self.assertRaises(FileExistsError):
            build_training_split(example_frame(), self.root / "existing")
        with self.assertRaisesRegex(ValueError, "只有"):
            PKBatchSampler([0, 0, 1], structures_per_batch=2)
        with self.assertRaises(ValueError):
            PKBatchSampler([0, 0, 1, 1], structures_per_batch=1)

    def test_invalid_training_input_is_not_silently_truncated(self):
        frame = example_frame()
        frame.at[0, "ms2_normalized_intensities"] = [1.0]
        path = self.root / "invalid.parquet"
        frame.to_parquet(path, index=False)
        with self.assertRaisesRegex(ValueError, "第 0 行"):
            SpectrumDataset(path)


if __name__ == "__main__":
    unittest.main()
