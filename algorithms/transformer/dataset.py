"""分箱质谱数据集与 P 个结构 × K 张谱采样；可直接接入 PyTorch DataLoader。"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from algorithms.baseline.preprocess import spectrum_vector


def validate_preprocessing(bin_width, max_mz, power):
    """避免错误配置产生空向量或非有限输入。"""
    if not all(np.isfinite(v) and v > 0 for v in (bin_width, max_mz, power)):
        raise ValueError("bin_width、max_mz、power 必须为有限正数。")


def spectrum_is_usable(row, max_mz=1500.0):
    """检查峰数组长度、数值和所选质量范围内是否有正强度峰。"""
    try:
        mzs = np.asarray(row.ms2_mzs, dtype=float)
        intensities = np.asarray(row.ms2_normalized_intensities, dtype=float)
    except (TypeError, ValueError):
        return False
    return bool(
        mzs.ndim == intensities.ndim == 1
        and len(mzs) == len(intensities)
        and np.isfinite(mzs).all()
        and np.isfinite(intensities).all()
        and (mzs >= 0).all()
        and (intensities >= 0).all()
        and ((mzs < max_mz) & (intensities > 0)).any()
    )


class SpectrumDataset:
    """加载已划分的训练谱，按需分箱，不预先构建全量稠密矩阵。

    实现 __len__ / __getitem__ 即可交给 torch.utils.data.DataLoader。
    默认返回 NumPy 数组和 int64 标签；DataLoader 会自动转换成张量。
    标签只用于计算损失，不应输入编码器。此版本将原始峰加载到内存，
    面向抽样实验，不适合直接加载数 GB 的原始训练文件。
    """

    def __init__(self, path, bin_width=1.0, max_mz=1500.0, power=0.5, strict=True):
        validate_preprocessing(bin_width, max_mz, power)
        self.frame = pd.read_parquet(path).reset_index(drop=True)
        required = {"ms2_mzs", "ms2_normalized_intensities", "inchikey14"}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"缺少训练字段：{sorted(missing)}")
        if self.frame.empty:
            raise ValueError("训练文件为空。")
        keys = self.frame.inchikey14
        valid = (~keys.isna() & keys.astype(str).str.strip().ne("")).to_numpy()
        valid &= np.array([spectrum_is_usable(row, max_mz)
                           for row in self.frame.itertuples()], dtype=bool)
        self.dropped_rows = int((~valid).sum())
        if self.dropped_rows and strict:
            first_bad = int(np.flatnonzero(~valid)[0])
            raise ValueError(f"第 {first_bad} 行不是可用谱；请用 build_training_split 重新准备数据。")
        if not valid.any():
            raise ValueError("清洗后没有可用训练谱。")
        if self.dropped_rows:
            self.frame = self.frame.loc[valid].reset_index(drop=True)
            keys = self.frame.inchikey14
        self.bin_width, self.max_mz, self.power = bin_width, max_mz, power
        self.structure_keys = sorted(keys.astype(str).str.strip().unique())
        key_to_label = {key: i for i, key in enumerate(self.structure_keys)}
        self.labels = keys.astype(str).str.strip().map(key_to_label).to_numpy(dtype=np.int64)
        self.n_bins = int(np.ceil(max_mz / bin_width))

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[int(index)]
        vector = spectrum_vector(
            row.ms2_mzs, row.ms2_normalized_intensities,
            bin_width=self.bin_width, max_mz=self.max_mz, power=self.power,
        )
        if not np.isfinite(vector).all() or not np.any(vector):
            raise ValueError(f"第 {index} 行分箱结果无效，请检查数据和强度变换参数。")
        return {"spectrum": vector, "label": self.labels[int(index)]}


class PKBatchSampler:
    """每批抽 P 个不同结构，每个结构抽 K 个不同记录。

    随机抽样允许跨批重复，不保证一个 epoch 遍历全部谱。
    set_epoch(epoch) 使不同轮次产生不同且可复现的样本序列。
    可作为 PyTorch DataLoader 的 batch_sampler，无需继承特定基类。
    """

    def __init__(self, labels, structures_per_batch=8, spectra_per_structure=2,
                 batches_per_epoch=None, seed=0):
        self.p, self.k = int(structures_per_batch), int(spectra_per_structure)
        if self.p < 2 or self.k < 2:
            raise ValueError("P 和 K 都必须至少为 2，以提供不同结构负样本和同结构正样本。")
        labels = np.asarray(labels)
        if labels.ndim != 1:
            raise ValueError("labels 必须是一维数组。")
        self.groups = {}
        for label in np.unique(labels):
            positions = np.flatnonzero(labels == label)
            if len(positions) >= self.k:
                self.groups[int(label)] = positions
        self.eligible_labels = np.array(sorted(self.groups), dtype=np.int64)
        if len(self.groups) < self.p:
            raise ValueError(
                f"只有 {len(self.groups)} 个结构至少有 {self.k} 张谱，无法抽取 P={self.p}。"
                "请降低 P/K 或扩大样本。"
            )
        self.steps = (math.ceil(len(self.groups) / self.p)
                      if batches_per_epoch is None else int(batches_per_epoch))
        if self.steps < 1 or seed < 0:
            raise ValueError("batches_per_epoch 必须为正数，seed 必须非负。")
        self.seed, self.epoch = int(seed), 0

    def set_epoch(self, epoch):
        if epoch < 0:
            raise ValueError("epoch 必须非负。")
        self.epoch = int(epoch)

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        for _ in range(self.steps):
            selected = rng.choice(self.eligible_labels, self.p, replace=False)
            batch = []
            for label in selected:
                batch.extend(rng.choice(self.groups[int(label)], self.k, replace=False).tolist())
            yield batch


def main():
    """无需安装 PyTorch 即可检查划分后的训练输入和正负样本关系。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--structures-per-batch", type=int, default=8)
    parser.add_argument("--spectra-per-structure", type=int, default=2)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bin-width", type=float, default=1.0)
    parser.add_argument("--max-mz", type=float, default=1500.0)
    parser.add_argument("--power", type=float, default=0.5)
    parser.add_argument("--out", help="可选：保存批次检查结果 JSON")
    args = parser.parse_args()
    dataset = SpectrumDataset(args.train, args.bin_width, args.max_mz, args.power)
    sampler = PKBatchSampler(dataset.labels, args.structures_per_batch,
                             args.spectra_per_structure, args.batches, args.seed)
    batches = []
    for indices in sampler:
        samples = [dataset[i] for i in indices]
        spectra = np.stack([s["spectrum"] for s in samples])
        labels = np.array([s["label"] for s in samples], dtype=np.int64)
        _, counts = np.unique(labels, return_counts=True)
        norms = np.linalg.norm(spectra, axis=1)
        if (len(set(indices)) != len(indices) or len(counts) != sampler.p
                or not np.all(counts == sampler.k) or not np.isfinite(spectra).all()
                or not np.allclose(norms, 1.0, atol=1e-5)):
            raise ValueError("批次检查失败。")
        batches.append(dict(spectra_shape=list(spectra.shape), labels_shape=list(labels.shape),
                            spectra_dtype=str(spectra.dtype), labels_dtype=str(labels.dtype),
                            unique_structures=len(counts), spectra_per_structure=counts.tolist(),
                            all_finite=True, all_unit_norm=True, distinct_records=True))
    report = dict(train=str(args.train), training_spectra=len(dataset),
                  training_structures=len(dataset.structure_keys),
                  eligible_structures=len(sampler.groups), seed=args.seed,
                  preprocessing=dict(bin_width=args.bin_width, max_mz=args.max_mz, power=args.power),
                  batches=batches)
    if args.out:
        path = Path(args.out)
        if path.exists():
            raise FileExistsError("检查报告已存在，请选择新的 --out。")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
