"""按结构划分神经网络训练集及独立检索验证集；先从抽样谱库开始。"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from algorithms.baseline.preprocess import COLUMNS
from algorithms.transformer.dataset import spectrum_is_usable, validate_preprocessing
from evaluation.build_validation import spectrum_signature


def build_training_split(frame, out, validation_fraction=0.1, holdout_fraction=0.1,
                         spectra_per_structure=2, queries_per_structure=1, seed=0,
                         max_mz=1500.0, source=None):
    """按 inchikey14 划分；验证/holdout 内保留同结构的其他谱作参考。

    删除缺失标签、无效谱、完全重复谱、标签冲突谱及谱数不足的结构。
    签名复用现有验证脚本的原始峰与前体字段摘要。
    每个查询分子可含多张查询谱，但至少留一张参考谱。
    """
    out = Path(out)
    if out.exists():
        raise FileExistsError("输出目录已存在，请使用新目录以保留已有划分。")
    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"输入缺少字段：{sorted(missing)}")
    if (not np.isfinite(validation_fraction) or not np.isfinite(holdout_fraction)
            or validation_fraction <= 0 or holdout_fraction <= 0
            or validation_fraction + holdout_fraction >= 1):
        raise ValueError("验证和 holdout 比例必须为正数，且和小于 1。")
    if spectra_per_structure < 2 or queries_per_structure < 1 or seed < 0:
        raise ValueError("每结构训练谱数至少 2，查询谱数至少 1，seed 非负。")
    validate_preprocessing(1.0, max_mz, 0.5)
    frame = frame.reset_index(drop=True).copy()
    record = dict(seed=seed, input_spectra=len(frame), source=source,
                  requested_fractions=dict(validation=validation_fraction, holdout=holdout_fraction),
                  spectra_per_structure=spectra_per_structure,
                  queries_per_structure=queries_per_structure, max_mz=max_mz)
    # input_row 总是指向此次输入文件行号；source_row 若存在则保留其原始训练文件行号。
    frame["input_row"] = np.arange(len(frame), dtype=np.int64)
    usable_labels = frame[["inchikey14", "normalized_smiles", "adduct", "ionization_mode"]].notna().all(axis=1)
    for column in ("inchikey14", "normalized_smiles", "adduct", "ionization_mode"):
        usable_labels &= frame[column].astype(str).str.strip().ne("")
    record["dropped_missing_metadata"] = int((~usable_labels).sum())
    frame = frame[usable_labels].copy()
    frame["inchikey14"] = frame.inchikey14.astype(str).str.strip()
    usable = np.array([spectrum_is_usable(row, max_mz) for row in frame.itertuples()], dtype=bool)
    masses = pd.to_numeric(frame.precursor_mz, errors="coerce").to_numpy(dtype=float)
    usable &= np.isfinite(masses) & (masses > 0)
    record["dropped_invalid_spectra"] = int((~usable).sum())
    frame = frame[usable].copy()
    if frame.empty:
        raise ValueError("清洗后没有可用谱。")
    frame["signature"] = frame.apply(spectrum_signature, axis=1)
    conflicts = frame.groupby("signature").inchikey14.nunique()
    conflicts = set(conflicts[conflicts > 1].index)
    record["conflicting_signatures"] = len(conflicts)
    conflicting_rows = frame.signature.isin(conflicts)
    record["dropped_conflicting_spectra"] = int(conflicting_rows.sum())
    frame = frame[~conflicting_rows].copy()
    record["dropped_exact_duplicates"] = int(frame.signature.duplicated().sum())
    frame = frame.drop_duplicates("signature")
    minimum = max(spectra_per_structure, queries_per_structure + 1)
    counts = frame.groupby("inchikey14").size()
    eligible = sorted(counts[counts >= minimum].index)
    record["dropped_insufficient_structures"] = int((counts < minimum).sum())
    record["dropped_insufficient_spectra"] = int(counts[counts < minimum].sum())
    n = len(eligible)
    # 至少两个结构才能在每组中观察正样本与不同结构候选。
    n_validation = max(2, int(np.floor(n * validation_fraction)))
    n_holdout = max(2, int(np.floor(n * holdout_fraction)))
    if n - n_validation - n_holdout < 2:
        raise ValueError("可用结构不足以划分三组；请扩大抽样量，至少需要 6 个合格结构。")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(eligible)
    assignments = dict(validation=shuffled[:n_validation],
                       holdout=shuffled[n_validation:n_validation + n_holdout],
                       train=shuffled[n_validation + n_holdout:])
    prepared = {}
    for name, keys in assignments.items():
        prepared[name] = frame[frame.inchikey14.isin(keys)].sort_values("input_row").copy()
    key_sets = {name: set(part.inchikey14) for name, part in prepared.items()}
    for left, right in (("train", "validation"), ("train", "holdout"), ("validation", "holdout")):
        if key_sets[left] & key_sets[right]:
            raise ValueError("结构划分发生交叉。")

    out.mkdir(parents=True)
    prepared["train"].drop(columns="signature").to_parquet(out / "train.parquet", index=False)
    manifest_parts = []
    stats = {}
    for name in ("train", "validation", "holdout"):
        part = prepared[name]
        manifest = part[["input_row", "inchikey14", "signature"]].copy()
        if "source_row" in part.columns:
            manifest["source_row"] = part.source_row
        manifest["split"] = name
        manifest["role"] = "train" if name == "train" else "reference"
        stats[name] = dict(structures=part.inchikey14.nunique(), spectra=len(part))
        if name != "train":
            pieces = []
            for i, key in enumerate(sorted(key_sets[name])):
                group = part[part.inchikey14 == key]
                indices = rng.choice(group.index.to_numpy(), queries_per_structure, replace=False)
                query = group.loc[indices].copy()
                query["molecule_id"] = f"{name}_{i:05d}"
                query["spectrum_id"] = [f"{name}_{i:05d}_s{j}" for j in range(len(query))]
                pieces.append(query)
            queries = pd.concat(pieces, ignore_index=True)
            reference = part[~part.signature.isin(queries.signature)].copy()
            if set(queries.signature) & set(reference.signature):
                raise ValueError("查询谱与参考谱重复。")
            if not set(queries.inchikey14).issubset(set(reference.inchikey14)):
                raise ValueError("参考库未覆盖全部查询结构。")
            folder = out / name
            folder.mkdir()
            reference.drop(columns="signature").to_parquet(folder / "reference.parquet", index=False)
            labels = queries.drop_duplicates("molecule_id")[["molecule_id", "normalized_smiles", "inchikey14"]]
            labels.rename(columns={"normalized_smiles": "smiles"}).to_csv(folder / "labels.csv", index=False)
            queries.drop(columns=["signature", "inchikey14", "normalized_smiles"]).to_parquet(
                folder / "queries.parquet", index=False)
            manifest.loc[manifest.signature.isin(queries.signature), "role"] = "query"
            stats[name].update(query_spectra=len(queries), reference_spectra=len(reference),
                               inchikey14_reference_coverage=1.0)
        manifest_parts.append(manifest)
    pd.concat(manifest_parts, ignore_index=True).to_csv(out / "manifest.csv", index=False)
    record.update(eligible_structures=n, retained_spectra=sum(s["spectra"] for s in stats.values()),
                  splits=stats, structure_disjoint=True, signature_disjoint=True,
                  duplicate_policy="Exact sorted raw peaks, precursor_mz, adduct and ionization_mode",
                  evaluation_scope="Known-library retrieval of structures unseen by model training",
                  note="Raw inchikey14 split; not scaffold-disjoint or audited for tautomer equivalence. "
                       "Structure coverage is before mass/adduct filtering. This is a small-sample pipeline.")
    (out / "split.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="抽样参考谱文件；会加载到内存")
    parser.add_argument("--out", required=True, help="新的输出目录")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    parser.add_argument("--spectra-per-structure", type=int, default=2)
    parser.add_argument("--queries-per-structure", type=int, default=1)
    parser.add_argument("--max-mz", type=float, default=1500.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    record = build_training_split(
        pd.read_parquet(args.reference), args.out, args.validation_fraction,
        args.holdout_fraction, args.spectra_per_structure,
        args.queries_per_structure, args.seed, args.max_mz,
        source=str(Path(args.reference).resolve()),
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
