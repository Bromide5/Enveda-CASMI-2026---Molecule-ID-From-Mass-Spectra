"""从已有验证划分构造库内/库外混合验证；不改变查询和答案。"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def build(source, out, unknown_fraction=0.5, seed=0):
    source, out = Path(source), Path(out)
    if out.exists():
        raise FileExistsError("请使用新的验证目录。")
    if not 0 < unknown_fraction < 1 or seed < 0:
        raise ValueError("unknown_fraction 必须在 (0,1)，seed 非负。")
    reference = pd.read_parquet(source / "reference.parquet")
    queries = pd.read_parquet(source / "queries.parquet")
    labels = pd.read_csv(source / "labels.csv", dtype=str)
    keys = sorted(labels.inchikey14.unique())
    if len(keys) < 2 or not set(keys).issubset(set(reference.inchikey14)):
        raise ValueError("原划分需至少两个结构，且参考库覆盖所有查询结构。")
    count = min(len(keys)-1, max(1, int(len(keys)*unknown_fraction)))
    unknown = set(np.random.default_rng(seed).choice(keys, count, replace=False))
    reference = reference[~reference.inchikey14.isin(unknown)]
    labels["in_library"] = ~labels.inchikey14.isin(unknown)
    if set(reference.inchikey14) & unknown:
        raise AssertionError("库外结构未完全移除。")
    out.mkdir(parents=True)
    reference.to_parquet(out / "reference.parquet", index=False)
    queries.to_parquet(out / "queries.parquet", index=False)
    labels.to_csv(out / "labels.csv", index=False)
    report = dict(source=str(source), seed=seed, known_structures=len(keys)-count,
                  unknown_structures=count, reference_spectra=len(reference),
                  scope="inchikey14-disjoint from reference; not scaffold-disjoint")
    (out / "split.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--unknown-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(build(args.source, args.out, args.unknown_fraction, args.seed))
