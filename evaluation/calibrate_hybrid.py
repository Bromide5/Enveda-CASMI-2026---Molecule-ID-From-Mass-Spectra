"""在混合验证集上选择检索接受阈值；不足以证明统计置信度。"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def calibrate(routing, labels, target_precision=0.95, max_unknown_accept=0.05, min_accepted=5):
    if not 0 < target_precision <= 1 or not 0 <= max_unknown_accept <= 1 or min_accepted < 1:
        raise ValueError("无效校准参数。")
    if not routing.molecule_id.is_unique or not labels.molecule_id.is_unique:
        raise ValueError("每个 molecule_id 必须唯一。")
    if set(routing.molecule_id) != set(labels.molecule_id):
        raise ValueError("校准预测与标签必须完全对应。")
    data = routing.merge(labels, on="molecule_id", validate="one_to_one")
    for column in ("top_score", "margin", "candidate_structures"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    known = data.in_library.astype(str).str.lower().eq("true").to_numpy()
    if known.all() or not known.any():
        raise ValueError("校准集必须同时包含库内和库外分子。")
    correct = data.top_key.eq(data.inchikey14).to_numpy() & known
    valid = data.candidate_structures.ge(2).to_numpy() & data.top_score.notna() & data.margin.notna()
    # 限制网格规模，所有候选阈值来自验证集，不使用 holdout。
    scores = np.unique(np.quantile(data.loc[valid, "top_score"], np.linspace(0, 1, 31))) if valid.any() else []
    margins = np.unique(np.quantile(data.loc[valid, "margin"], np.linspace(0, 1, 31))) if valid.any() else []
    best = dict(reject_all=True, accepted=0, reason="没有满足经验精度/库外误接受率及样本数条件的阈值")
    for score in scores:
        for margin in margins:
            accepted = np.asarray(valid & data.top_score.ge(score) & data.margin.ge(margin))
            n = int(accepted.sum())
            if n < min_accepted:
                continue
            precision = float(correct[accepted].mean())
            unknown_rate = float(accepted[~known].mean())
            if precision >= target_precision and unknown_rate <= max_unknown_accept and n > best["accepted"]:
                best = dict(reject_all=False, min_score=float(score), min_margin=float(margin),
                            accepted=n, empirical_precision=precision, unknown_accept_rate=unknown_rate)
    return dict(**best, calibration_molecules=len(data), target_precision=target_precision,
                max_unknown_accept=max_unknown_accept, min_accepted=min_accepted,
                warning="在校准样本上的经验指标，不是概率或独立测试结果；需在 holdout 验证。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-precision", type=float, default=0.95)
    parser.add_argument("--max-unknown-accept", type=float, default=0.05)
    parser.add_argument("--min-accepted", type=int, default=5)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError("门控文件已存在。")
    run = Path(args.run)
    report = calibrate(pd.read_csv(run / "routing.csv", keep_default_na=False),
                       pd.read_csv(args.labels), args.target_precision, args.max_unknown_accept, args.min_accepted)
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    report.update({key: config[key] for key in ("checkpoint_sha256", "reference_sha256", "ppm")})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
