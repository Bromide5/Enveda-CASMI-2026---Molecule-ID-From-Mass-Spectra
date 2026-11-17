"""按库内/库外报告排名及路由；采用 InChIKey 前 14 位，不等同官方评分。"""

import argparse
import json
from pathlib import Path
import pandas as pd
from rdkit import Chem, rdBase


def structure_key(smiles):
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(mol).split("-")[0] if mol is not None and mol.GetNumAtoms() else None


def evaluate(run, labels_path):
    run = Path(run)
    predictions = pd.read_csv(run / "predictions.csv", keep_default_na=False, dtype=str)
    routing = pd.read_csv(run / "routing.csv", keep_default_na=False, dtype={"molecule_id": str})
    labels = pd.read_csv(labels_path, dtype=str)
    for frame in (predictions, routing):
        if not frame.molecule_id.is_unique or set(frame.molecule_id) != set(labels.molecule_id):
            raise ValueError("预测/路由和标签的编号必须唯一且完全对应。")
    data = labels.merge(predictions, on="molecule_id", suffixes=("_truth", "_pred"), validate="one_to_one")
    data = data.merge(routing[["molecule_id", "route"]], on="molecule_id", validate="one_to_one")
    rows = []
    for row in data.itertuples():
        truth = structure_key(row.smiles_truth)
        if truth is None:
            raise ValueError("非法真实标签。")
        candidates = row.smiles_pred.split(";") if row.smiles_pred else []
        keys = [structure_key(s) for s in candidates[:25]]
        rank = keys.index(truth)+1 if truth in keys else 0
        rows.append(dict(molecule_id=row.molecule_id, in_library=str(row.in_library).lower()=="true",
                         route=row.route, rank=rank, reciprocal_rank=1/rank if rank else 0,
                         top1=int(rank==1), top5=int(0<rank<=5), top25=int(rank>0),
                         valid_candidates=sum(k is not None for k in keys)))
    frame = pd.DataFrame(rows)
    report = dict(matching="InChIKey14; no tautomer standardization; not official score")
    for name, part in (("all", frame), ("known", frame[frame.in_library]), ("unknown", frame[~frame.in_library])):
        report[name] = dict(molecules=len(part),
                           **{col: float(part[col].mean()) if len(part) else None
                              for col in ("top1", "top5", "top25", "reciprocal_rank")},
                           routes=part.route.value_counts().to_dict())
    unknown = frame[~frame.in_library]
    report["unknown_retrieval_accept_rate"] = float(unknown.route.eq("retrieval").mean()) if len(unknown) else None
    return frame, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--labels", required=True)
    args = parser.parse_args()
    out = Path(args.run)
    if (out / "hybrid_metrics.json").exists():
        raise FileExistsError("评估结果已存在。")
    frame, report = evaluate(out, args.labels)
    frame.to_csv(out / "hybrid_per_molecule.csv", index=False)
    (out / "hybrid_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
