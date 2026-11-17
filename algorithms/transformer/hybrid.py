"""严格筛选后的谱库检索；未通过门控时使用谱条件 Decoder。"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from algorithms.baseline.preprocess import spectrum_vector
from algorithms.transformer.generation import load_generator


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensorize(frame, config):
    arrays = [spectrum_vector(row.ms2_mzs, row.ms2_normalized_intensities,
               bin_width=config.bin_width, max_mz=config.max_mz, power=config.power)
              for row in frame.itertuples()]
    return torch.from_numpy(np.stack(arrays)) if arrays else torch.empty(0, config.n_bins)


@torch.no_grad()
def retrieve(encoder, reference, queries, device="cpu", batch_size=2, ppm=20.0):
    """多谱 max 合并，按结构去重；无质量匹配时不放宽筛选。"""
    if batch_size < 1 or not math.isfinite(ppm) or ppm < 0:
        raise ValueError("batch_size 必须为正，ppm 有限且非负。")
    encoder.eval()
    reference = reference.reset_index(drop=True)
    queries = queries.reset_index(drop=True)
    rv = tensorize(reference, encoder.config)
    qv = tensorize(queries, encoder.config)
    embeddings = (torch.cat([encoder(x.to(device)).cpu() for x in rv.split(batch_size)])
                  if len(rv) else torch.empty(0, encoder.config.embedding_dim))
    qemb = torch.cat([encoder(x.to(device)).cpu() for x in qv.split(batch_size)])
    ranked, diagnostics = {}, []
    masses = reference.precursor_mz.to_numpy(float)
    for molecule_id, group in queries.groupby("molecule_id", sort=False):
        evidence = {}
        for i, row in group.iterrows():
            error = np.abs(masses-row.precursor_mz) / max(abs(row.precursor_mz), 1e-12) * 1e6
            chosen = np.flatnonzero((reference.adduct.eq(row.adduct)
                       & reference.ionization_mode.eq(row.ionization_mode)).to_numpy() & (error <= ppm))
            scores = embeddings[chosen] @ qemb[i]
            for index, score in zip(chosen, scores.tolist()):
                item = reference.iloc[index]
                key = str(item.inchikey14)
                if key not in evidence or score > evidence[key][0]:
                    evidence[key] = (score, str(item.normalized_smiles))
        ranking = sorted(evidence, key=lambda key: (-evidence[key][0], key))
        ranked[str(molecule_id)] = [evidence[key][1] for key in ranking[:25]]
        top = evidence[ranking[0]][0] if ranking else None
        # 单候选时缺乏分数差证据，保守地交给生成分支。
        margin = top-evidence[ranking[1]][0] if len(ranking) > 1 else None
        diagnostics.append(dict(molecule_id=str(molecule_id), top_score=top, margin=margin,
                                candidate_structures=len(ranking),
                                top_key=ranking[0] if ranking else ""))
    return ranked, pd.DataFrame(diagnostics), qv


def accepts(row, gate):
    return (not gate.get("reject_all", False) and row.candidate_structures >= 2
            and pd.notna(row.top_score) and pd.notna(row.margin)
            and row.top_score >= gate["min_score"] and row.margin >= gate["min_margin"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gate", help="验证集校准得到的门控 JSON；缺省时全部交给生成")
    parser.add_argument("--retrieval-only", action="store_true", help="输出检索诊断用于校准，不运行生成")
    parser.add_argument("--ppm", type=float, default=20)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, help="可选生成预算；超限但未输出 EOS 的序列丢弃")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError("请使用新的输出目录。")
    if args.beam_size < 1 or (args.max_new_tokens is not None and args.max_new_tokens < 1):
        parser.error("beam-size/max-new-tokens 必须为正。")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = load_generator(args.checkpoint, args.device)
    reference, queries = pd.read_parquet(args.reference), pd.read_parquet(args.queries)
    provenance = dict(checkpoint_sha256=file_hash(args.checkpoint),
                      reference_sha256=file_hash(args.reference), ppm=args.ppm)
    gate = json.loads(Path(args.gate).read_text(encoding="utf-8")) if args.gate else {"reject_all": True}
    if args.gate:
        # 更换模型/谱库/筛选规则后必须重新验证门控。
        if gate["checkpoint_sha256"] != provenance["checkpoint_sha256"] or gate["ppm"] != args.ppm:
            raise ValueError("门控的模型或 ppm 与本次运行不一致。")
    candidates, diagnostic, qv = retrieve(model.encoder, reference, queries, args.device, ppm=args.ppm)
    outputs, routes = [], []
    for row in diagnostic.itertuples():
        if args.retrieval_only or accepts(row, gate):
            result = candidates[row.molecule_id]
            route = "retrieval_only" if args.retrieval_only else "retrieval"
        else:
            generated = {}
            indices = np.flatnonzero(queries.molecule_id.astype(str).eq(row.molecule_id).to_numpy())
            for i in indices:
                for smiles, score in model.generate(qv[i:i+1].to(args.device), args.beam_size,
                                                     max_new_tokens=args.max_new_tokens):
                    generated[smiles] = max(score, generated.get(smiles, -math.inf))
            result = sorted(generated, key=lambda s: (-generated[s], s))[:25]
            route = "generation" if result else "generation_failed"
        outputs.append(dict(molecule_id=row.molecule_id, smiles=";".join(result)))
        routes.append(route)
        print(f"{row.molecule_id}: {route}, candidates={len(result)}", flush=True)
    diagnostic["route"] = routes
    out.mkdir(parents=True)
    pd.DataFrame(outputs).to_csv(out / "predictions.csv", index=False)
    diagnostic.to_csv(out / "routing.csv", index=False)
    (out / "config.json").write_text(json.dumps(dict(arguments=vars(args), **provenance,
        reference_changed_since_calibration=bool(args.gate and gate.get("reference_sha256") != provenance["reference_sha256"]),
        threshold_is_probability=False), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
