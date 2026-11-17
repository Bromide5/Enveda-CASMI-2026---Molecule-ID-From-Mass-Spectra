"""训练共享谱编码器并保存最后一轮权重；尚不执行检索验证或选择最佳模型。"""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from algorithms.transformer.dataset import SpectrumDataset, PKBatchSampler
from algorithms.transformer.losses import supervised_contrastive_loss
from algorithms.transformer.model import SpectrumEncoder, SpectrumEncoderConfig


def train_step(model, optimizer, spectra, labels, temperature=0.1, max_grad_norm=1.0):
    """完整批次共同计算正负样本关系，反向传播后更新一次权重。"""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    vectors = model(spectra)
    loss = supervised_contrastive_loss(vectors, labels, temperature)
    if not torch.isfinite(loss):
        raise RuntimeError("训练损失非有限，停止更新。")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm,
                                          error_if_nonfinite=True)
    optimizer.step()
    return loss.item(), float(norm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--out", required=True, help="新建实验目录，不覆盖已有目录")
    parser.add_argument("--config", default="configs/transformer.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batches-per-epoch", type=int)
    parser.add_argument("--structures-per-batch", type=int, default=2)
    parser.add_argument("--spectra-per-structure", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.epochs < 1 or args.seed < 0:
        parser.error("epochs 必须为正数，seed 必须非负。")
    for name in ("lr", "temperature", "max_grad_norm"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name} 必须为有限正数。")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight-decay 必须有限且非负。")
    out = Path(args.out)
    if out.exists():
        raise FileExistsError("实验目录已存在，请选择新的 --out。")
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    config = SpectrumEncoderConfig(**json.loads(Path(args.config).read_text(encoding="utf-8")))
    dataset = SpectrumDataset(args.train, config.bin_width, config.max_mz, config.power)
    sampler = PKBatchSampler(dataset.labels, args.structures_per_batch,
                             args.spectra_per_structure, args.batches_per_epoch, args.seed)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    model = SpectrumEncoder(config).to(args.device)
    initial = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    out.mkdir(parents=True, exist_ok=False)
    metadata = dict(arguments=vars(args), model_config=asdict(config),
                    torch_version=str(torch.__version__), training_spectra=len(dataset),
                    training_structures=len(dataset.structure_keys),
                    eligible_structures=len(sampler.groups), batches_per_epoch=len(sampler),
                    selection="last epoch; no validation-based selection")
    (out / "config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    history = []
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        losses, norms = [], []
        for step, batch in enumerate(loader):
            loss, norm = train_step(model, optimizer, batch["spectrum"].to(args.device),
                                    batch["label"].to(args.device), args.temperature, args.max_grad_norm)
            losses.append(loss)
            norms.append(norm)
            print(f"epoch={epoch + 1}/{args.epochs} step={step + 1}/{len(sampler)} loss={loss:.6f}", flush=True)
        row = dict(epoch=epoch + 1, mean_loss=sum(losses) / len(losses),
                   mean_gradient_norm=sum(norms) / len(norms), updates=(epoch + 1) * len(sampler))
        history.append(row)
        (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        checkpoint = dict(model_config=asdict(config), model_state_dict=model.state_dict(),
                          optimizer_state_dict=optimizer.state_dict(), epoch=epoch + 1,
                          updates=row["updates"], training_arguments=vars(args),
                          torch_rng_state=torch.get_rng_state())
        temporary = out / "last.tmp.pt"
        torch.save(checkpoint, temporary)
        temporary.replace(out / "last.pt")
    delta = sum((p.detach().cpu() - initial[name]).square().sum().item()
                for name, p in model.named_parameters()) ** 0.5
    report = dict(updates=history[-1]["updates"], parameter_change_l2=delta,
                  weights_updated=delta > 0, retrieval_evaluated=False,
                  final_train_loss=history[-1]["mean_loss"])
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
