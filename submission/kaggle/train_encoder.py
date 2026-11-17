"""在 Kaggle GPU 上训练谱 Encoder；默认读取竞赛挂载的 train.parquet。

这是训练入口，不是提交入口。训练完成后请下载 --out 中的 checkpoint，
再用本地验证集校准检索/生成门控。
"""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

# Kaggle 打包时会把 algorithms/configs 放在脚本旁边；本地运行时则从项目根导入。
for PROJECT_ROOT in (Path(__file__).resolve().parent, Path.cwd(), Path(__file__).resolve().parents[2]):
    if (PROJECT_ROOT / "algorithms").exists():
        sys.path.insert(0, str(PROJECT_ROOT))
        break

import numpy as np
import torch
from torch.utils.data import DataLoader

from algorithms.transformer.dataset import PKBatchSampler, SpectrumDataset
from algorithms.transformer.losses import supervised_contrastive_loss
from algorithms.transformer.model import SpectrumEncoder, SpectrumEncoderConfig


def find_competition_train():
    """在 Kaggle 的只读输入目录递归查找竞赛 train.parquet。"""
    root = Path("/kaggle/input")
    candidates = sorted(root.rglob("train.parquet")) if root.exists() else []
    if not candidates:
        raise FileNotFoundError("未在 /kaggle/input 找到 train.parquet，请用 --train 指定路径。")
    if len(candidates) > 1:
        print("发现多个 train.parquet，使用：", candidates[0], flush=True)
    return candidates[0]


def make_loader(dataset, p, k, batches, seed, workers):
    sampler = PKBatchSampler(dataset.labels, p, k, batches_per_epoch=batches, seed=seed)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=workers,
                        pin_memory=torch.cuda.is_available())
    return loader, sampler


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.amp and device.type != "cuda":
        print("--amp 只在 CUDA 上启用；当前改为关闭。", flush=True)
        args.amp = False
    torch.set_num_threads(min(4, torch.get_num_threads()))
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).resolve().parent / args.config
    config = SpectrumEncoderConfig(**json.loads(config_path.read_text(encoding="utf-8")))
    train_path = Path(args.train) if args.train else find_competition_train()
    # 官方全量 parquet 可能含空峰、越界峰或缺失结构标签；训练时跳过坏行并记录数量。
    dataset = SpectrumDataset(train_path, config.bin_width, config.max_mz, config.power, strict=False)
    if dataset.dropped_rows:
        print(f"dropped_invalid_training_rows={dataset.dropped_rows}", flush=True)
    loader, sampler = make_loader(dataset, args.structures_per_batch,
                                  args.spectra_per_structure, args.batches_per_epoch,
                                  args.seed, args.num_workers)
    model = SpectrumEncoder(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"输出目录已存在：{out}；请使用新目录。")
    out.mkdir(parents=True)
    metadata = dict(train_path=str(train_path), device=str(device), amp=args.amp,
                    dataset_spectra=len(dataset), dataset_structures=len(dataset.structure_keys),
                    dropped_invalid_training_rows=dataset.dropped_rows,
                    model_config=asdict(config), arguments=vars(args),
                    note="Kaggle training checkpoint; validate and calibrate locally before submission")
    (out / "config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    history = []
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        losses, grad_norms = [], []
        for step, batch in enumerate(loader):
            spectra = batch["spectrum"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                vectors = model(spectra)
                loss = supervised_contrastive_loss(vectors, labels, args.temperature)
            if not torch.isfinite(loss):
                raise RuntimeError(f"epoch {epoch + 1} step {step + 1} 损失非有限。")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm,
                                                  error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            grad_norms.append(float(norm.detach().cpu() if torch.is_tensor(norm) else norm))
            if step % args.log_every == 0:
                print(f"epoch={epoch + 1}/{args.epochs} step={step + 1}/{len(sampler)} "
                      f"loss={losses[-1]:.6f}", flush=True)
        row = dict(epoch=epoch + 1, mean_loss=sum(losses) / len(losses),
                   mean_gradient_norm=sum(grad_norms) / len(grad_norms),
                   updates=(epoch + 1) * len(sampler))
        history.append(row)
        (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        temporary = out / "last.tmp.pt"
        torch.save(dict(model_config=asdict(config), model_state_dict=model.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(), epoch=epoch + 1,
                        updates=row["updates"], training_arguments=vars(args),
                        torch_rng_state=torch.get_rng_state()), temporary)
        temporary.replace(out / "last.pt")
    summary = dict(epochs=args.epochs, updates=history[-1]["updates"],
                   final_train_loss=history[-1]["mean_loss"], checkpoint=str(out / "last.pt"),
                   device=str(device), amp=args.amp)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", help="可选；Kaggle 上默认自动查找 /kaggle/input/**/train.parquet")
    parser.add_argument("--config", default="configs/transformer.json")
    parser.add_argument("--out", default="/kaggle/working/encoder_run")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batches-per-epoch", type=int, default=1000)
    parser.add_argument("--structures-per-batch", type=int, default=8)
    parser.add_argument("--spectra-per-structure", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="CUDA 上启用混合精度")
    args = parser.parse_args()
    if args.epochs < 1 or args.batches_per_epoch < 1 or args.structures_per_batch < 2 \
            or args.spectra_per_structure < 2 or args.num_workers < 0 or args.log_every < 1:
        parser.error("epochs/batches/P/K/log 参数无效。")
    for name in ("lr", "temperature", "max_grad_norm"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} 必须为有限正数。")
    if args.weight_decay < 0 or not math.isfinite(args.weight_decay):
        parser.error("weight-decay 必须有限且非负。")
    train(args)


if __name__ == "__main__":
    main()
