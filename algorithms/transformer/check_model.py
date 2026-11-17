"""检查真实数据的编码与梯度连接；不训练模型、不评价检索准确率。"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from algorithms.transformer.dataset import SpectrumDataset, PKBatchSampler
from algorithms.transformer.model import SpectrumEncoder, SpectrumEncoderConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--config", default="configs/transformer.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--structures-per-batch", type=int, default=8)
    parser.add_argument("--spectra-per-structure", type=int, default=2)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.micro_batch_size < 1:
        parser.error("micro-batch-size 必须为正数。")
    if args.out and Path(args.out).exists():
        raise FileExistsError("报告已存在，请选择新的 --out。")
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    config = SpectrumEncoderConfig(**json.loads(Path(args.config).read_text(encoding="utf-8")))
    dataset = SpectrumDataset(args.train, config.bin_width, config.max_mz, config.power)
    sampler = PKBatchSampler(dataset.labels, args.structures_per_batch,
                             args.spectra_per_structure, batches_per_epoch=1, seed=args.seed)
    batch = next(iter(DataLoader(dataset, batch_sampler=sampler, num_workers=0)))
    spectra = batch["spectrum"].to(args.device)
    model = SpectrumEncoder(config).to(args.device)
    model.eval()
    with torch.no_grad():
        embedded = model.input_embedding(spectra[:args.micro_batch_size])
        outputs = torch.cat([model(chunk) for chunk in spectra.split(args.micro_batch_size)])
    if not torch.isfinite(outputs).all() or not torch.allclose(
            outputs.norm(dim=1), torch.ones(len(spectra), device=args.device), atol=1e-5):
        raise RuntimeError("输出必须有限且具有单位范数。")
    report = dict(config=asdict(config), seed=args.seed, torch_version=torch.__version__,
                  device=args.device, input_shape=list(spectra.shape),
                  micro_batch_token_shape=list(embedded.shape), output_shape=list(outputs.shape),
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  finite_unit_embeddings=True, trained=False, backward_checked=False)
    if args.backward:
        model.train()
        probe = model(spectra[:args.micro_batch_size])
        target = F.normalize(torch.randn_like(probe), dim=-1)
        loss = F.mse_loss(probe, target)
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        if any(g is None or not torch.isfinite(g).all() for g in gradients):
            raise RuntimeError("参数梯度缺失或非有限。")
        grad_norm = sum(g.square().sum().item() for g in gradients) ** 0.5
        if grad_norm == 0:
            raise RuntimeError("梯度全为零。")
        report.update(backward_checked=True, gradient_norm=grad_norm,
                      diagnostic_loss=loss.item(),
                      diagnostic_note="随机目标仅测试梯度连接，不是正式训练损失；没有更新参数。")
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
