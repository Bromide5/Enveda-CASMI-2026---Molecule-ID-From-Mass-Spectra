"""训练谱条件 Decoder，使用训练集 SMILES；不读取验证/holdout 标签。"""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from algorithms.transformer.dataset import SpectrumDataset
from algorithms.transformer.generation import SpectrumGenerator, SmilesTokenizer, canonical_smiles
from algorithms.transformer.model import SpectrumEncoder, SpectrumEncoderConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--encoder-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.num_layers) < 1 or args.max_length < 3 or args.seed < 0:
        parser.error("轮次、批次和层数需为正；长度至少 3，seed 非负。")
    if args.steps_per_epoch is not None and args.steps_per_epoch < 1:
        parser.error("steps-per-epoch 必须为正。")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("lr 必须为有限正数。")
    out = Path(args.out)
    if out.exists():
        raise FileExistsError("请使用新的实验目录。")
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    checkpoint = torch.load(args.encoder_checkpoint, map_location="cpu", weights_only=True)
    config = SpectrumEncoderConfig(**checkpoint["model_config"])
    encoder = SpectrumEncoder(config)
    encoder.load_state_dict(checkpoint["model_state_dict"])
    dataset = SpectrumDataset(args.train, config.bin_width, config.max_mz, config.power)
    smiles = [canonical_smiles(str(s)) for s in dataset.frame.normalized_smiles]
    if any(s is None for s in smiles):
        raise ValueError("训练标签包含非法/空 SMILES；请先清洗。")
    if max(map(len, smiles)) + 2 > args.max_length:
        raise ValueError(f"max-length 至少需要 {max(map(len, smiles)) + 2}，不截断标签。")
    tokenizer = SmilesTokenizer("".join(smiles))
    encoded = [tokenizer.encode(s) for s in smiles]
    model = SpectrumGenerator(encoder, tokenizer, args.num_layers, args.max_length).to(args.device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    decoder_config = dict(num_layers=args.num_layers, max_length=args.max_length)
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps(dict(arguments=vars(args), model_config=asdict(config),
        decoder_config=decoder_config, characters=tokenizer.characters, encoder_frozen=True,
        training_spectra=len(dataset)), ensure_ascii=False, indent=2), encoding="utf-8")
    history = []
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(dataset)).tolist()
        batches = [order[i:i + args.batch_size] for i in range(0, len(order), args.batch_size)]
        if args.steps_per_epoch:
            batches = batches[:args.steps_per_epoch]
        total, tokens_seen = 0.0, 0
        for step, indices in enumerate(batches):
            spectra = torch.stack([torch.from_numpy(dataset[i]["spectrum"]) for i in indices]).to(args.device)
            ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(encoded[i]) for i in indices],
                                                  batch_first=True).to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(spectra, ids[:, :-1])
            targets = ids[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, len(tokenizer)), targets.reshape(-1), ignore_index=0)
            if not torch.isfinite(loss):
                raise RuntimeError("非有限生成损失。")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            count = int(targets.ne(0).sum())
            total += loss.item() * count
            tokens_seen += count
            print(f"epoch={epoch+1} step={step+1}/{len(batches)} token_loss={loss.item():.5f}", flush=True)
        history.append(dict(epoch=epoch+1, token_loss=total/tokens_seen, steps=len(batches)))
        (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        temporary = out / "last.tmp.pt"
        torch.save(dict(model_config=asdict(config), decoder_config=decoder_config,
                        characters=tokenizer.characters, model_state_dict=model.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(), epoch=epoch+1,
                        encoder_frozen=True, training_arguments=vars(args)), temporary)
        temporary.replace(out / "last.pt")


if __name__ == "__main__":
    main()
