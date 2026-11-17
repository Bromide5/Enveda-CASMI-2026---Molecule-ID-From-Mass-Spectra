"""监督式多正样本对比损失：同结构的其他谱都是正样本。"""

import math

import torch
from torch.nn import functional as F


def supervised_contrastive_loss(vectors, labels, temperature=0.1):
    """对每个 anchor 的所有正样本 log 概率求平均，再对 anchor 求平均。

    分母包含除自身以外的所有谱；自身既不作为正样本也不进入分母。
    要求每个结构至少两张谱，且至少包含两个结构（由 PK 采样保证）。
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature 必须为有限正数。")
    if vectors.ndim != 2 or vectors.shape[0] < 4 or vectors.shape[1] < 1:
        raise ValueError("vectors 必须为 [B,D]，B 至少为 4。")
    if labels.ndim != 1 or len(labels) != len(vectors):
        raise ValueError("labels 必须为与 vectors 对齐的一维标签。")
    if labels.device != vectors.device:
        raise ValueError("标签和向量必须在同一设备。")
    if not vectors.is_floating_point() or not torch.isfinite(vectors).all():
        raise ValueError("向量必须为有限浮点数。")
    if (vectors.norm(dim=1) == 0).any():
        raise ValueError("不能使用零向量。")
    diagonal = torch.eye(len(labels), dtype=torch.bool, device=vectors.device)
    positive = labels[:, None].eq(labels[None, :]) & ~diagonal
    if not positive.any(dim=1).all() or labels.unique().numel() < 2:
        raise ValueError("每个结构至少需要两张谱，且批次至少包含两个结构。")
    vectors = F.normalize(vectors, dim=1)
    logits = vectors @ vectors.T / temperature
    denominator = torch.logsumexp(logits.masked_fill(diagonal, -torch.inf), dim=1)
    log_probabilities = logits - denominator[:, None]
    return -(log_probabilities.masked_fill(~positive, 0).sum(dim=1)
             / positive.sum(dim=1)).mean()
