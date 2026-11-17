"""共享的双向谱编码器；查询谱和参考谱调用同一个模型实例。"""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from algorithms.transformer.dataset import validate_preprocessing


@dataclass(frozen=True)
class SpectrumEncoderConfig:
    bin_width: float = 1.0
    max_mz: float = 1500.0
    power: float = 0.5
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 512
    dropout: float = 0.1
    embedding_dim: int = 128

    def __post_init__(self):
        validate_preprocessing(self.bin_width, self.max_mz, self.power)
        for name in ("d_model", "nhead", "num_layers", "dim_feedforward", "embedding_dim"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} 必须为正整数。")
        if self.d_model % self.nhead:
            raise ValueError("d_model 必须能被 nhead 整除。")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout 必须在 [0, 1) 内。")

    @property
    def n_bins(self):
        return math.ceil(self.max_mz / self.bin_width)


class BinnedSpectrumEmbedding(nn.Module):
    """强度线性映射 + 以质量分箱中心为坐标的固定正余弦编码。"""

    def __init__(self, config):
        super().__init__()
        self.n_bins = config.n_bins
        self.intensity_projection = nn.Linear(1, config.d_model)
        positions = (torch.arange(config.n_bins, dtype=torch.float32) + 0.5) * config.bin_width
        frequencies = torch.exp(torch.arange(0, config.d_model, 2) * (-math.log(10000) / config.d_model))
        angles = positions[:, None] * frequencies[None, :]
        encoding = torch.zeros(config.n_bins, config.d_model)
        encoding[:, 0::2] = torch.sin(angles)
        encoding[:, 1::2] = torch.cos(angles[:, :config.d_model // 2])
        self.register_buffer("mass_encoding", encoding.unsqueeze(0))

    def forward(self, spectra):
        if spectra.ndim != 2 or spectra.shape[0] == 0 or spectra.shape[1] != self.n_bins:
            raise ValueError(f"输入必须为非空 [batch, {self.n_bins}] 张量。")
        if not spectra.is_floating_point():
            raise ValueError("输入必须为浮点强度。")
        if not torch.isfinite(spectra).all() or (spectra < 0).any():
            raise ValueError("输入必须有限且非负。")
        if not (spectra > 0).any(dim=1).all():
            raise ValueError("每张谱至少需要一个非零分箱。")
        spectra = spectra.to(dtype=self.intensity_projection.weight.dtype)
        return self.intensity_projection(spectra.unsqueeze(-1)) + self.mass_encoding


class SpectrumEncoder(nn.Module):
    """双向注意力、掩码均值池化、单位范数输出。

    空分箱不能作为 attention 的 key/value，池化也忽略它们。
    仍使用稠密序列，注意力计算量随分箱数平方增长。
    标签不进入网络；孪生训练共享此编码器的全部权重。
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config or SpectrumEncoderConfig()
        c = self.config
        self.input_embedding = BinnedSpectrumEmbedding(c)
        # 各层独立初始化，避免复制同一层的初始权重。
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(c.d_model, c.nhead, c.dim_feedforward,
                                       c.dropout, activation="gelu", batch_first=True,
                                       norm_first=True)
            for _ in range(c.num_layers)
        ])
        self.norm = nn.LayerNorm(c.d_model)
        self.output_projection = nn.Linear(c.d_model, c.embedding_dim)

    def encode_tokens(self, spectra):
        tokens = self.input_embedding(spectra)
        empty = spectra.eq(0)
        for layer in self.layers:
            tokens = layer(tokens, src_key_padding_mask=empty, is_causal=False)
        return self.norm(tokens)

    def forward(self, spectra):
        tokens = self.encode_tokens(spectra)
        valid = spectra.gt(0).unsqueeze(-1)
        pooled = tokens.masked_fill(~valid, 0).sum(dim=1) / valid.sum(dim=1)
        return F.normalize(self.output_projection(pooled), p=2, dim=-1)
