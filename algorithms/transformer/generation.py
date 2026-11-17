"""质谱条件自回归 SMILES Decoder；首版冻结 Encoder，保护检索表示。"""

import math
import torch
from torch import nn
from rdkit import Chem, rdBase

from algorithms.transformer.model import SpectrumEncoder, SpectrumEncoderConfig


def canonical_smiles(text):
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(text)
    return Chem.MolToSmiles(mol) if mol is not None and mol.GetNumAtoms() else None


class SmilesTokenizer:
    """可逆字符分词；词表只从训练标签构建，不截断超长 SMILES。"""
    pad, bos, eos = 0, 1, 2

    def __init__(self, characters):
        self.characters = sorted(set(characters))
        self.lookup = {char: i + 3 for i, char in enumerate(self.characters)}

    def __len__(self):
        return len(self.characters) + 3

    def encode(self, text):
        return [self.bos] + [self.lookup[char] for char in text] + [self.eos]

    def decode(self, ids):
        chars = []
        for token in ids:
            if token == self.eos:
                break
            if token >= 3:
                chars.append(self.characters[token - 3])
        return "".join(chars)


class SpectrumGenerator(nn.Module):
    def __init__(self, encoder, tokenizer, num_layers=2, max_length=256):
        super().__init__()
        if num_layers < 1 or max_length < 3:
            raise ValueError("Decoder 层数需为正，max_length 至少为 3。")
        self.encoder, self.tokenizer = encoder, tokenizer
        self.max_length, self.num_layers = max_length, num_layers
        c = encoder.config
        self.token_embedding = nn.Embedding(len(tokenizer), c.d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, c.d_model)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(c.d_model, c.nhead, c.dim_feedforward,
                                       c.dropout, activation="gelu", batch_first=True, norm_first=True)
            for _ in range(num_layers)])
        self.norm = nn.LayerNorm(c.d_model)
        self.output = nn.Linear(c.d_model, len(tokenizer))
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()  # 冻结后也关闭 Encoder 的 Dropout。
        return self

    def memory(self, spectra):
        with torch.no_grad():
            return self.encoder.encode_tokens(spectra)

    def decode_logits(self, tokens, memory, memory_padding):
        length = tokens.shape[1]
        if length > self.max_length:
            raise ValueError("SMILES 超过配置最大长度。")
        positions = torch.arange(length, device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        causal = torch.ones(length, length, device=tokens.device, dtype=torch.bool).triu(1)
        for layer in self.layers:
            hidden = layer(hidden, memory, tgt_mask=causal, tgt_key_padding_mask=tokens.eq(0),
                           memory_key_padding_mask=memory_padding)
        return self.output(self.norm(hidden))

    def forward(self, spectra, tokens):
        return self.decode_logits(tokens, self.memory(spectra), spectra.eq(0))

    @torch.no_grad()
    def generate(self, spectra, beam_size=5, top_k=25, max_new_tokens=None):
        """单谱束搜索，仅返回 EOS 完成且 RDKit 可解析的候选；可返回空列表。"""
        if spectra.shape[0] != 1 or beam_size < 1 or top_k < 1:
            raise ValueError("生成要求单张谱，beam_size/top_k 为正。")
        if max_new_tokens is not None and max_new_tokens < 1:
            raise ValueError("max_new_tokens 必须为正。")
        self.eval()
        memory, padding = self.memory(spectra), spectra.eq(0)
        beams = [([self.tokenizer.bos], 0.0)]
        completed = {}
        limit = min(self.max_length - 1, max_new_tokens or self.max_length - 1)
        for _ in range(limit):
            if not beams:
                break
            tokens = torch.tensor([ids for ids, _ in beams], device=spectra.device)
            logits = self.decode_logits(tokens, memory.expand(len(beams), -1, -1),
                                        padding.expand(len(beams), -1))[:, -1]
            logits[:, :2] = -torch.inf
            log_probs = logits.log_softmax(dim=-1)
            choices = []
            for i, (ids, score) in enumerate(beams):
                values, indices = log_probs[i].topk(min(beam_size, len(self.tokenizer) - 2))
                for value, index in zip(values.tolist(), indices.tolist()):
                    new_ids, new_score = ids + [index], score + value
                    if index == self.tokenizer.eos:
                        text = canonical_smiles(self.tokenizer.decode(new_ids))
                        if text:
                            normalized = new_score / (len(new_ids) - 1)
                            completed[text] = max(completed.get(text, -math.inf), normalized)
                    else:
                        choices.append((new_ids, new_score))
            beams = sorted(choices, key=lambda item: item[1], reverse=True)[:beam_size]
        return sorted(completed.items(), key=lambda item: (-item[1], item[0]))[:top_k]


def load_generator(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    encoder = SpectrumEncoder(SpectrumEncoderConfig(**checkpoint["model_config"]))
    tokenizer = SmilesTokenizer(checkpoint["characters"])
    model = SpectrumGenerator(encoder, tokenizer, **checkpoint["decoder_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
