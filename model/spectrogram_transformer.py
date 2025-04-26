import torch
import torch.nn as nn
import math

class PatchEmbed(nn.Module):
    def __init__(self, in_channels=1, patch_size=10, emb_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):  # [B, 1, 80, T]
        x = self.proj(x)  # [B, emb_dim, H_patch, W_patch]
        x = x.flatten(2)  # [B, emb_dim, N_patches]
        x = x.transpose(1, 2)  # [B, N_patches, emb_dim]
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2) * -(math.log(10000.0) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, emb_dim]
        self.register_buffer('pe', pe)

    def forward(self, x):  # [B, N, emb_dim]
        x = x + self.pe[:, :x.size(1)]
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, emb_dim=256, depth=6, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=emb_dim,
                nhead=num_heads,
                dim_feedforward=int(emb_dim * mlp_ratio),
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ) for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SpectrogramTransformer(nn.Module):
    def __init__(self, num_classes=10, patch_size=10, emb_dim=256, depth=6, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(patch_size=patch_size, emb_dim=emb_dim)
        self.pos_encoder = PositionalEncoding(emb_dim)
        self.transformer = TransformerEncoder(emb_dim, depth, num_heads, mlp_ratio, dropout)
        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, num_classes)

    def forward(self, x):  # [B, 1, 80, T]
        x = self.patch_embed(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # global average pooling
        x = self.head(x)
        return x
