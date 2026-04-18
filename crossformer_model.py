"""
Crossformer model for multivariate macro sequence encoding.
Adapted from official implementation: https://github.com/Thinklab-SJTU/Crossformer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]


class DSWEmbedding(nn.Module):
    """Dimension-Segment-Wise Embedding for Crossformer."""
    def __init__(self, seg_len, d_model):
        super().__init__()
        self.seg_len = seg_len
        self.d_model = d_model
        self.linear = nn.Linear(seg_len, d_model)

    def forward(self, x):
        # x: [batch, seq_len, n_vars]
        batch, seq_len, n_vars = x.shape
        pad_len = (self.seg_len - seq_len % self.seg_len) % self.seg_len
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        x = x.view(batch, -1, self.seg_len, n_vars)
        x = x.permute(0, 3, 1, 2)  # [batch, n_vars, n_segments, seg_len]
        x = self.linear(x)         # [batch, n_vars, n_segments, d_model]
        return x


class TwoStageAttentionLayer(nn.Module):
    """Cross-Time and Cross-Dimension attention with router mechanism."""
    def __init__(self, d_model, n_heads, dropout=0.1, router_factor=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.router_factor = router_factor

        # Cross-Time Attention
        self.time_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # Cross-Dimension Attention with Routers
        self.router_dim = max(1, d_model // router_factor)
        self.router = nn.Parameter(torch.randn(1, self.router_dim, d_model))

    def forward(self, x):
        # x: [batch, n_vars, n_segments, d_model]
        batch, n_vars, n_segments, d_model = x.shape

        # Cross-Time Stage: attend over segments within each variable
        x_time = x.permute(0, 1, 3, 2).reshape(batch * n_vars, n_segments, d_model)
        x_time, _ = self.time_attn(x_time, x_time, x_time)
        x = x_time.reshape(batch, n_vars, d_model, n_segments).permute(0, 1, 3, 2)

        # Cross-Dimension Stage: use routers to reduce complexity
        x_dim = x.permute(0, 2, 1, 3).reshape(batch * n_segments, n_vars, d_model)
        routers = self.router.expand(batch * n_segments, -1, -1)
        router_out, _ = self.time_attn(routers, x_dim, x_dim)
        x_dim, _ = self.time_attn(x_dim, router_out, router_out)
        x = x_dim.reshape(batch, n_segments, n_vars, d_model).permute(0, 2, 1, 3)

        return x


class CrossformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, router_factor=4):
        super().__init__()
        self.attn = TwoStageAttentionLayer(d_model, n_heads, dropout, router_factor)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ffn(x))
        return x


class CrossformerEncoder(nn.Module):
    def __init__(self, seg_len, d_model, n_heads, n_layers, dropout=0.1):
        super().__init__()
        self.dsw = DSWEmbedding(seg_len, d_model)
        self.layers = nn.ModuleList([
            CrossformerEncoderLayer(d_model, n_heads, dropout) for _ in range(n_layers)
        ])

    def forward(self, x):
        x = self.dsw(x)
        for layer in self.layers:
            x = layer(x)
        return x


class CrossformerETF(nn.Module):
    """Crossformer for ETF return prediction from macro sequences."""
    def __init__(self, n_vars, seg_len, d_model, n_heads, n_layers, n_etfs, dropout=0.1):
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model
        self.encoder = CrossformerEncoder(seg_len, d_model, n_heads, n_layers, dropout)
        # Predictor input dimension is d_model
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_etfs)
        )

    def forward(self, x):
        # x: [batch, seq_len, n_vars]
        enc = self.encoder(x)              # [batch, n_vars, n_segments, d_model]
        # Average over n_segments dimension (dim=2)
        pooled = enc.mean(dim=2)           # [batch, n_vars, d_model]
        # Average over n_vars dimension (dim=1)
        pooled = pooled.mean(dim=1)        # [batch, d_model]
        pred = self.predictor(pooled)      # [batch, n_etfs]
        return pred
