from __future__ import annotations
import math, torch, torch.nn as nn


#ProbSparse Self‑Attention、
#Input/Output: x (B, L, D)
#Complexity O(L log L) — Informer sampling strategy (top‑k ≈ sqrt(L))
class ProbSparseSelfAttn(nn.Module):

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.d = n_heads, d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out    = nn.Linear(d_model, d_model)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):                       # (B,L,D)
        B, L, _ = x.shape
        k_top = int(math.ceil(math.sqrt(L)))
        k_top = min(k_top, L)
        Q = self.q_proj(x).view(B, L, self.h, self.d).transpose(1, 2)  # (B,h,L,d)
        K = self.k_proj(x).view(B, L, self.h, self.d).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.h, self.d).transpose(1, 2)

        # Top‑k queries
        q_norm = Q.norm(dim=-1)                                   # (B,h,L)
        idx = q_norm.topk(k_top, dim=-1).indices                  # (B,h,k)
        Q_k = torch.gather(Q, 2, idx.unsqueeze(-1).expand(-1, -1, -1, self.d))

        attn = torch.softmax((Q_k @ K.transpose(-2, -1)) / math.sqrt(self.d), dim=-1)
        attn = self.drop(attn)
        ctx  = attn @ V                                           # (B,h,k,d)

        ctx_full = torch.zeros_like(Q)
        ctx_full.scatter_(2, idx.unsqueeze(-1).expand_as(ctx), ctx)

        return self.out(ctx_full.transpose(1, 2).reshape(B, L, -1))



#ProbSparse Cross‑Attention
#      Input: Q_in  (B, L_q, D):tgt/query, KV_in (B, L_kv, D):mem/key‑value
#      Output:same shape as Q_in
class ProbSparseCrossAttn(nn.Module):

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.d = n_heads, d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out    = nn.Linear(d_model, d_model)
        self.drop   = nn.Dropout(dropout)

    def forward(self, Q_in: torch.Tensor, KV_in: torch.Tensor):
        B, Lq, _ = Q_in.shape
        _, Lkv, _ = KV_in.shape
        k_top = int(math.ceil(math.sqrt(Lkv)))
        k_top = min(k_top, Lq)

        Q = self.q_proj(Q_in).view(B, Lq,  self.h, self.d).transpose(1, 2)
        K = self.k_proj(KV_in).view(B, Lkv, self.h, self.d).transpose(1, 2)
        V = self.v_proj(KV_in).view(B, Lkv, self.h, self.d).transpose(1, 2)

        q_norm = Q.norm(dim=-1)                                  # (B,h,Lq)
        idx = q_norm.topk(k_top, dim=-1).indices                 # (B,h,k)
        Q_k = torch.gather(Q, 2, idx.unsqueeze(-1).expand(-1, -1, -1, self.d))

        attn = torch.softmax((Q_k @ K.transpose(-2, -1)) / math.sqrt(self.d), dim=-1)
        attn = self.drop(attn)
        ctx  = attn @ V                                          # (B,h,k,d)

        ctx_full = torch.zeros_like(Q)
        ctx_full.scatter_(2, idx.unsqueeze(-1).expand_as(ctx), ctx)

        return self.out(ctx_full.transpose(1, 2).reshape(B, Lq, -1))


#Encoder / Decoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.attn  = ProbSparseSelfAttn(d_model, n_heads, dropout)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.self_attn  = ProbSparseSelfAttn(d_model, n_heads, dropout)
        self.cross_attn = ProbSparseCrossAttn(d_model, n_heads, dropout)
        self.ff         = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])

    def forward(self, tgt, mem):
        tgt = tgt + self.self_attn(self.norms[0](tgt))
        tgt = tgt + self.cross_attn(self.norms[1](tgt), mem)
        tgt = tgt + self.ff(self.norms[2](tgt))
        return tgt


#InformerDecoder
class InformerDecoder(nn.Module):
    def __init__(self, d_model, n_heads=4, layers=1, d_ff=256, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, mem):
        for layer in self.layers:
            tgt = layer(tgt, mem)
        return self.norm(tgt)


#Informer main body
#Input: (B, T_in, C, H, W)
#Output: (B, T_out, 1, H, W) —— First flatten (C, H, W) to F=C*H*W, then perform attention in the temporal dimension
class Informer(nn.Module):

    def __init__(
        self,
        C: int, H: int, W: int,
        T_out: int = 3,
        d_model: int = 128,
        n_heads: int = 4,
        e_layers: int = 2,
        d_layers: int = 1,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.H, self.W, self.T_out = H, W, T_out
        in_dim = C * H * W

        self.enc_embed = nn.Sequential(nn.Linear(in_dim, d_model), nn.Dropout(dropout))
        self.dec_embed = nn.Sequential(nn.Linear(in_dim, d_model), nn.Dropout(dropout))

        self.encoder = nn.Sequential(
            *[EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)]
        )
        self.decoder = InformerDecoder(
            d_model=d_model,
            n_heads=n_heads,
            layers=d_layers,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.proj = nn.Linear(d_model, H * W)

    def forward(self, x: torch.Tensor):
        # x: (B,T_in,C,H,W)
        B, T_in, C, H, W = x.shape
        x_flat = x.reshape(B, T_in, -1)                    # (B,T_in,F)
        enc_out = self.encoder(self.enc_embed(x_flat))     # (B,T_in,D)

        #Decoder input: Copy T_out times in the final step
        dec_in = x_flat[:, -1:, :].repeat(1, self.T_out, 1)  # (B,T_out,F)
        dec_out = self.decoder(self.dec_embed(dec_in), enc_out)

        y = self.proj(dec_out).view(B, self.T_out, 1, H, W)
        return y
