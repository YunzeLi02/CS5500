__all__ = ['Transpose', 'get_activation_fn', 'moving_avg', 'series_decomp',
           'PositionalEncoding', 'SinCosPosEncoding', 'Coord2dPosEncoding',
           'Coord1dPosEncoding', 'positional_encoding']

import torch
from torch import nn
import math

# ==== UPDATE: 轻量打印工具，避免 pv 未定义报错 ====
def pv(msg, verbose: bool):
    if verbose:
        print(msg)
# ==================================================


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


def get_activation_fn(activation):
    if callable(activation): return activation()
    elif activation.lower() == "relu": return nn.ReLU()
    elif activation.lower() == "gelu": return nn.GELU()
    # ==== UPDATE: 友好报错 ====
    raise ValueError(f'{activation} is not available. Use "relu", "gelu", or a callable')


# ---------- decomposition ----------

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        # ==== UPDATE: 保证对称核；偶数核自动+1 ====
        ks = int(kernel_size)
        if ks < 1:
            ks = 1
        if ks % 2 == 0:
            ks += 1
        self.kernel_size = ks
        self.stride = int(stride)
        # 使用 AvgPool1d，但配合 ReplicationPad1d，边界更稳
        self.pad = nn.ReplicationPad1d(( (self.kernel_size - 1)//2,
                                         (self.kernel_size - 1)//2 ))   # ==== UPDATE ====
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=self.stride, padding=0)

    def forward(self, x):
        # x: [B, T, C]
        # ==== UPDATE: 显式“same”式 padding，避免前后拼接导致的 shape/grad 小问题 ====
        x = x.permute(0, 2, 1)        # [B, C, T]
        x = self.pad(x)               # pad on time dimension
        x = self.avg(x)               # [B, C, T']
        x = x.permute(0, 2, 1)        # [B, T', C]
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        # x: [B, T, C]
        moving_mean = self.moving_avg(x)
        # ==== UPDATE: 与输入对齐时序长度（AvgPool stride=1 已对齐；此处防御式裁剪） ====
        if moving_mean.size(1) != x.size(1):                                    # ==== UPDATE ====
            T = min(moving_mean.size(1), x.size(1))                              # ==== UPDATE ====
            moving_mean = moving_mean[:, :T, :]                                  # ==== UPDATE ====
            x = x[:, :T, :]                                                      # ==== UPDATE ====
        res = x - moving_mean
        return res, moving_mean


# ---------- pos_encoding ----------

def PositionalEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model, dtype=torch.float32)                        # ==== UPDATE ====
    position = torch.arange(0, q_len, dtype=torch.float32).unsqueeze(1)          # ==== UPDATE ====
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                         -(math.log(10000.0) / d_model))                          # ==== UPDATE ====
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        # ==== UPDATE: 数值稳定的归一化 ====
        pe = pe - pe.mean()
        pe_std = pe.std()
        pe = pe / (pe_std * 10 + 1e-8)
    return pe

SinCosPosEncoding = PositionalEncoding


def Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True, eps=1e-3, verbose=False):
    # 2D 坐标式位置编码，带简单二分式“零均值”调参
    x = .5 if exponential else 1
    i = 0
    for i in range(100):
        cpe = 2 * (torch.linspace(0, 1, q_len, dtype=torch.float32).reshape(-1, 1) ** x) * \
                 (torch.linspace(0, 1, d_model, dtype=torch.float32).reshape(1, -1) ** x) - 1
        pv(f'{i:4.0f}  {x:5.3f}  {cpe.mean():+6.3f}', verbose)
        if abs(cpe.mean()) <= eps: break
        elif cpe.mean() > eps: x += .001
        else: x -= .001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10 + 1e-8)                                       # ==== UPDATE ====
    return cpe


def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    cpe = (2 * (torch.linspace(0, 1, q_len, dtype=torch.float32).reshape(-1, 1)  # ==== UPDATE ====
              ** (.5 if exponential else 1)) - 1)
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10 + 1e-8)                                       # ==== UPDATE ====
    return cpe


def positional_encoding(pe, learn_pe, q_len, d_model):
    """
    构造可学习/不可学习的位置编码参数。
    说明：
      - 与原库保持一致：'zeros' / 'zero' 默认初始化为小的随机值（不是全 0），利于训练；
      - 返回 nn.Parameter，随模型移动到相应 device。
    """
    # Positional encoding
    if pe is None:
        W_pos = torch.empty((q_len, d_model), dtype=torch.float32)                # ==== UPDATE ====
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1), dtype=torch.float32)                      # ==== UPDATE ====
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model), dtype=torch.float32)                # ==== UPDATE ====
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1), dtype=torch.float32)                      # ==== UPDATE ====
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1), dtype=torch.float32)                      # ==== UPDATE ====
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'lin1d':
        W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == 'exp1d':
        W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == 'lin2d':
        W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == 'exp2d':
        W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == 'sincos':
        W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else:
        raise ValueError(
            f"{pe} is not a valid pe. Available: 'gauss'/'normal', 'zeros', 'zero', 'uniform', "
            f"'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None."
        )
    return nn.Parameter(W_pos, requires_grad=learn_pe)
