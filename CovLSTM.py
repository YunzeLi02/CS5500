from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn


#ConvLSTM Block
class ConvLSTMCell(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 kernel_size: int | Tuple[int, int] = 3):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        pad = kernel_size[0] // 2, kernel_size[1] // 2

        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=pad,
        )
        self.hidden_channels = hidden_channels    # Set the number of hidden layer channels

    def forward(self, x: torch.Tensor,
                hc: Tuple[torch.Tensor, torch.Tensor]):
        h, c = hc                                     # (B, H_ch, H, W)
        gates = self.conv(torch.cat([x, h], dim=1))   # concat X&H
        i, f, o, g = torch.chunk(gates, 4, dim=1)     # Concatenate input and hidden state and perform convolution
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)   ## Calculate the sigmoid values for the input gate, forget gate, and output gate
        g = torch.tanh(g)       # Compute the tanh value of the candidate state
        c = f * c + i * g       # Update the unit state
        h = o * torch.tanh(c)    # Update the hidden state
        return h, c              # Return the new hidden state and unit state


#Encapsulation Network
class ConvLSTM(nn.Module):
    """
    Encoder 3×3 Conv → ConvLSTMCell → Decoder 1×1 Conv → repeat T_out
    """
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 64,
                 out_channels: int = 1,
                 out_len: int = 6,
                 kernel_size: int | Tuple[int, int] = 3):
        super().__init__()
        self.out_len = out_len                                  # Set Output Sequence Length
        self.encoder = nn.Conv2d(in_channels,
                                 hidden_channels,
                                 kernel_size=3,
                                 padding=1)           # Define Encoder Convolution Layer
        self.cell = ConvLSTMCell(hidden_channels,
                                 hidden_channels,
                                 kernel_size)        # Define  ConvLSTM Unit
        self.decoder = nn.Conv2d(hidden_channels,
                                 out_channels,
                                 kernel_size=1)        # Define Decoder Convolution Layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T_in, C, H, W = x.shape                            # x: (B,T_in,C,H,W)
        device = x.device
        h = torch.zeros(B, self.cell.hidden_channels, H, W, device=device)
        c = torch.zeros_like(h)

        for t in range(T_in):
            inp = self.encoder(x[:, t])               # Encode the input for each time step
            h, c = self.cell(inp, (h, c))             # Update the hidden state and unit state through the ConvLSTM unit

        y = self.decoder(h).unsqueeze(1)              # (B,1,H,W) -> (B,1,1,H,W)
        y = y.repeat(1, self.out_len, 1, 1, 1)        # broadcast T_out
        return y
