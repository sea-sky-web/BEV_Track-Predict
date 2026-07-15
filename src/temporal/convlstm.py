"""ConvLSTM cell and spatiotemporal field prediction model.

Architecture (from Module 2 plan Section 9.2):
    Input:  (B, T_hist=4, C_in=5, H=120, W=360)
    Encoder: 2-layer ConvLSTM, hidden=32, kernel=3
    Decoder: autoregressive, shared ConvLSTM layer + output conv
    Output: (B, T_future=4, 3, H, W) — [occ_logit, vx, vy] per step
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.gates = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = x.shape
        if state is None:
            device = x.device
            dtype = x.dtype
            hx = torch.zeros(b, self.hidden_channels, h, w, device=device, dtype=dtype)
            cx = torch.zeros(b, self.hidden_channels, h, w, device=device, dtype=dtype)
        else:
            hx, cx = state

        combined = torch.cat([x, hx], dim=1)
        gates = self.gates(combined)
        i, f, o, g = gates.chunk(4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        cy = f * cx + i * g
        hy = o * torch.tanh(cy)
        return hy, cy


class SpatioTemporalPredictor(nn.Module):

    def __init__(
        self,
        in_channels: int = 5,
        hidden_channels: int = 32,
        kernel_size: int = 3,
        n_encoder_layers: int = 2,
        n_future: int = 4,
        out_channels: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.n_future = n_future
        self.out_channels = out_channels

        self.encoder_cells = nn.ModuleList()
        for i in range(n_encoder_layers):
            c_in = in_channels if i == 0 else hidden_channels
            self.encoder_cells.append(ConvLSTMCell(c_in, hidden_channels, kernel_size))

        self.decoder_cell = ConvLSTMCell(out_channels, hidden_channels, kernel_size)

        self.output_conv = nn.Conv2d(hidden_channels, out_channels, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T_hist, C_in, H, W)

        Returns:
            (B, T_future, out_channels, H, W) — raw logits for occupancy, vx, vy
        """
        b, t_hist, c, h, w = x.shape

        encoder_states: list[tuple[torch.Tensor, torch.Tensor] | None] = [
            None for _ in self.encoder_cells
        ]

        for t in range(t_hist):
            inp = x[:, t]
            for i, cell in enumerate(self.encoder_cells):
                inp, c_state = cell(inp, encoder_states[i])
                encoder_states[i] = (inp, c_state)

        decoder_state = encoder_states[-1]

        outputs = []
        decoder_input = torch.zeros(b, self.out_channels, h, w, device=x.device, dtype=x.dtype)

        for _ in range(self.n_future):
            h_dec, c_dec = self.decoder_cell(decoder_input, decoder_state)
            decoder_state = (h_dec, c_dec)

            out = self.output_conv(h_dec)
            outputs.append(out)
            decoder_input = out

        return torch.stack(outputs, dim=1)
