from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .kv_cache import KVCacheEntry


class RWKV7TimeMixForTabICL(nn.Module):
    """Minimal RWKV-7 style TimeMix adapted to TabICL's [B, N, D] tensors.

    This module intentionally keeps only the time-mixing ideas needed for a first
    integration pass:
    - previous-token shift with learned channel-wise mixing
    - recurrent per-channel state update
    - a support/query mode compatible with TabICL's train_size masking

    It is shape-compatible with the original attention path, but not semantically
    equivalent to the original support/query attention.
    """

    def __init__(self, d_model: int, nhead: int) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.d_model = d_model
        self.nhead = nhead

        ddd = torch.linspace(0.0, 1.0, d_model).view(1, 1, d_model)
        self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.20))
        self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.90))
        self.x_k = nn.Parameter(1.0 - torch.pow(ddd, 0.70))
        self.x_v = nn.Parameter(1.0 - torch.pow(ddd, 0.70))
        self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.90))
        self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.20))

        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.time_decay = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.in_context = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model)
        self.ln_x = nn.GroupNorm(nhead, d_model, eps=64e-5)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in (
            self.receptance,
            self.time_decay,
            self.key,
            self.value,
            self.in_context,
            self.gate,
        ):
            nn.init.xavier_uniform_(layer.weight)

        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor | int] = None,
        cached_state: Optional[KVCacheEntry] = None,
        need_state: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        if x.dim() != 3:
            raise ValueError(f"RWKV7TimeMixForTabICL expects [B, N, D], but got {tuple(x.shape)}")
        if x.size(-1) != self.d_model:
            raise ValueError(f"Last dim {x.size(-1)} does not match d_model {self.d_model}")

        padding_mask = None
        if mask is not None and not isinstance(mask, int):
            padding_mask = self._normalize_padding_mask(mask, x)

        if cached_state is not None:
            support_state, last_token = self._unpack_cached_state(cached_state, x)
            out, final_state, final_token = self._forward_query_only(x, support_state, last_token, padding_mask)
        elif isinstance(mask, int):
            out, final_state, final_token = self._forward_support_query(x, mask)
        else:
            out, final_state, final_token = self._scan_sequence(x, padding_mask=padding_mask)

        if need_state:
            return out, final_state, final_token
        return out

    def _normalize_padding_mask(self, mask: Tensor, x: Tensor) -> Tensor:
        if mask.shape != x.shape[:-1]:
            raise ValueError(f"Expected padding mask shape {x.shape[:-1]}, but got {tuple(mask.shape)}")
        return mask.to(dtype=torch.bool, device=x.device)

    def _unpack_cached_state(self, cached_state: KVCacheEntry, x: Tensor) -> tuple[Tensor, Tensor]:
        if not cached_state.is_valid():
            raise ValueError("RWKV7TimeMixForTabICL requires a populated cached_state")

        support_state = cached_state.key.to(device=x.device, dtype=x.dtype)
        last_token = cached_state.value.to(device=x.device, dtype=x.dtype)
        expected = (x.size(0), self.d_model)
        if support_state.shape != expected or last_token.shape != expected:
            raise ValueError(
                f"Expected cached state tensors of shape {expected}, got "
                f"{tuple(support_state.shape)} and {tuple(last_token.shape)}"
            )
        return support_state, last_token

    def _shift_delta(self, x: Tensor, prev_token: Optional[Tensor] = None) -> Tensor:
        if x.size(1) == 0:
            return torch.empty_like(x)

        if prev_token is None:
            prev = torch.zeros_like(x)
            prev[:, 1:, :] = x[:, :-1, :]
        else:
            prev = prev_token.unsqueeze(1).expand(-1, x.size(1), -1)

        return prev - x

    def _project(self, x: Tensor, prev_token: Optional[Tensor] = None) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        xx = self._shift_delta(x, prev_token=prev_token)

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = torch.sigmoid(self.receptance(xr))
        decay = torch.exp(-F.softplus(self.time_decay(xw)))
        k = torch.tanh(self.key(xk))
        v = self.value(xv)
        alpha = torch.sigmoid(self.in_context(xa))
        g = torch.sigmoid(self.gate(xg))
        return r, decay, k, v, alpha, g

    def _finalize(self, mixed: Tensor, gate: Tensor) -> Tensor:
        if mixed.numel() == 0:
            return mixed

        bsz, seq_len, _ = mixed.shape
        mixed = self.ln_x(mixed.reshape(bsz * seq_len, self.d_model)).view(bsz, seq_len, self.d_model)
        return self.output(mixed * gate)

    def _scan_sequence(
        self,
        x: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        bsz, seq_len, _ = x.shape
        state = x.new_zeros(bsz, self.d_model)
        outputs = []

        r, decay, k, v, alpha, g = self._project(x)
        valid_tokens = None if padding_mask is None else ~padding_mask

        for idx in range(seq_len):
            updated_state = state * decay[:, idx, :] + alpha[:, idx, :] * k[:, idx, :] * v[:, idx, :]
            if valid_tokens is None:
                state = updated_state
                outputs.append(r[:, idx, :] * state)
            else:
                valid = valid_tokens[:, idx].unsqueeze(-1)
                state = torch.where(valid, updated_state, state)
                outputs.append(torch.where(valid, r[:, idx, :] * state, torch.zeros_like(state)))

        mixed = x.new_empty(bsz, 0, self.d_model) if seq_len == 0 else torch.stack(outputs, dim=1)
        out = self._finalize(mixed, g)
        if padding_mask is not None and seq_len > 0:
            out = out.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        last_token = x.new_zeros(bsz, self.d_model) if seq_len == 0 else x[:, seq_len - 1, :]
        return out, state, last_token

    def _forward_query_only(
        self,
        x: Tensor,
        support_state: Tensor,
        last_support_token: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        _, seq_len, _ = x.shape
        if seq_len == 0:
            return x, support_state, last_support_token

        r, decay, _, _, _, g = self._project(x, prev_token=last_support_token)
        mixed = r * (decay * support_state.unsqueeze(1))
        out = self._finalize(mixed, g)
        if padding_mask is not None:
            out = out.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return out, support_state, last_support_token

    def _forward_support_query(self, x: Tensor, train_size: int) -> tuple[Tensor, Tensor, Tensor]:
        bsz, seq_len, _ = x.shape
        train_size = max(0, min(int(train_size), seq_len))

        support = x[:, :train_size, :]
        query = x[:, train_size:, :]

        if train_size > 0:
            support_out, support_state, last_support_token = self._scan_sequence(support)
        else:
            support_out = x.new_empty(bsz, 0, self.d_model)
            support_state = x.new_zeros(bsz, self.d_model)
            last_support_token = x.new_zeros(bsz, self.d_model)

        if query.size(1) == 0:
            return support_out, support_state, last_support_token

        query_out, _, _ = self._forward_query_only(query, support_state, last_support_token)
        return torch.cat([support_out, query_out], dim=1), support_state, last_support_token
