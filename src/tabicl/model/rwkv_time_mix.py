from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .kv_cache import KVCacheEntry
from .rwkv7_kernel_backend import generalized_delta_rule


class RWKV7TimeMixForTabICL(nn.Module):
    """RWKV7-style time mix for TabICL's [B, T, D] ICL encoder path.

    The cached state stores:
    - `key`: the per-head delta-rule state [B, H, K, K]
    - `value`: the last support token [B, D]

    Query tokens reuse the cached state but are evaluated with an all-zero update mask,
    so they can read support context without mutating the cache.
    """

    def __init__(self, d_model: int, nhead: int) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.last_backend = "torch"

        ddd = torch.linspace(0.0, 1.0, d_model).view(1, 1, d_model)
        self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.20))
        self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.90))
        self.x_k = nn.Parameter(1.0 - torch.pow(ddd, 0.70))
        self.x_v = nn.Parameter(1.0 - torch.pow(ddd, 0.70))
        self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.90))
        self.x_b = nn.Parameter(1.0 - torch.pow(ddd, 0.90))
        self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.20))

        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.time_decay = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.time_a = nn.Linear(d_model, d_model, bias=False)
        self.time_b = nn.Linear(d_model, d_model, bias=False)
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
            self.time_a,
            self.time_b,
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
            raise ValueError(f"RWKV7TimeMixForTabICL expects [B, T, D], but got {tuple(x.shape)}")
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

        support_state = cached_state.key.to(device=x.device, dtype=torch.float32)
        last_token = cached_state.value.to(device=x.device, dtype=x.dtype)
        expected_state = (x.size(0), self.nhead, self.head_dim, self.head_dim)
        expected_token = (x.size(0), self.d_model)
        if support_state.shape != expected_state or last_token.shape != expected_token:
            raise ValueError(
                f"Expected cached state shapes {expected_state} and {expected_token}, got "
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

    def _project(
        self,
        x: Tensor,
        prev_token: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        xx = self._shift_delta(x, prev_token=prev_token)

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xb = x + xx * self.x_b
        xg = x + xx * self.x_g

        r = self._reshape_heads(self.receptance(xr))
        w = self._reshape_heads(-F.softplus(self.time_decay(xw)))
        k = self._reshape_heads(torch.tanh(self.key(xk)))
        v = self._reshape_heads(self.value(xv))
        a = F.normalize(self._reshape_heads(self.time_a(xa)), dim=-1, eps=1e-6)
        b = F.normalize(self._reshape_heads(self.time_b(xb)), dim=-1, eps=1e-6)
        g = torch.sigmoid(self.gate(xg))
        return r, w, k, v, a, b, g

    def _reshape_heads(self, tensor: Tensor) -> Tensor:
        batch_size, seq_len, _ = tensor.shape
        return tensor.view(batch_size, seq_len, self.nhead, self.head_dim)

    def _mask_projections(
        self,
        r: Tensor,
        w: Tensor,
        k: Tensor,
        v: Tensor,
        a: Tensor,
        b: Tensor,
        g: Tensor,
        valid_tokens: Optional[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        if valid_tokens is None:
            return r, w, k, v, a, b, g

        token_mask = valid_tokens.unsqueeze(-1).unsqueeze(-1).to(r.dtype)
        gate_mask = valid_tokens.unsqueeze(-1).to(g.dtype)
        return (
            r * token_mask,
            w,
            k * token_mask,
            v * token_mask,
            a * token_mask,
            b * token_mask,
            g * gate_mask,
        )

    def _run_delta_rule(
        self,
        r: Tensor,
        w: Tensor,
        k: Tensor,
        v: Tensor,
        a: Tensor,
        b: Tensor,
        initial_state: Optional[Tensor] = None,
        update_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        mixed, state, backend = generalized_delta_rule(
            r=r,
            w=w,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state=initial_state,
            output_final_state=True,
            mask=update_mask,
            prefer_kernel=True,
            return_backend=True,
        )
        self.last_backend = backend
        return mixed, state

    def _finalize(self, mixed: Tensor, gate: Tensor) -> Tensor:
        if mixed.numel() == 0:
            return mixed.reshape(mixed.size(0), mixed.size(1), self.d_model)

        bsz, seq_len, _, _ = mixed.shape
        merged = mixed.reshape(bsz, seq_len, self.d_model)
        merged = self.ln_x(merged.reshape(bsz * seq_len, self.d_model)).view(bsz, seq_len, self.d_model)
        return self.output(merged * gate)

    def _scan_sequence(
        self,
        x: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        bsz, seq_len, _ = x.shape
        if seq_len == 0:
            empty_state = torch.zeros(bsz, self.nhead, self.head_dim, self.head_dim, dtype=torch.float32, device=x.device)
            empty_token = x.new_zeros(bsz, self.d_model)
            return x, empty_state, empty_token

        valid_tokens = None if padding_mask is None else ~padding_mask
        r, w, k, v, a, b, g = self._project(x)
        r, w, k, v, a, b, g = self._mask_projections(r, w, k, v, a, b, g, valid_tokens)
        update_mask = None if valid_tokens is None else valid_tokens.to(dtype=torch.float32, device=x.device)

        mixed, final_state = self._run_delta_rule(r, w, k, v, a, b, update_mask=update_mask)
        out = self._finalize(mixed, g)
        if padding_mask is not None:
            out = out.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        last_token = self._gather_last_valid_token(x, valid_tokens)
        return out, final_state, last_token

    def _forward_query_only(
        self,
        x: Tensor,
        support_state: Tensor,
        last_support_token: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        bsz, seq_len, _ = x.shape
        if seq_len == 0:
            return x, support_state, last_support_token

        valid_tokens = None if padding_mask is None else ~padding_mask
        r, w, k, v, a, b, g = self._project(x, prev_token=last_support_token)
        r, w, k, v, a, b, g = self._mask_projections(r, w, k, v, a, b, g, valid_tokens)

        # Query tokens should read the support state without updating it.
        freeze_mask = torch.zeros(bsz, seq_len, dtype=torch.float32, device=x.device)
        mixed, _ = self._run_delta_rule(r, w, k, v, a, b, initial_state=support_state, update_mask=freeze_mask)
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
            support_state = torch.zeros(
                bsz,
                self.nhead,
                self.head_dim,
                self.head_dim,
                dtype=torch.float32,
                device=x.device,
            )
            last_support_token = x.new_zeros(bsz, self.d_model)

        if query.size(1) == 0:
            return support_out, support_state, last_support_token

        query_out, _, _ = self._forward_query_only(query, support_state, last_support_token)
        return torch.cat([support_out, query_out], dim=1), support_state, last_support_token

    def _gather_last_valid_token(self, x: Tensor, valid_tokens: Optional[Tensor]) -> Tensor:
        if x.size(1) == 0:
            return x.new_zeros(x.size(0), self.d_model)
        if valid_tokens is None:
            return x[:, -1, :]

        has_valid = valid_tokens.any(dim=1)
        reversed_mask = valid_tokens.flip(1).to(torch.int64)
        last_idx = x.size(1) - 1 - reversed_mask.argmax(dim=1)
        gathered = x[torch.arange(x.size(0), device=x.device), last_idx]
        return torch.where(has_valid.unsqueeze(-1), gathered, torch.zeros_like(gathered))
