from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Callable, Optional

import torch
from torch import Tensor
from torch.utils.cpp_extension import load

CHUNK_LEN = 16
_MIN_CUDA_HEAD_SIZE = 4
_CUDA_KERNELS: dict[int, Callable] = {}
_CUDA_KERNEL_ERRORS: dict[int, str] = {}


def get_kernel_error(head_size: int) -> Optional[str]:
    return _CUDA_KERNEL_ERRORS.get(head_size)


def generalized_delta_rule(
    r: Tensor,
    w: Tensor,
    k: Tensor,
    v: Tensor,
    a: Tensor,
    b: Tensor,
    initial_state: Optional[Tensor] = None,
    output_final_state: bool = True,
    head_first: bool = False,
    mask: Optional[Tensor] = None,
    prefer_kernel: bool = True,
    return_backend: bool = False,
):
    r, w, k, v, a, b, initial_state, mask = _canonicalize_inputs(
        r=r,
        w=w,
        k=k,
        v=v,
        a=a,
        b=b,
        initial_state=initial_state,
        mask=mask,
        head_first=head_first,
    )

    backend = "torch"
    kernel = None
    if prefer_kernel:
        kernel = _get_cuda_kernel(head_size=r.shape[-1], device=r.device)

    if kernel is not None:
        try:
            out, final_state = _run_cuda_kernel(
                kernel=kernel,
                r=r,
                w=w,
                k=k,
                v=v,
                a=a,
                b=b,
                initial_state=initial_state,
                mask=mask,
            )
            backend = "cuda"
        except Exception as exc:  # pragma: no cover - exercised only when kernel runtime fails
            _CUDA_KERNELS.pop(r.shape[-1], None)
            _CUDA_KERNEL_ERRORS[r.shape[-1]] = str(exc)
            warnings.warn(
                f"RWKV7 CUDA kernel runtime failed for head_size={r.shape[-1]}; falling back to PyTorch. "
                f"Reason: {exc}",
                RuntimeWarning,
            )
            out, final_state = generalized_delta_rule_torch(
                r=r,
                w=w,
                k=k,
                v=v,
                a=a,
                b=b,
                initial_state=initial_state,
                output_final_state=True,
                mask=mask,
            )
    else:
        out, final_state = generalized_delta_rule_torch(
            r=r,
            w=w,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state=initial_state,
            output_final_state=True,
            mask=mask,
        )

    if head_first:
        out = out.transpose(1, 2)

    if return_backend:
        if output_final_state:
            return out, final_state, backend
        return out, backend

    if output_final_state:
        return out, final_state
    return out


def generalized_delta_rule_torch(
    r: Tensor,
    w: Tensor,
    k: Tensor,
    v: Tensor,
    a: Tensor,
    b: Tensor,
    initial_state: Optional[Tensor] = None,
    output_final_state: bool = True,
    mask: Optional[Tensor] = None,
):
    if r.dim() != 4:
        raise ValueError(f"Expected [B, T, H, K] inputs, got {tuple(r.shape)}")

    batch_size, seq_len, nhead, head_dim = r.shape
    if initial_state is None:
        state = torch.zeros(batch_size, nhead, head_dim, head_dim, dtype=torch.float32, device=r.device)
    else:
        state = initial_state.to(device=r.device, dtype=torch.float32)
        if state.shape[0] == 1 and batch_size != 1:
            state = state.expand(batch_size, -1, -1, -1).contiguous()

    dtype = r.dtype
    r_f = r.to(torch.float32)
    w_decay = torch.exp(-torch.exp(w.to(torch.float32)))
    k_f = k.to(torch.float32)
    v_f = v.to(torch.float32)
    a_f = a.to(torch.float32)
    b_f = b.to(torch.float32)
    mask_f = None if mask is None else mask.to(device=r.device, dtype=torch.float32)

    outputs = torch.empty(batch_size, seq_len, nhead, head_dim, dtype=dtype, device=r.device)
    for idx in range(seq_len):
        prev_state = state
        kk = k_f[:, idx].unsqueeze(-2)
        vv = v_f[:, idx].unsqueeze(-1)
        aa = a_f[:, idx].unsqueeze(-1)
        bb = b_f[:, idx].unsqueeze(-2)

        candidate_state = state * w_decay[:, idx, :, None, :] + (state @ aa) @ bb + vv @ kk
        rr = r_f[:, idx].unsqueeze(-1)
        outputs[:, idx] = (candidate_state @ rr).squeeze(-1).to(dtype)

        if mask_f is None:
            state = candidate_state
        else:
            step_mask = mask_f[:, idx].view(batch_size, 1, 1, 1)
            state = candidate_state * step_mask + prev_state * (1.0 - step_mask)

    if output_final_state:
        return outputs, state
    return outputs


def _canonicalize_inputs(
    r: Tensor,
    w: Tensor,
    k: Tensor,
    v: Tensor,
    a: Tensor,
    b: Tensor,
    initial_state: Optional[Tensor],
    mask: Optional[Tensor],
    head_first: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    tensors = [r, w, k, v, a, b]
    if head_first:
        tensors = [tensor.transpose(1, 2) for tensor in tensors]
    r, w, k, v, a, b = tensors

    if len({tuple(tensor.shape) for tensor in tensors}) != 1:
        raise ValueError("All RWKV7 projection tensors must share the same shape")

    if mask is not None:
        batch_size, seq_len = r.shape[:2]
        if mask.shape != (batch_size, seq_len):
            raise ValueError(f"Expected mask shape {(batch_size, seq_len)}, got {tuple(mask.shape)}")

    if initial_state is not None:
        batch_size, _, nhead, head_dim = r.shape
        expected = (batch_size, nhead, head_dim, head_dim)
        relaxed = (1, nhead, head_dim, head_dim)
        if initial_state.shape not in (expected, relaxed):
            raise ValueError(f"Expected initial_state shape {expected} or {relaxed}, got {tuple(initial_state.shape)}")

    return r, w, k, v, a, b, initial_state, mask


def _get_cuda_kernel(head_size: int, device: torch.device) -> Optional[Callable]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    if head_size % _MIN_CUDA_HEAD_SIZE != 0:
        _CUDA_KERNEL_ERRORS.setdefault(
            head_size,
            f"head_size must be divisible by {_MIN_CUDA_HEAD_SIZE} for the CUDA kernel",
        )
        return None
    if head_size in _CUDA_KERNELS:
        return _CUDA_KERNELS[head_size]
    if head_size in _CUDA_KERNEL_ERRORS:
        return None

    source_dir = Path(__file__).with_name("rwkv7_cuda_kernel")
    namespace = f"tabicl_wind_backstepping_h{head_size}"
    build_name = f"tabicl_rwkv7_kernel_h{head_size}"

    if os.name == "nt":
        extra_cflags = [f"/DTABICL_RWKV7_OP_NAMESPACE={namespace}", "/O2"]
    else:
        extra_cflags = [f"-DTABICL_RWKV7_OP_NAMESPACE={namespace}", "-O3"]

    extra_cuda_cflags = [
        f"-DTABICL_RWKV7_OP_NAMESPACE={namespace}",
        f"-D_C_={head_size}",
        f"-D_CHUNK_LEN_={CHUNK_LEN}",
        "-res-usage",
        "--use_fast_math",
        "-O3",
        "-Xptxas",
        "-O3",
        "--extra-device-vectorization",
    ]

    try:
        load(
            name=build_name,
            sources=[
                str(source_dir / "wkv7_op.cpp"),
                str(source_dir / "wkv7_cuda.cu"),
            ],
            is_python_module=False,
            verbose=False,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
        )
    except Exception as exc:  # pragma: no cover - depends on host CUDA toolchain
        _CUDA_KERNEL_ERRORS[head_size] = str(exc)
        warnings.warn(
            f"RWKV7 CUDA kernel could not be compiled for head_size={head_size}; falling back to PyTorch. "
            f"Reason: {exc}",
            RuntimeWarning,
        )
        return None

    kernel = _build_cuda_kernel(namespace=namespace)
    _CUDA_KERNELS[head_size] = kernel
    return kernel


def _build_cuda_kernel(namespace: str) -> Callable:
    ops = getattr(torch.ops, namespace)

    class WindBackstepping(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w: Tensor, q: Tensor, k: Tensor, v: Tensor, a: Tensor, b: Tensor, h0: Tensor):
            batch_size, seq_len, nhead, head_dim = w.shape
            dtype = q.dtype
            q_bf, k_bf, v_bf, a_bf, b_bf, w_bf = [
                tensor.to(dtype=torch.bfloat16).contiguous() for tensor in (q, k, v, a, b, w)
            ]
            h0_f32 = h0.to(dtype=torch.float32).contiguous()

            y = torch.empty_like(v_bf)
            states = torch.empty(
                batch_size,
                nhead,
                seq_len // CHUNK_LEN,
                head_dim,
                head_dim,
                dtype=torch.float32,
                device=w.device,
            )
            sa = torch.empty(batch_size, seq_len, nhead, head_dim, dtype=torch.float32, device=w.device)

            ops.forward(w_bf, q_bf, k_bf, v_bf, a_bf, b_bf, y, states, sa, h0_f32)
            ctx.save_for_backward(w_bf, q_bf, k_bf, v_bf, a_bf, b_bf, states, sa)

            last_state = states[:, :, -1].transpose(-1, -2).contiguous()
            return y.to(dtype), last_state

        @staticmethod
        def backward(ctx, dy: Tensor, dht: Optional[Tensor]):
            dtype = dy.dtype
            w_bf, q_bf, k_bf, v_bf, a_bf, b_bf, states, sa = ctx.saved_tensors
            dy_bf = dy.to(dtype=torch.bfloat16).contiguous()
            if dht is None:
                dht_f32 = torch.zeros_like(states[:, :, -1].transpose(-1, -2))
            else:
                dht_f32 = dht.to(dtype=torch.float32).contiguous()

            dh0 = torch.empty_like(dht_f32)
            dw, dq, dk, dv, da, db = [torch.empty_like(tensor) for tensor in (w_bf, q_bf, k_bf, v_bf, a_bf, b_bf)]
            ops.backward(w_bf, q_bf, k_bf, v_bf, a_bf, b_bf, dy_bf, states, sa, dht_f32, dh0, dw, dq, dk, dv, da, db)
            return (
                dw.to(dtype),
                dq.to(dtype),
                dk.to(dtype),
                dv.to(dtype),
                da.to(dtype),
                db.to(dtype),
                dh0,
            )

    class WindBacksteppingWithMask(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            w: Tensor,
            q: Tensor,
            k: Tensor,
            v: Tensor,
            a: Tensor,
            b: Tensor,
            mask: Tensor,
            h0: Tensor,
        ):
            batch_size, seq_len, nhead, head_dim = w.shape
            dtype = q.dtype
            q_bf, k_bf, v_bf, a_bf, b_bf, w_bf = [
                tensor.to(dtype=torch.bfloat16).contiguous() for tensor in (q, k, v, a, b, w)
            ]
            mask_bf = mask.to(device=q.device, dtype=torch.bfloat16).contiguous()
            h0_f32 = h0.to(dtype=torch.float32).contiguous()

            y = torch.empty_like(v_bf)
            states = torch.empty(
                batch_size,
                nhead,
                seq_len // CHUNK_LEN,
                head_dim,
                head_dim,
                dtype=torch.float32,
                device=w.device,
            )
            sa = torch.empty(batch_size, seq_len, nhead, head_dim, dtype=torch.float32, device=w.device)

            ops.forward_with_mask(w_bf, q_bf, k_bf, v_bf, a_bf, b_bf, mask_bf, y, states, sa, h0_f32)
            ctx.save_for_backward(w_bf, q_bf, k_bf, v_bf, a_bf, b_bf, mask_bf, states, sa)

            last_state = states[:, :, -1].transpose(-1, -2).contiguous()
            return y.to(dtype), last_state

        @staticmethod
        def backward(ctx, dy: Tensor, dht: Optional[Tensor]):
            dtype = dy.dtype
            w_bf, q_bf, k_bf, v_bf, a_bf, b_bf, mask_bf, states, sa = ctx.saved_tensors
            dy_bf = dy.to(dtype=torch.bfloat16).contiguous()
            if dht is None:
                dht_f32 = torch.zeros_like(states[:, :, -1].transpose(-1, -2))
            else:
                dht_f32 = dht.to(dtype=torch.float32).contiguous()

            dh0 = torch.empty_like(dht_f32)
            dw, dq, dk, dv, da, db = [torch.empty_like(tensor) for tensor in (w_bf, q_bf, k_bf, v_bf, a_bf, b_bf)]
            ops.backward_with_mask(
                w_bf,
                q_bf,
                k_bf,
                v_bf,
                a_bf,
                b_bf,
                mask_bf,
                dy_bf,
                states,
                sa,
                dht_f32,
                dh0,
                dw,
                dq,
                dk,
                dv,
                da,
                db,
            )
            return (
                dw.to(dtype),
                dq.to(dtype),
                dk.to(dtype),
                dv.to(dtype),
                da.to(dtype),
                db.to(dtype),
                None,
                dh0,
            )

    def generalized_delta_rule_cuda(
        *,
        r: Tensor,
        w: Tensor,
        k: Tensor,
        v: Tensor,
        a: Tensor,
        b: Tensor,
        initial_state: Optional[Tensor],
        mask: Optional[Tensor],
    ) -> tuple[Tensor, Tensor]:
        if initial_state is None:
            initial_state = torch.zeros(
                r.size(0),
                r.size(2),
                r.size(3),
                r.size(3),
                dtype=torch.float32,
                device=r.device,
            )
        elif initial_state.shape[0] == 1 and r.size(0) != 1:
            initial_state = initial_state.expand(r.size(0), -1, -1, -1).contiguous()
        else:
            initial_state = initial_state.to(device=r.device, dtype=torch.float32).contiguous()

        requires_grad = torch.is_grad_enabled() and any(tensor.requires_grad for tensor in (r, w, k, v, a, b))
        if requires_grad:
            if mask is None:
                return WindBackstepping.apply(w, r, k, v, a, b, initial_state)
            return WindBacksteppingWithMask.apply(w, r, k, v, a, b, mask, initial_state)

        q_bf, k_bf, v_bf, a_bf, b_bf, w_bf = [
            tensor.to(dtype=torch.bfloat16).contiguous() for tensor in (r, k, v, a, b, w)
        ]
        dtype = r.dtype
        if mask is None:
            y = torch.empty_like(v_bf)
            final_state = torch.empty_like(initial_state)
            ops.forward_inference(w_bf, q_bf, k_bf, v_bf, a_bf, b_bf, y, final_state, initial_state)
        else:
            y = torch.empty_like(v_bf)
            final_state = torch.empty_like(initial_state)
            mask_bf = mask.to(device=r.device, dtype=torch.bfloat16).contiguous()
            ops.forward_inference_with_mask(w_bf, q_bf, k_bf, v_bf, a_bf, b_bf, mask_bf, y, final_state, initial_state)
        return y.to(dtype), final_state

    return generalized_delta_rule_cuda


def _run_cuda_kernel(
    kernel: Callable,
    r: Tensor,
    w: Tensor,
    k: Tensor,
    v: Tensor,
    a: Tensor,
    b: Tensor,
    initial_state: Optional[Tensor],
    mask: Optional[Tensor],
) -> tuple[Tensor, Tensor]:
    padded = _pad_inputs_for_chunk_alignment(r=r, w=w, k=k, v=v, a=a, b=b, mask=mask)
    out, state = kernel(
        r=padded["r"],
        w=padded["w"],
        k=padded["k"],
        v=padded["v"],
        a=padded["a"],
        b=padded["b"],
        initial_state=initial_state,
        mask=padded["mask"],
    )
    return out[:, : r.size(1)].contiguous(), state


def _pad_inputs_for_chunk_alignment(
    *,
    r: Tensor,
    w: Tensor,
    k: Tensor,
    v: Tensor,
    a: Tensor,
    b: Tensor,
    mask: Optional[Tensor],
) -> dict[str, Tensor | None]:
    seq_len = r.size(1)
    padded_len = ((seq_len + CHUNK_LEN - 1) // CHUNK_LEN) * CHUNK_LEN
    effective_mask = None if mask is None else mask.to(device=r.device, dtype=torch.float32)

    if padded_len == seq_len:
        return {"r": r, "w": w, "k": k, "v": v, "a": a, "b": b, "mask": effective_mask}

    if effective_mask is None:
        effective_mask = torch.ones(r.size(0), seq_len, dtype=torch.float32, device=r.device)

    return {
        "r": _pad_sequence_tensor(r, padded_len, pad_value=0.0),
        "w": _pad_sequence_tensor(w, padded_len, pad_value=0.0),
        "k": _pad_sequence_tensor(k, padded_len, pad_value=0.0),
        "v": _pad_sequence_tensor(v, padded_len, pad_value=0.0),
        "a": _pad_sequence_tensor(a, padded_len, pad_value=0.0),
        "b": _pad_sequence_tensor(b, padded_len, pad_value=0.0),
        "mask": _pad_sequence_tensor(effective_mask, padded_len, pad_value=0.0),
    }


def _pad_sequence_tensor(tensor: Tensor, padded_len: int, pad_value: float) -> Tensor:
    if tensor.size(1) == padded_len:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[1] = padded_len - tensor.size(1)
    pad = tensor.new_full(pad_shape, pad_value)
    return torch.cat([tensor, pad], dim=1)
