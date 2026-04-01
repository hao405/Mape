from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor

TRITON_CHUNK_LEN = 16
TRITON_SUPPORTED_HEAD_SIZE = 64

_TRITON_IMPORT_ERROR: Optional[str] = None

try:  # pragma: no cover - exercised only on hosts with Triton installed
    import triton
    import triton.language as tl
except Exception as exc:  # pragma: no cover - depends on local Triton install
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = str(exc)


def is_triton_available() -> bool:
    return triton is not None and tl is not None


def get_triton_import_error() -> Optional[str]:
    return _TRITON_IMPORT_ERROR


def build_triton_backend() -> Optional[Callable]:
    if not is_triton_available():
        return None
    return generalized_delta_rule_triton


if is_triton_available():  # pragma: no branch - the whole block is runtime gated

    @triton.autotune(
        configs=[
            # AMD/ROCm is much more stable when each block handles a single batch item.
            triton.Config({"MINI_BSZ": 1}, num_warps=num_warps, num_stages=num_stages)
            for num_warps in [4, 8]
            for num_stages in [2, 3, 4]
        ],
        key=["H_SIZE"],
    )
    @triton.jit
    def rwkv7_fwd_kernel(
        R,
        W,
        K,
        V,
        A,
        B_param,
        H0,
        B_BATCH,
        N_HEAD,
        T_LEN,
        OUT,
        SA_OUT,
        STATE_CHKP,
        H_SIZE: tl.constexpr,
        CHUNK_LEN: tl.constexpr,
        MINI_BSZ: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)

        b_range = pid_b * MINI_BSZ + tl.arange(0, MINI_BSZ)
        b_mask = b_range < B_BATCH
        cols = tl.arange(0, H_SIZE)

        # Use int64 for address math so large tensors stay correct on ROCm as well.
        b_range_i64 = b_range.to(tl.int64)
        n_head_i64 = tl.cast(N_HEAD, tl.int64)
        t_len_i64 = tl.cast(T_LEN, tl.int64)
        h_size_i64 = tl.cast(H_SIZE, tl.int64)
        pid_h_i64 = tl.cast(pid_h, tl.int64)
        chunk_len_i64 = tl.cast(CHUNK_LEN, tl.int64)

        ptr_mask = b_mask[:, None]
        state_mask = b_mask[:, None, None]

        base_ptr_off = (
            (b_range_i64[:, None] * n_head_i64 * t_len_i64 * h_size_i64)
            + (pid_h_i64 * t_len_i64 * h_size_i64)
            + cols[None, :]
        )
        h0_base = (b_range_i64[:, None, None] * n_head_i64 * h_size_i64 * h_size_i64) + (
            pid_h_i64 * h_size_i64 * h_size_i64
        )

        row_idx = tl.arange(0, H_SIZE)[None, :, None]
        col_idx = tl.arange(0, H_SIZE)[None, None, :]
        state = tl.load(
            H0 + h0_base + row_idx * H_SIZE + col_idx,
            mask=state_mask,
            other=0.0,
        ).to(tl.float32)

        for t in range(0, T_LEN):
            t_off = t * H_SIZE

            rv = tl.load(R + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
            wv = tl.load(W + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
            kv = tl.load(K + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
            vv = tl.load(V + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
            av = tl.load(A + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
            bv = tl.load(B_param + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)

            w_decay = tl.exp(-tl.exp(wv))
            sa_vec = tl.sum(state * av[:, None, :], axis=2)
            tl.store(
                SA_OUT + base_ptr_off + t_off,
                sa_vec.to(SA_OUT.dtype.element_ty),
                mask=ptr_mask,
            )

            state = (
                state * w_decay[:, None, :]
                + sa_vec[:, :, None] * bv[:, None, :]
                + vv[:, :, None] * kv[:, None, :]
            )

            y_vec = tl.sum(state * rv[:, None, :], axis=2)
            tl.store(
                OUT + base_ptr_off + t_off,
                y_vec.to(OUT.dtype.element_ty),
                mask=ptr_mask,
            )

            if (t + 1) % CHUNK_LEN == 0:
                chkp_t = (tl.cast(t, tl.int64) + 1) // chunk_len_i64 - 1
                chunk_num_i64 = t_len_i64 // chunk_len_i64
                chkp_base = (
                    (
                        b_range_i64[:, None, None]
                        * n_head_i64
                        * chunk_num_i64
                        * h_size_i64
                        * h_size_i64
                    )
                    + (pid_h_i64 * chunk_num_i64 * h_size_i64 * h_size_i64)
                    + (chkp_t * h_size_i64 * h_size_i64)
                )
                tl.store(
                    STATE_CHKP + chkp_base + row_idx * H_SIZE + col_idx,
                    state,
                    mask=state_mask,
                )


    @triton.autotune(
        configs=[
            triton.Config({"MINI_BSZ": 1}, num_warps=num_warps, num_stages=num_stages)
            for num_warps in [4, 8]
            for num_stages in [2, 3, 4]
        ],
        key=["H_SIZE"],
    )
    @triton.jit
    def rwkv7_bwd_kernel(
        R,
        W,
        K,
        V,
        A,
        B_param,
        SA,
        STATE_CHKP,
        B_BATCH,
        N_HEAD,
        T_LEN,
        DY,
        DHT,
        DR,
        DW,
        DK,
        DV,
        DA,
        DB,
        DH0,
        H_SIZE: tl.constexpr,
        CHUNK_LEN: tl.constexpr,
        MINI_BSZ: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)

        b_range = pid_b * MINI_BSZ + tl.arange(0, MINI_BSZ)
        b_mask = b_range < B_BATCH
        cols = tl.arange(0, H_SIZE)
        row_idx = tl.arange(0, H_SIZE)[None, :, None]
        col_idx = tl.arange(0, H_SIZE)[None, None, :]

        b_range_i64 = b_range.to(tl.int64)
        n_head_i64 = tl.cast(N_HEAD, tl.int64)
        t_len_i64 = tl.cast(T_LEN, tl.int64)
        h_size_i64 = tl.cast(H_SIZE, tl.int64)
        pid_h_i64 = tl.cast(pid_h, tl.int64)
        chunk_len_i64 = tl.cast(CHUNK_LEN, tl.int64)

        ptr_mask = b_mask[:, None]
        state_mask = b_mask[:, None, None]

        base_ptr_off = (
            (b_range_i64[:, None] * n_head_i64 * t_len_i64 * h_size_i64)
            + (pid_h_i64 * t_len_i64 * h_size_i64)
            + cols[None, :]
        )
        dht_base = (b_range_i64[:, None, None] * n_head_i64 * h_size_i64 * h_size_i64) + (
            pid_h_i64 * h_size_i64 * h_size_i64
        )

        d_state = tl.load(
            DHT + dht_base + row_idx * H_SIZE + col_idx,
            mask=state_mask,
            other=0.0,
        ).to(tl.float32)
        state_t = tl.zeros([MINI_BSZ, H_SIZE, H_SIZE], dtype=tl.float32)

        for t in range(T_LEN - 1, -1, -1):
            t_off = t * H_SIZE

            rv = tl.load(R + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
            wv = tl.load(W + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
            kv = tl.load(K + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
            vv = tl.load(V + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
            av = tl.load(A + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
            bv = tl.load(B_param + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
            dyv = tl.load(DY + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
            sav = tl.load(SA + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)

            w_decay = tl.exp(-tl.exp(wv))
            w_grad_factor = w_decay * (-tl.exp(wv))

            if (t + 1) % CHUNK_LEN == 0:
                chkp_t = (tl.cast(t, tl.int64) + 1) // chunk_len_i64 - 1
                chunk_num_i64 = t_len_i64 // chunk_len_i64
                chkp_base = (
                    (
                        b_range_i64[:, None, None]
                        * n_head_i64
                        * chunk_num_i64
                        * h_size_i64
                        * h_size_i64
                    )
                    + (pid_h_i64 * chunk_num_i64 * h_size_i64 * h_size_i64)
                    + (chkp_t * h_size_i64 * h_size_i64)
                )
                state_t = tl.load(
                    STATE_CHKP + chkp_base + row_idx * H_SIZE + col_idx,
                    mask=state_mask,
                    other=0.0,
                ).to(tl.float32)

            dr = tl.sum(state_t * dyv[:, :, None], axis=1)
            tl.store(DR + base_ptr_off + t_off, dr.to(DR.dtype.element_ty), mask=ptr_mask)

            inv_w = 1.0 / (w_decay + 1e-6)
            state_t = (
                state_t - vv[:, :, None] * kv[:, None, :] - sav[:, :, None] * bv[:, None, :]
            ) * inv_w[:, None, :]

            d_state = d_state + dyv[:, :, None] * rv[:, None, :]

            dw = tl.sum(d_state * state_t, axis=1) * w_grad_factor
            dk = tl.sum(d_state * vv[:, :, None], axis=1)
            dv = tl.sum(d_state * kv[:, None, :], axis=2)
            db = tl.sum(d_state * sav[:, :, None], axis=1)
            dsa = tl.sum(d_state * bv[:, None, :], axis=2)
            da = tl.sum(state_t * dsa[:, :, None], axis=1)

            tl.store(DW + base_ptr_off + t_off, dw.to(DW.dtype.element_ty), mask=ptr_mask)
            tl.store(DK + base_ptr_off + t_off, dk.to(DK.dtype.element_ty), mask=ptr_mask)
            tl.store(DV + base_ptr_off + t_off, dv.to(DV.dtype.element_ty), mask=ptr_mask)
            tl.store(DB + base_ptr_off + t_off, db.to(DB.dtype.element_ty), mask=ptr_mask)
            tl.store(DA + base_ptr_off + t_off, da.to(DA.dtype.element_ty), mask=ptr_mask)

            d_state = d_state * w_decay[:, None, :] + dsa[:, :, None] * av[:, None, :]

            if t == 0:
                tl.store(
                    DH0 + dht_base + row_idx * H_SIZE + col_idx,
                    d_state.to(DH0.dtype.element_ty),
                    mask=state_mask,
                )


    @triton.autotune(
        configs=[
            triton.Config({"MINI_BSZ": 1}, num_warps=num_warps, num_stages=num_stages)
            for num_warps in [4, 8]
            for num_stages in [2, 3, 4]
        ],
        key=["H_SIZE"],
    )
    @triton.jit
    def rwkv7_fwd_kernel_with_mask(
        R,
        W,
        K,
        V,
        A,
        B_param,
        MASK,
        H0,
        B_BATCH,
        N_HEAD,
        T_LEN,
        OUT,
        SA_OUT,
        STATE_CHKP,
        H_SIZE: tl.constexpr,
        CHUNK_LEN: tl.constexpr,
        MINI_BSZ: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)

        b_range = pid_b * MINI_BSZ + tl.arange(0, MINI_BSZ)
        b_mask = b_range < B_BATCH
        cols = tl.arange(0, H_SIZE)

        b_range_i64 = b_range.to(tl.int64)
        n_head_i64 = tl.cast(N_HEAD, tl.int64)
        t_len_i64 = tl.cast(T_LEN, tl.int64)
        h_size_i64 = tl.cast(H_SIZE, tl.int64)
        pid_h_i64 = tl.cast(pid_h, tl.int64)
        chunk_len_i64 = tl.cast(CHUNK_LEN, tl.int64)

        ptr_mask = b_mask[:, None]
        state_mask = b_mask[:, None, None]

        base_ptr_off = (
            (b_range_i64[:, None] * n_head_i64 * t_len_i64 * h_size_i64)
            + (pid_h_i64 * t_len_i64 * h_size_i64)
            + cols[None, :]
        )
        mask_ptr_base = b_range_i64 * t_len_i64

        h0_base = (b_range_i64[:, None, None] * n_head_i64 * h_size_i64 * h_size_i64) + (
            pid_h_i64 * h_size_i64 * h_size_i64
        )
        row_idx = tl.arange(0, H_SIZE)[None, :, None]
        col_idx = tl.arange(0, H_SIZE)[None, None, :]

        state = tl.load(
            H0 + h0_base + row_idx * H_SIZE + col_idx,
            mask=state_mask,
            other=0.0,
        ).to(tl.float32)

        for t in range(0, T_LEN):
            t_off = t * H_SIZE

            rv = tl.load(R + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
            wv = tl.load(W + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
            kv = tl.load(K + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
            vv = tl.load(V + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
            av = tl.load(A + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
            bv = tl.load(B_param + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)

            m_val = tl.load(MASK + mask_ptr_base + t, mask=b_mask, other=0.0).to(tl.float32)
            m_3d = m_val[:, None, None]

            w_decay = tl.exp(-tl.exp(wv))
            sa_vec = tl.sum(state * av[:, None, :], axis=2)
            tl.store(
                SA_OUT + base_ptr_off + t_off,
                sa_vec.to(SA_OUT.dtype.element_ty),
                mask=ptr_mask,
            )

            state_cand = (
                state * w_decay[:, None, :]
                + sa_vec[:, :, None] * bv[:, None, :]
                + vv[:, :, None] * kv[:, None, :]
            )

            y_vec = tl.sum(state_cand * rv[:, None, :], axis=2)
            tl.store(
                OUT + base_ptr_off + t_off,
                y_vec.to(OUT.dtype.element_ty),
                mask=ptr_mask,
            )

            state = m_3d * state_cand + (1.0 - m_3d) * state

            if (t + 1) % CHUNK_LEN == 0:
                chkp_t = (tl.cast(t, tl.int64) + 1) // chunk_len_i64 - 1
                chunk_num_i64 = t_len_i64 // chunk_len_i64
                chkp_base = (
                    (
                        b_range_i64[:, None, None]
                        * n_head_i64
                        * chunk_num_i64
                        * h_size_i64
                        * h_size_i64
                    )
                    + (pid_h_i64 * chunk_num_i64 * h_size_i64 * h_size_i64)
                    + (chkp_t * h_size_i64 * h_size_i64)
                )
                tl.store(
                    STATE_CHKP + chkp_base + row_idx * H_SIZE + col_idx,
                    state,
                    mask=state_mask,
                )


    @triton.autotune(
        configs=[
            triton.Config({"MINI_BSZ": 1}, num_warps=num_warps, num_stages=num_stages)
            for num_warps in [4, 8]
            for num_stages in [2, 3, 4]
        ],
        key=["H_SIZE"],
    )
    @triton.jit
    def rwkv7_bwd_kernel_with_mask(
        R,
        W,
        K,
        V,
        A,
        B_param,
        MASK,
        SA,
        STATE_CHKP,
        B_BATCH,
        N_HEAD,
        T_LEN,
        DY,
        DHT,
        DR,
        DW,
        DK,
        DV,
        DA,
        DB,
        DH0,
        H_SIZE: tl.constexpr,
        CHUNK_LEN: tl.constexpr,
        MINI_BSZ: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)

        b_range = pid_b * MINI_BSZ + tl.arange(0, MINI_BSZ)
        b_mask = b_range < B_BATCH
        cols = tl.arange(0, H_SIZE)
        row_idx = tl.arange(0, H_SIZE)[None, :, None]
        col_idx = tl.arange(0, H_SIZE)[None, None, :]

        b_range_i64 = b_range.to(tl.int64)
        n_head_i64 = tl.cast(N_HEAD, tl.int64)
        t_len_i64 = tl.cast(T_LEN, tl.int64)
        h_size_i64 = tl.cast(H_SIZE, tl.int64)
        pid_h_i64 = tl.cast(pid_h, tl.int64)
        chunk_len_i64 = tl.cast(CHUNK_LEN, tl.int64)

        ptr_mask = b_mask[:, None]
        state_mask = b_mask[:, None, None]

        base_ptr_off = (
            (b_range_i64[:, None] * n_head_i64 * t_len_i64 * h_size_i64)
            + (pid_h_i64 * t_len_i64 * h_size_i64)
            + cols[None, :]
        )
        mask_ptr_base = b_range_i64 * t_len_i64

        dht_base = (b_range_i64[:, None, None] * n_head_i64 * h_size_i64 * h_size_i64) + (
            pid_h_i64 * h_size_i64 * h_size_i64
        )
        d_state = tl.load(
            DHT + dht_base + row_idx * H_SIZE + col_idx,
            mask=state_mask,
            other=0.0,
        ).to(tl.float32)
        state_t = tl.zeros([MINI_BSZ, H_SIZE, H_SIZE], dtype=tl.float32)

        for t in range(T_LEN - 1, -1, -1):
            t_off = t * H_SIZE

            rv = tl.load(R + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
            wv = tl.load(W + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
            kv = tl.load(K + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
            vv = tl.load(V + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
            av = tl.load(A + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
            bv = tl.load(B_param + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
            dyv = tl.load(DY + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)
            sav = tl.load(SA + base_ptr_off + t_off, mask=ptr_mask, other=0.0).to(tl.float32)

            m_val = tl.load(MASK + mask_ptr_base + t, mask=b_mask, other=0.0).to(tl.float32)
            m_3d = m_val[:, None, None]

            w_decay = tl.exp(-tl.exp(wv))
            w_grad_factor = w_decay * (-tl.exp(wv))

            if (t + 1) % CHUNK_LEN == 0:
                chkp_t = (tl.cast(t, tl.int64) + 1) // chunk_len_i64 - 1
                chunk_num_i64 = t_len_i64 // chunk_len_i64
                chkp_base = (
                    (
                        b_range_i64[:, None, None]
                        * n_head_i64
                        * chunk_num_i64
                        * h_size_i64
                        * h_size_i64
                    )
                    + (pid_h_i64 * chunk_num_i64 * h_size_i64 * h_size_i64)
                    + (chkp_t * h_size_i64 * h_size_i64)
                )
                state_t = tl.load(
                    STATE_CHKP + chkp_base + row_idx * H_SIZE + col_idx,
                    mask=state_mask,
                    other=0.0,
                ).to(tl.float32)

            state_cand_for_dr = (
                state_t * w_decay[:, None, :]
                + sav[:, :, None] * bv[:, None, :]
                + vv[:, :, None] * kv[:, None, :]
            )
            state_for_dr = m_3d * state_t + (1.0 - m_3d) * state_cand_for_dr
            dr = tl.sum(state_for_dr * dyv[:, :, None], axis=1)
            tl.store(DR + base_ptr_off + t_off, dr.to(DR.dtype.element_ty), mask=ptr_mask)

            d_state_curr = dyv[:, :, None] * rv[:, None, :]
            d_state = d_state + d_state_curr
            d_state_old = d_state

            inv_w = 1.0 / (w_decay + 1e-6)
            state_prev = (
                state_t - vv[:, :, None] * kv[:, None, :] - sav[:, :, None] * bv[:, None, :]
            ) * inv_w[:, None, :]
            state_t = m_3d * state_prev + (1.0 - m_3d) * state_t

            d_state_param = m_3d * d_state_old + (1.0 - m_3d) * d_state_curr

            dw = tl.sum(d_state_param * state_t, axis=1) * w_grad_factor
            dk = tl.sum(d_state_param * vv[:, :, None], axis=1)
            dv = tl.sum(d_state_param * kv[:, None, :], axis=2)
            db = tl.sum(d_state_param * sav[:, :, None], axis=1)
            dsa = tl.sum(d_state_param * bv[:, None, :], axis=2)
            da = tl.sum(state_t * dsa[:, :, None], axis=1)

            tl.store(DW + base_ptr_off + t_off, dw.to(DW.dtype.element_ty), mask=ptr_mask)
            tl.store(DK + base_ptr_off + t_off, dk.to(DK.dtype.element_ty), mask=ptr_mask)
            tl.store(DV + base_ptr_off + t_off, dv.to(DV.dtype.element_ty), mask=ptr_mask)
            tl.store(DB + base_ptr_off + t_off, db.to(DB.dtype.element_ty), mask=ptr_mask)
            tl.store(DA + base_ptr_off + t_off, da.to(DA.dtype.element_ty), mask=ptr_mask)

            trans = d_state_param * w_decay[:, None, :] + dsa[:, :, None] * av[:, None, :]
            penetration = d_state_old - d_state_param
            d_state = trans + (1.0 - m_3d) * penetration

            if t == 0:
                tl.store(
                    DH0 + dht_base + row_idx * H_SIZE + col_idx,
                    d_state.to(DH0.dtype.element_ty),
                    mask=state_mask,
                )


    class TritonWindBackstepping(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w: Tensor, q: Tensor, k: Tensor, v: Tensor, a: Tensor, b: Tensor, h0: Tensor):
            batch_size, nhead, seq_len, head_dim = w.shape
            if seq_len % TRITON_CHUNK_LEN != 0:
                raise ValueError(f"Sequence length T={seq_len} must be divisible by {TRITON_CHUNK_LEN}")
            if head_dim != TRITON_SUPPORTED_HEAD_SIZE:
                raise ValueError(
                    f"Triton kernel currently only supports head_size={TRITON_SUPPORTED_HEAD_SIZE}, got {head_dim}"
                )

            out = torch.empty_like(v)
            sa_out = torch.empty(batch_size, nhead, seq_len, head_dim, dtype=torch.float32, device=w.device)
            state_chkp = torch.empty(
                batch_size,
                nhead,
                seq_len // TRITON_CHUNK_LEN,
                head_dim,
                head_dim,
                dtype=torch.float32,
                device=w.device,
            )

            def grid(meta):
                return (triton.cdiv(batch_size, meta["MINI_BSZ"]), nhead)

            rwkv7_fwd_kernel[grid](
                R=q,
                W=w,
                K=k,
                V=v,
                A=a,
                B_param=b,
                H0=h0,
                B_BATCH=batch_size,
                N_HEAD=nhead,
                T_LEN=seq_len,
                OUT=out,
                SA_OUT=sa_out,
                STATE_CHKP=state_chkp,
                H_SIZE=head_dim,
                CHUNK_LEN=TRITON_CHUNK_LEN,
            )

            ctx.save_for_backward(w, q, k, v, a, b, state_chkp, sa_out)
            return out, state_chkp[:, :, -1].clone()

        @staticmethod
        def backward(ctx, dy: Tensor, dht: Optional[Tensor]):
            w, q, k, v, a, b, state_chkp, sa_out = ctx.saved_tensors
            batch_size, nhead, seq_len, head_dim = w.shape

            dw = torch.empty_like(w)
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            da = torch.empty_like(a)
            db = torch.empty_like(b)
            dh0 = torch.empty(batch_size, nhead, head_dim, head_dim, dtype=torch.float32, device=w.device)

            dy = dy.contiguous()
            if dht is None:
                dht = torch.zeros(batch_size, nhead, head_dim, head_dim, dtype=torch.float32, device=w.device)
            else:
                dht = dht.contiguous().to(torch.float32)

            def grid(meta):
                return (triton.cdiv(batch_size, meta["MINI_BSZ"]), nhead)

            rwkv7_bwd_kernel[grid](
                R=q,
                W=w,
                K=k,
                V=v,
                A=a,
                B_param=b,
                SA=sa_out,
                STATE_CHKP=state_chkp,
                B_BATCH=batch_size,
                N_HEAD=nhead,
                T_LEN=seq_len,
                DY=dy,
                DHT=dht,
                DR=dq,
                DW=dw,
                DK=dk,
                DV=dv,
                DA=da,
                DB=db,
                DH0=dh0,
                H_SIZE=head_dim,
                CHUNK_LEN=TRITON_CHUNK_LEN,
            )

            return dw, dq, dk, dv, da, db, dh0


    class TritonWindBacksteppingWithMask(torch.autograd.Function):
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
            batch_size, nhead, seq_len, head_dim = w.shape
            if seq_len % TRITON_CHUNK_LEN != 0:
                raise ValueError(f"Sequence length T={seq_len} must be divisible by {TRITON_CHUNK_LEN}")
            if head_dim != TRITON_SUPPORTED_HEAD_SIZE:
                raise ValueError(
                    f"Triton kernel currently only supports head_size={TRITON_SUPPORTED_HEAD_SIZE}, got {head_dim}"
                )

            mask = mask.contiguous().to(torch.float32)
            out = torch.empty_like(v)
            sa_out = torch.empty(batch_size, nhead, seq_len, head_dim, dtype=torch.float32, device=w.device)
            state_chkp = torch.empty(
                batch_size,
                nhead,
                seq_len // TRITON_CHUNK_LEN,
                head_dim,
                head_dim,
                dtype=torch.float32,
                device=w.device,
            )

            def grid(meta):
                return (triton.cdiv(batch_size, meta["MINI_BSZ"]), nhead)

            rwkv7_fwd_kernel_with_mask[grid](
                R=q,
                W=w,
                K=k,
                V=v,
                A=a,
                B_param=b,
                MASK=mask,
                H0=h0,
                B_BATCH=batch_size,
                N_HEAD=nhead,
                T_LEN=seq_len,
                OUT=out,
                SA_OUT=sa_out,
                STATE_CHKP=state_chkp,
                H_SIZE=head_dim,
                CHUNK_LEN=TRITON_CHUNK_LEN,
            )

            ctx.save_for_backward(w, q, k, v, a, b, mask, state_chkp, sa_out)
            return out, state_chkp[:, :, -1].clone()

        @staticmethod
        def backward(ctx, dy: Tensor, dht: Optional[Tensor]):
            w, q, k, v, a, b, mask, state_chkp, sa_out = ctx.saved_tensors
            batch_size, nhead, seq_len, head_dim = w.shape

            dw = torch.empty_like(w)
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            da = torch.empty_like(a)
            db = torch.empty_like(b)
            dh0 = torch.empty(batch_size, nhead, head_dim, head_dim, dtype=torch.float32, device=w.device)

            dy = dy.contiguous()
            if dht is None:
                dht = torch.zeros(batch_size, nhead, head_dim, head_dim, dtype=torch.float32, device=w.device)
            else:
                dht = dht.contiguous().to(torch.float32)

            def grid(meta):
                return (triton.cdiv(batch_size, meta["MINI_BSZ"]), nhead)

            rwkv7_bwd_kernel_with_mask[grid](
                R=q,
                W=w,
                K=k,
                V=v,
                A=a,
                B_param=b,
                MASK=mask,
                SA=sa_out,
                STATE_CHKP=state_chkp,
                B_BATCH=batch_size,
                N_HEAD=nhead,
                T_LEN=seq_len,
                DY=dy,
                DHT=dht,
                DR=dq,
                DW=dw,
                DK=dk,
                DV=dv,
                DA=da,
                DB=db,
                DH0=dh0,
                H_SIZE=head_dim,
                CHUNK_LEN=TRITON_CHUNK_LEN,
            )

            return dw, dq, dk, dv, da, db, None, dh0


    def generalized_delta_rule_triton(
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
        if w.device.type != "cuda":
            raise RuntimeError("Triton backend only supports CUDA-style devices")

        r, w, k, v, a, b = [tensor.transpose(1, 2).contiguous() for tensor in (r, w, k, v, a, b)]
        batch_size, nhead, _, head_dim = w.shape
        if head_dim != TRITON_SUPPORTED_HEAD_SIZE:
            raise ValueError(
                f"Triton backend only supports head_size={TRITON_SUPPORTED_HEAD_SIZE}, got {head_dim}"
            )

        if initial_state is None:
            initial_state = torch.zeros(
                batch_size,
                nhead,
                head_dim,
                head_dim,
                dtype=torch.float32,
                device=r.device,
            )
        elif initial_state.shape[0] == 1 and batch_size != 1:
            initial_state = initial_state.expand(batch_size, -1, -1, -1).contiguous()
        else:
            initial_state = initial_state.to(device=r.device, dtype=torch.float32).contiguous()

        if mask is None:
            out, state = TritonWindBackstepping.apply(w, r, k, v, a, b, initial_state)
        else:
            mask = mask.to(device=r.device, dtype=torch.float32).contiguous()
            out, state = TritonWindBacksteppingWithMask.apply(w, r, k, v, a, b, mask, initial_state)

        return out.transpose(1, 2).contiguous(), state
