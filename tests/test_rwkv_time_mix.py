import pytest
import torch

from tabicl import TabICL
from tabicl.model import rwkv7_kernel_backend
from tabicl.model.rwkv_time_mix import RWKV7TimeMixForTabICL
from tabicl.model.tabicl import _forward_with_explicit_cache, _initialize_tabicl_cache


def _build_rwkv7_model() -> TabICL:
    return TabICL(
        max_classes=3,
        embed_dim=8,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=4,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=2,
        icl_num_blocks=2,
        icl_nhead=2,
        ff_factor=2,
        dropout=0.0,
        attention_impl="rwkv7",
    )


def test_rwkv7_time_mix_shape_smoke():
    module = RWKV7TimeMixForTabICL(d_model=8, nhead=2)
    x = torch.randn(2, 5, 8)

    out = module(x, mask=3)

    assert out.shape == x.shape


def test_rwkv7_time_mix_cpu_fallback_smoke():
    module = RWKV7TimeMixForTabICL(d_model=8, nhead=2)
    x = torch.randn(2, 5, 8)

    out = module(x, mask=3)

    assert out.shape == x.shape
    assert module.last_backend == "torch"


def test_tabicl_rwkv7_train_forward_smoke():
    model = _build_rwkv7_model()
    model.train()

    x = torch.randn(2, 6, 4)
    y_train = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1]], dtype=torch.float32)

    out = model(x, y_train=y_train)

    assert out.shape == (2, 2, 3)


def test_tabicl_rwkv7_cache_smoke():
    model = _build_rwkv7_model()
    model.eval()

    x_train = torch.randn(2, 4, 4)
    y_train = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1]], dtype=torch.float32)
    x_test = torch.randn(2, 2, 4)

    cache = _initialize_tabicl_cache(x_train, y_train)
    stored = _forward_with_explicit_cache(
        model,
        cache=cache,
        X_train=x_train,
        y_train=y_train,
        use_cache=False,
        store_cache=True,
    )
    pred = _forward_with_explicit_cache(
        model,
        cache=cache,
        X_test=x_test,
        use_cache=True,
        store_cache=False,
    )

    assert stored is None
    assert pred.shape == (2, 2, 3)


def test_tabicl_rwkv7_non_multiple_of_16_sequence_lengths():
    model = _build_rwkv7_model()
    model.eval()

    x_train = torch.randn(2, 5, 4)
    y_train = torch.tensor([[0, 1, 2, 0, 1], [1, 2, 0, 1, 2]], dtype=torch.float32)
    x_test = torch.randn(2, 3, 4)

    cache = _initialize_tabicl_cache(x_train, y_train)
    stored = _forward_with_explicit_cache(
        model,
        cache=cache,
        X_train=x_train,
        y_train=y_train,
        use_cache=False,
        store_cache=True,
    )
    pred = _forward_with_explicit_cache(
        model,
        cache=cache,
        X_test=x_test,
        use_cache=True,
        store_cache=False,
    )

    assert stored is None
    assert pred.shape == (2, 3, 3)


def test_tabicl_rwkv7_query_path_does_not_mutate_cache():
    model = _build_rwkv7_model()
    model.eval()

    x_train = torch.randn(2, 4, 4)
    y_train = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1]], dtype=torch.float32)
    x_test = torch.randn(2, 3, 4)

    cache = _initialize_tabicl_cache(x_train, y_train)
    _forward_with_explicit_cache(
        model,
        cache=cache,
        X_train=x_train,
        y_train=y_train,
        use_cache=False,
        store_cache=True,
    )

    before = {
        idx: (entry.key.clone(), entry.value.clone())
        for idx, entry in cache.icl_cache.kv.items()
        if entry.is_valid()
    }

    pred = _forward_with_explicit_cache(
        model,
        cache=cache,
        X_test=x_test,
        use_cache=True,
        store_cache=False,
    )

    assert pred.shape == (2, 3, 3)
    after = {
        idx: (entry.key, entry.value)
        for idx, entry in cache.icl_cache.kv.items()
        if entry.is_valid()
    }
    assert before.keys() == after.keys()
    for idx, (key_before, value_before) in before.items():
        key_after, value_after = after[idx]
        assert torch.equal(key_before, key_after)
        assert torch.equal(value_before, value_after)


def test_select_backend_prefers_triton_on_rocm(monkeypatch):
    monkeypatch.delenv("TABICL_RWKV7_BACKEND", raising=False)
    monkeypatch.setattr(rwkv7_kernel_backend.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(rwkv7_kernel_backend.torch.version, "hip", "6.1", raising=False)
    monkeypatch.setattr(rwkv7_kernel_backend, "_has_triton_backend_support", lambda: True)

    backend = rwkv7_kernel_backend._select_backend(
        head_size=64,
        device=torch.device("cuda"),
        prefer_kernel=True,
    )

    assert backend == "triton"


def test_rocm_skips_cuda_extension_backend(monkeypatch):
    monkeypatch.setattr(rwkv7_kernel_backend.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(rwkv7_kernel_backend.torch.version, "hip", "6.1", raising=False)
    rwkv7_kernel_backend._CUDA_KERNEL_ERRORS.clear()
    rwkv7_kernel_backend._TRITON_KERNEL_ERRORS.clear()

    kernel = rwkv7_kernel_backend._get_cuda_kernel(head_size=64, device=torch.device("cuda"))

    assert kernel is None
    assert "ROCm devices skip the CUDA extension backend" in rwkv7_kernel_backend.get_kernel_error(64)


def test_select_backend_falls_back_to_torch_when_triton_head_size_is_unsupported(monkeypatch):
    monkeypatch.delenv("TABICL_RWKV7_BACKEND", raising=False)
    monkeypatch.setattr(rwkv7_kernel_backend.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(rwkv7_kernel_backend.torch.version, "hip", "6.1", raising=False)
    monkeypatch.setattr(rwkv7_kernel_backend, "_has_triton_backend_support", lambda: True)

    backend = rwkv7_kernel_backend._select_backend(
        head_size=32,
        device=torch.device("cuda"),
        prefer_kernel=True,
    )

    assert backend == "torch"


def test_backend_override_can_force_routing(monkeypatch):
    monkeypatch.setattr(rwkv7_kernel_backend.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(rwkv7_kernel_backend.torch.version, "hip", "6.1", raising=False)
    monkeypatch.setattr(rwkv7_kernel_backend, "_has_triton_backend_support", lambda: True)

    monkeypatch.setenv("TABICL_RWKV7_BACKEND", "torch")
    assert (
        rwkv7_kernel_backend._select_backend(
            head_size=64,
            device=torch.device("cuda"),
            prefer_kernel=True,
        )
        == "torch"
    )

    monkeypatch.setenv("TABICL_RWKV7_BACKEND", "triton")
    assert (
        rwkv7_kernel_backend._select_backend(
            head_size=64,
            device=torch.device("cuda"),
            prefer_kernel=True,
        )
        == "triton"
    )

    monkeypatch.setenv("TABICL_RWKV7_BACKEND", "cuda")
    assert (
        rwkv7_kernel_backend._select_backend(
            head_size=64,
            device=torch.device("cuda"),
            prefer_kernel=True,
        )
        == "torch"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_rwkv7_time_mix_cuda_kernel_smoke():
    module = RWKV7TimeMixForTabICL(d_model=8, nhead=2).cuda()
    x = torch.randn(2, 5, 8, device="cuda")

    out = module(x, mask=3)

    assert out.shape == x.shape
    if module.last_backend != "cuda":
        pytest.skip("CUDA device is available, but the RWKV7 extension did not load on this host")


@pytest.mark.skipif(
    not (torch.cuda.is_available() and getattr(torch.version, "hip", None) is not None),
    reason="ROCm is unavailable",
)
def test_rwkv7_time_mix_rocm_triton_smoke(monkeypatch):
    monkeypatch.setenv("TABICL_RWKV7_BACKEND", "triton")
    module = RWKV7TimeMixForTabICL(d_model=64, nhead=1).cuda()
    x = torch.randn(2, 17, 64, device="cuda")

    out = module(x, mask=8)

    assert out.shape == x.shape
    if module.last_backend != "triton":
        pytest.skip("ROCm device is available, but the Triton backend did not load on this host")
