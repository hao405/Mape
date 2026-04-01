import pytest
import torch

from tabicl import TabICL
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_rwkv7_time_mix_cuda_kernel_smoke():
    module = RWKV7TimeMixForTabICL(d_model=8, nhead=2).cuda()
    x = torch.randn(2, 5, 8, device="cuda")

    out = module(x, mask=3)

    assert out.shape == x.shape
    if module.last_backend != "cuda":
        pytest.skip("CUDA device is available, but the RWKV7 extension did not load on this host")
