import torch

from tabicl import TabICL
from tabicl.model.rwkv_time_mix import RWKV7TimeMixForTabICL
from tabicl.model.tabicl import _forward_with_explicit_cache, _initialize_tabicl_cache


def test_rwkv7_time_mix_shape_smoke():
    module = RWKV7TimeMixForTabICL(d_model=8, nhead=2)
    x = torch.randn(2, 5, 8)

    out = module(x, mask=3)

    assert out.shape == x.shape


def test_tabicl_rwkv7_train_forward_smoke():
    model = TabICL(
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
    model.train()

    x = torch.randn(2, 6, 4)
    y_train = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1]], dtype=torch.float32)

    out = model(x, y_train=y_train)

    assert out.shape == (2, 2, 3)


def test_tabicl_rwkv7_cache_smoke():
    model = TabICL(
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
