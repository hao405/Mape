import numpy as np
import pytest
import torch
from sklearn.datasets import make_classification
from sklearn.utils.estimator_checks import parametrize_with_checks

from tabicl import TabICL, TabICLClassifier


def _mock_load_model(self, max_classes: int = 10):
    torch.manual_seed(0)
    self.model_path_ = "mock.ckpt"
    self.model_ = TabICL(
        max_classes=max_classes,
        embed_dim=16,
        col_num_blocks=1,
        col_nhead=4,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=4,
        row_num_cls=2,
        icl_num_blocks=2,
        icl_nhead=2,
        ff_factor=2,
        dropout=0.0,
    )
    self.model_.eval()


# use n_estimators=2 to test other preprocessing as well
@parametrize_with_checks([TabICLClassifier(n_estimators=2)])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


def test_predict_proba_matches_with_kv_cache(monkeypatch):
    monkeypatch.setattr(TabICLClassifier, "_load_model", lambda self: _mock_load_model(self, max_classes=10))

    X, y = make_classification(
        n_samples=50,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        random_state=42,
    )
    X_train, X_test = X[:40], X[40:]
    y_train = y[:40]

    clf = TabICLClassifier(n_estimators=2, batch_size=1, device="cpu", use_amp=False, use_kv_cache=False)
    clf.fit(X_train, y_train)
    pred_no_cache = clf.predict_proba(X_test)

    clf_cached = TabICLClassifier(n_estimators=2, batch_size=1, device="cpu", use_amp=False, use_kv_cache=True)
    clf_cached.fit(X_train, y_train)
    pred_cached = clf_cached.predict_proba(X_test)

    assert clf_cached.model_kv_cache_ is not None
    np.testing.assert_allclose(pred_no_cache, pred_cached, rtol=1e-4, atol=1e-4)


def test_many_class_kv_cache_is_rejected(monkeypatch):
    monkeypatch.setattr(TabICLClassifier, "_load_model", lambda self: _mock_load_model(self, max_classes=2))

    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 5))
    y = np.array([0, 1, 2] * 10)

    clf = TabICLClassifier(n_estimators=2, batch_size=1, device="cpu", use_amp=False, use_kv_cache=True)
    with pytest.raises(ValueError, match="KV caching is not supported"):
        clf.fit(X, y)


def test_cached_classifier_path_does_not_call_model_forward_with_cache(monkeypatch):
    monkeypatch.setattr(TabICLClassifier, "_load_model", lambda self: _mock_load_model(self, max_classes=10))

    def _forbidden(*args, **kwargs):
        raise AssertionError("classifier cached path should not call TabICL.forward_with_cache")

    monkeypatch.setattr(TabICL, "forward_with_cache", _forbidden)

    X, y = make_classification(
        n_samples=50,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        random_state=0,
    )
    X_train, X_test = X[:40], X[40:]
    y_train = y[:40]

    clf = TabICLClassifier(n_estimators=2, batch_size=1, device="cpu", use_amp=False, use_kv_cache=True)
    clf.fit(X_train, y_train)
    pred = clf.predict_proba(X_test)

    assert clf.model_kv_cache_ is not None
    assert pred.shape == (X_test.shape[0], len(np.unique(y_train)))


def test_cached_classifier_path_ignores_model_cache_state(monkeypatch):
    monkeypatch.setattr(TabICLClassifier, "_load_model", lambda self: _mock_load_model(self, max_classes=10))

    X, y = make_classification(
        n_samples=50,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        random_state=1,
    )
    X_train, X_test = X[:40], X[40:]
    y_train = y[:40]

    clf = TabICLClassifier(n_estimators=2, batch_size=1, device="cpu", use_amp=False, use_kv_cache=True)
    clf.fit(X_train, y_train)
    clf.model_._cache = object()

    pred = clf.predict_proba(X_test)

    assert clf.model_kv_cache_ is not None
    assert pred.shape == (X_test.shape[0], len(np.unique(y_train)))
