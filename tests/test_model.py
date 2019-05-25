import numpy as np
from src.model import build_model


def test_build_default_rf():
    cfg = {"model": {"type": "random_forest", "n_estimators": 10, "max_depth": 3}}
    m = build_model(cfg)
    assert m.n_estimators == 10
    assert m.max_depth == 3


def test_build_logreg():
    cfg = {"model": {"type": "logreg", "C": 0.5}}
    m = build_model(cfg)
    assert m.C == 0.5


def test_unknown_model_raises():
    try:
        build_model({"model": {"type": "blah"}})
    except ValueError:
        return
    assert False, "expected ValueError"


def test_rf_can_fit_predict():
    rng = np.random.RandomState(0)
    X = rng.randn(50, 4)
    y = (X[:, 0] > 0).astype(int)
    m = build_model({"model": {"type": "random_forest", "n_estimators": 5, "max_depth": 2}})
    m.fit(X, y)
    p = m.predict(X)
    assert p.shape == y.shape
