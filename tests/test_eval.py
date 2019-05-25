import numpy as np
from src.eval import report


def test_report_keys():
    y = np.array([0, 1, 1, 0, 1, 0])
    p = np.array([0, 1, 0, 0, 1, 0])
    out = report(y, p)
    for k in ("accuracy", "precision", "recall", "f1"):
        assert k in out


def test_report_with_proba():
    y = np.array([0, 1, 1, 0])
    p = np.array([0, 1, 1, 0])
    proba = np.array([0.1, 0.9, 0.8, 0.2])
    out = report(y, p, proba)
    assert "roc_auc" in out
    assert out["accuracy"] == 1.0
