"""SHAP explanations for the titanic model."""
import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import numpy as np


def make_explainer(model):
    return shap.TreeExplainer(model)


def shap_values_for(model, X):
    expl = make_explainer(model)
    sv = expl.shap_values(X)
    # binary classification TreeExplainer output has changed across shap
    # versions: older shap returned a list [class0, class1] of 2D arrays,
    # newer shap returns a single 3D array (n_samples, n_features, n_classes).
    if isinstance(sv, list):
        sv = sv[1]
    elif isinstance(sv, np.ndarray) and sv.ndim == 3:
        sv = sv[:, :, 1]
    return expl, sv


def force_plot_png(model, feat_cols, X_row):
    expl, sv = shap_values_for(model, X_row.reshape(1, -1))
    base = expl.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = float(np.array(base).flatten()[-1])
    fig = plt.figure(figsize=(8, 3))
    contribs = list(zip(feat_cols, sv[0], X_row))
    contribs.sort(key=lambda t: abs(t[1]), reverse=True)
    names = [c[0] for c in contribs]
    vals = [c[1] for c in contribs]
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in vals]
    plt.barh(names[::-1], vals[::-1], color=colors[::-1])
    plt.title("SHAP contribution to survival probability")
    plt.xlabel("contribution")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
