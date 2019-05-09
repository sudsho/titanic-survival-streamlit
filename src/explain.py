"""SHAP explanations for the titanic model."""
import shap
import numpy as np


def make_explainer(model):
    return shap.TreeExplainer(model)


def shap_values_for(model, X):
    expl = make_explainer(model)
    sv = expl.shap_values(X)
    # for binary classification TreeExplainer returns a list of two arrays
    if isinstance(sv, list):
        sv = sv[1]
    return expl, sv
