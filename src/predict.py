"""Run prediction on a single passenger record."""
import joblib
import numpy as np
import pandas as pd
from src.preprocess import make_features


def load_artifacts(model_path):
    return joblib.load(model_path)


def predict_one(model, feat_cols, record):
    df = pd.DataFrame([record])
    df = make_features(df)
    # ensure column order
    df = df[feat_cols]
    proba = model.predict_proba(df.values)[0][1]
    pred = int(proba >= 0.5)
    return pred, float(proba)
