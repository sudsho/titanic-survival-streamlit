"""Run prediction on a single passenger record."""
import joblib
import pandas as pd
from src.preprocess import make_features


def load_artifacts(model_path):
    return joblib.load(model_path)


def predict_one(model, feat_cols, record):
    df = pd.DataFrame([record])
    df = make_features(df)
    # some columns might be missing on a single-record input,
    # reindex against the saved feat_cols and zero-fill
    df = df.reindex(columns=feat_cols, fill_value=0)
    X = df.values.astype(float)
    proba = model.predict_proba(X)[0][1]
    pred = int(proba >= 0.5)
    return pred, float(proba)
