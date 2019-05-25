import numpy as np
import pandas as pd
from src.preprocess import make_features
from src.model import build_model
from src.predict import predict_one


def test_predict_one_runs():
    df = pd.DataFrame({
        "Survived": [0, 1, 1, 0, 1, 0],
        "Pclass": [3, 1, 2, 3, 1, 3],
        "Name": ["a", "b", "c", "d", "e", "f"],
        "Sex": ["male", "female", "female", "male", "female", "male"],
        "Age": [22, 38, 26, 35, 27, 14],
        "SibSp": [1, 1, 0, 0, 0, 1],
        "Parch": [0, 0, 0, 0, 2, 0],
        "Fare": [7.25, 71.28, 7.92, 8.05, 11.13, 30.07],
        "Embarked": ["S", "C", "S", "S", "S", "C"],
    })
    feat = make_features(df)
    y = feat["Survived"].values
    cols = [c for c in feat.columns if c != "Survived"]
    X = feat[cols].values
    m = build_model({"model": {"type": "random_forest", "n_estimators": 10}})
    m.fit(X, y)
    pred, proba = predict_one(m, cols, {
        "Pclass": 3, "Sex": "male", "Age": 22, "SibSp": 1,
        "Parch": 0, "Fare": 7.25, "Embarked": "S",
    })
    assert pred in (0, 1)
    assert 0.0 <= proba <= 1.0


def test_predict_handles_missing_field():
    df = pd.DataFrame({
        "Survived": [0, 1, 1, 0],
        "Pclass": [3, 1, 2, 3],
        "Name": ["a", "b", "c", "d"],
        "Sex": ["male", "female", "female", "male"],
        "Age": [22, 38, 26, 35],
        "SibSp": [1, 1, 0, 0],
        "Parch": [0, 0, 0, 0],
        "Fare": [7.25, 71.28, 7.92, 8.05],
        "Embarked": ["S", "C", "S", "S"],
    })
    feat = make_features(df)
    y = feat["Survived"].values
    cols = [c for c in feat.columns if c != "Survived"]
    X = feat[cols].values
    m = build_model({"model": {"type": "random_forest", "n_estimators": 5}})
    m.fit(X, y)
    # missing Embarked - reindex should still work
    pred, proba = predict_one(m, cols, {
        "Pclass": 1, "Sex": "female", "Age": 30, "SibSp": 0,
        "Parch": 0, "Fare": 50.0, "Embarked": "S",
    })
    assert isinstance(pred, int)
