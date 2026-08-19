"""Offline end-to-end smoke test for the Titanic survival project.

What it does (no network required):
  1. Load Titanic-schema data. Prefer the bundled data/train.csv; if that is
     missing, generate a small synthetic dataframe with the same columns so the
     smoke still runs completely offline.
  2. Build features and train the sklearn classifier, printing accuracy and AUC.
  3. Exercise the serving/predict path the web app calls:
       a. src.predict.predict_one on a sample passenger (asserts proba in [0,1]).
       b. the Flask app's POST /predict route via its test client, which runs
          make_features + predict_proba + the SHAP explanation, asserting a 200
          response with a rendered probability.

Run:  python scripts/smoke.py
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# Make the repo root importable regardless of where this is launched from.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import load_config, ensure_dir
from src.preprocess import make_features
from src.model import build_model
from src.eval import report
from src.predict import predict_one


def synthetic_titanic(n=200, seed=42):
    """Generate a small Titanic-schema dataframe for offline fallback."""
    rng = np.random.RandomState(seed)
    sex = rng.choice(["male", "female"], size=n, p=[0.65, 0.35])
    pclass = rng.choice([1, 2, 3], size=n, p=[0.24, 0.21, 0.55])
    age = np.clip(rng.normal(29, 14, size=n), 0.5, 80).round(1)
    sibsp = rng.poisson(0.5, size=n)
    parch = rng.poisson(0.4, size=n)
    fare = np.round(np.clip(rng.exponential(30, size=n), 4, 260), 2)
    embarked = rng.choice(["S", "C", "Q"], size=n, p=[0.72, 0.19, 0.09])
    titles = np.where(sex == "female", "Mrs", "Mr")
    # Survival driven by sex/class/age so the classifier has signal to learn.
    logit = (
        1.6 * (sex == "female")
        - 0.9 * (pclass - 1)
        - 0.02 * (age - 29)
        + 0.15 * (fare / 30.0)
        - 0.6
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    survived = (rng.rand(n) < prob).astype(int)
    return pd.DataFrame(
        {
            "PassengerId": np.arange(1, n + 1),
            "Survived": survived,
            "Pclass": pclass,
            "Name": [f"Synthetic, {t}. Passenger {i}" for i, t in enumerate(titles)],
            "Sex": sex,
            "Age": age,
            "SibSp": sibsp,
            "Parch": parch,
            "Ticket": ["TICKET" for _ in range(n)],
            "Fare": fare,
            "Cabin": [None for _ in range(n)],
            "Embarked": embarked,
        }
    )


def load_data(cfg):
    path = cfg["data"]["train_csv"]
    abspath = path if os.path.isabs(path) else os.path.join(ROOT, path)
    if os.path.exists(abspath):
        df = pd.read_csv(abspath)
        print(f"[data] using bundled CSV: {path}  ({len(df)} rows)")
        return df
    print("[data] bundled CSV not found -> using synthetic Titanic-schema data")
    return synthetic_titanic()


def main():
    print("=" * 60)
    print("Titanic survival - offline smoke test")
    print("=" * 60)

    cfg = load_config(os.path.join(ROOT, "configs", "default.yaml"))

    df = load_data(cfg)
    df = make_features(df)
    y = df["Survived"].values
    feat_cols = [c for c in df.columns if c != "Survived"]
    X = df[feat_cols].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"], stratify=y,
    )

    print("\n[train] fitting classifier:", cfg["model"]["type"])
    model = build_model(cfg)
    model.fit(X_tr, y_tr)

    train_acc = model.score(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]
    metrics = report(y_te, y_pred, y_proba)
    print(f"[train] train accuracy : {train_acc:.3f}")
    print(f"[train] test accuracy  : {metrics['accuracy']:.3f}")
    print(f"[train] test roc_auc   : {metrics['roc_auc']:.3f}")
    print(f"[train] feature cols   : {feat_cols}")

    # Persist artifacts so the Flask app can load the model in step 3b.
    out = cfg["artifacts"]["model_path"]
    out = out if os.path.isabs(out) else os.path.join(ROOT, out)
    ensure_dir(out)
    import joblib
    joblib.dump({"model": model, "feat_cols": feat_cols}, out)
    print(f"[train] saved model    : {os.path.relpath(out, ROOT)}")

    # 3a. Direct call to the predict helper on a sample passenger.
    sample = {
        "Pclass": 3, "Sex": "male", "Age": 22, "SibSp": 1,
        "Parch": 0, "Fare": 7.25, "Embarked": "S",
    }
    pred, proba = predict_one(model, feat_cols, sample)
    print("\n[serve] predict_one(sample) ->",
          f"pred={pred}  proba={proba:.4f}")
    assert pred in (0, 1), "prediction must be 0 or 1"
    assert 0.0 <= proba <= 1.0, "probability must be in [0, 1]"

    # 3b. Exercise the actual Flask serving route (make_features + predict +
    # SHAP explanation) through the app's test client.
    os.environ["TITANIC_CFG"] = os.path.join(ROOT, "configs", "default.yaml")
    import app as appmod
    client = appmod.app.test_client()

    h = client.get("/health")
    assert h.status_code == 200 and h.get_json()["status"] == "ok"
    print("[serve] GET /health ->", h.status_code, h.get_json())

    form = {k: str(v) for k, v in sample.items()}
    r = client.post("/predict", data=form)
    assert r.status_code == 200, f"/predict returned {r.status_code}"
    body = r.get_data(as_text=True)
    assert "data:image/png;base64" in body, "SHAP chart missing from response"
    assert ("Survived" in body or "Did not survive" in body), "no label rendered"
    print("[serve] POST /predict ->", r.status_code,
          "(rendered label + SHAP chart)")

    print("\n" + "=" * 60)
    print("SMOKE PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
