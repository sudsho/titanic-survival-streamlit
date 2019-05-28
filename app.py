"""Flask app for titanic survival prediction."""
import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from src.utils import load_config
from src.predict import load_artifacts
from src.preprocess import make_features
from src.explain import force_plot_png


app = Flask(__name__)

CFG = load_config(os.environ.get("TITANIC_CFG", "configs/default.yaml"))
ARTIFACTS = None


def get_artifacts():
    global ARTIFACTS
    if ARTIFACTS is None:
        ARTIFACTS = load_artifacts(CFG["artifacts"]["model_path"])
    return ARTIFACTS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return {"status": "ok"}


@app.route("/predict", methods=["POST"])
def predict():
    art = get_artifacts()
    record = {
        "Pclass": int(request.form["Pclass"]),
        "Sex": request.form["Sex"],
        "Age": float(request.form["Age"]),
        "SibSp": int(request.form["SibSp"]),
        "Parch": int(request.form["Parch"]),
        "Fare": float(request.form["Fare"]),
        "Embarked": request.form["Embarked"],
    }
    df = pd.DataFrame([record])
    df = make_features(df)
    feat_cols = art["feat_cols"]
    df = df.reindex(columns=feat_cols, fill_value=0)
    X = df.values.astype(float)
    proba = float(art["model"].predict_proba(X)[0][1])
    pred = int(proba >= 0.5)
    pred_label = "Survived" if pred == 1 else "Did not survive"
    img = force_plot_png(art["model"], feat_cols, X[0])
    return render_template(
        "result.html",
        pred_label=pred_label,
        proba=proba,
        shap_img=img,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
