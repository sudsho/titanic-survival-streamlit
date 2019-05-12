"""Flask app for titanic survival prediction."""
import os
from flask import Flask, render_template, request
from src.utils import load_config
from src.predict import load_artifacts, predict_one


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
    pred, proba = predict_one(art["model"], art["feat_cols"], record)
    pred_label = "Survived" if pred == 1 else "Did not survive"
    return render_template(
        "result.html",
        pred_label=pred_label,
        proba=proba,
        shap_img=None,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
