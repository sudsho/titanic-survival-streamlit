# titanic-survival-streamlit

[![Build Status](https://travis-ci.org/sudsho/titanic-survival-streamlit.svg?branch=master)](https://travis-ci.org/sudsho/titanic-survival-streamlit)

Predict survival on the Titanic using the classic Kaggle dataset, with a small Flask web dashboard and SHAP explanations for each prediction.

> Note: the repo is named *streamlit* but the dashboard is built with Flask + matplotlib (Streamlit's first public release was Oct 2019, after this project was started).

## Quick start (runs offline)

No network or download needed. A small slice of the Kaggle Titanic train set is committed at `data/train.csv`, so the smoke trains and exercises the full predict path out of the box (the smoke falls back to a synthetic Titanic-schema dataframe if that CSV is ever removed).

```
python scripts/smoke.py
```

Real output on Python 3.11 (scikit-learn 1.8, shap 0.51):

```
============================================================
Titanic survival - offline smoke test
============================================================
[data] using bundled CSV: data/train.csv  (80 rows)

[train] fitting classifier: random_forest
[train] train accuracy : 0.938
[train] test accuracy  : 0.625
[train] test roc_auc   : 0.683
[train] feature cols   : ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone']
[train] saved model    : artifacts\model.pkl

[serve] predict_one(sample) -> pred=0  proba=0.0377
[serve] GET /health -> 200 {'status': 'ok'}
[serve] POST /predict -> 200 (rendered label + SHAP chart)

============================================================
SMOKE PASSED
============================================================
```

The smoke trains the classifier (prints accuracy and ROC AUC), then exercises both the `predict_one` helper and the Flask `POST /predict` route through the app's test client, so the make_features, prediction, and SHAP explanation path all run headless. It does not launch the web server. `make smoke` runs the same thing.

The metrics above come from the 80-row committed slice, so they are illustrative only; on the full Kaggle train set the RandomForest reaches roughly 0.82 5-fold CV accuracy.

## Problem

Given passenger info (age, sex, class, fare, etc.), predict the probability that they survived the sinking of the Titanic. Surface which features pushed the prediction up or down.

## Dataset

Kaggle Titanic competition train.csv (passenger manifest with `Survived` label). A small slice is committed at `data/train.csv` so the project runs out of the box; download the full file from kaggle for serious training.

## Approach

- preprocess (impute Age by Sex+Pclass median, Title from Name, FamilySize, encode Sex/Embarked, fill Fare)
- model: sklearn RandomForest (default), GradientBoosting and LogisticRegression also available via config
- evaluation: 5-fold CV accuracy + held-out test accuracy/precision/recall/F1/ROC AUC
- serving: Flask app with an HTML form, returns prediction + SHAP feature contribution chart (matplotlib PNG)

## Results

On the public Kaggle train split (5-fold CV) the RF gets around 0.82 accuracy with the engineered Title and FamilySize features. With logreg as a sanity check we get around 0.78.

## Setup

```
pip install -r requirements.txt
python -m src.train --config configs/default.yaml
python app.py
```

Open http://localhost:5000 in a browser.

## Deploy

Heroku-friendly. `Procfile` and `runtime.txt` included; with Heroku CLI:

```
heroku create titanic-survival-<your-handle>
git push heroku master
heroku open
```

A live demo (when up) lives at `https://titanic-survival-<your-handle>.herokuapp.com`. There's a Dockerfile too for container-based deploys.

## Tests

```
pytest -q
```

Travis CI runs the test suite on each push (Python 3.6 and 3.7).

## Layout

```
app.py                  Flask entry point
src/                    preprocess, model, train, predict, explain, eval, utils
configs/                YAML configs (default RF, gbm)
templates/, static/     HTML form + result page + CSS
tests/                  pytest unit tests
notebooks/              EDA, feature importance
data/train.csv          small slice of the Kaggle Titanic train set
Procfile, runtime.txt   Heroku deploy
Dockerfile, heroku.yml  Container deploy
.travis.yml             CI
```

## License

MIT.
