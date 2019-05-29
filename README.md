# titanic-survival-streamlit

[![Build Status](https://travis-ci.org/sudsho/titanic-survival-streamlit.svg?branch=master)](https://travis-ci.org/sudsho/titanic-survival-streamlit)

Predict survival on the Titanic using the classic Kaggle dataset, with a small Flask web dashboard and SHAP explanations for each prediction.

> Note: the repo is named *streamlit* but the dashboard is built with Flask + matplotlib (Streamlit's first public release was Oct 2019, after this project was started).

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
```
