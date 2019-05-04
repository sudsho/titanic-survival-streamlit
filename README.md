# titanic-survival-streamlit

Predict survival on the Titanic using the classic Kaggle dataset.

## Problem

Given passenger info (age, sex, class, fare, etc.), predict whether they survived the sinking of the Titanic.

## Dataset

Kaggle Titanic competition dataset. A small copy of `train.csv` is included in `data/`.

## Plan

- preprocess (impute missing Age, encode Sex/Embarked, fill Fare)
- train a sklearn classifier (RandomForest)
- serve predictions via a small Flask web app
- show SHAP feature contributions per prediction (force plot rendered to PNG)

## Setup

```
pip install -r requirements.txt
python -m src.train
python app.py
```

Open http://localhost:5000.
