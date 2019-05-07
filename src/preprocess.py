"""Preprocessing for titanic data."""
import pandas as pd
import numpy as np


def load_csv(path):
    df = pd.read_csv(path)
    return df


def basic_clean(df):
    # drop columns we won't use
    cols_to_drop = ["PassengerId", "Name", "Ticket", "Cabin"]
    for c in cols_to_drop:
        if c in df.columns:
            df = df.drop(columns=[c])
    return df


def fill_age(df):
    # use median by sex+pclass for plausibility
    if "Age" not in df.columns:
        return df
    df = df.copy()
    df["Age"] = df.groupby(["Sex", "Pclass"])["Age"].transform(
        lambda x: x.fillna(x.median())
    )
    # any leftovers, use overall median
    df["Age"] = df["Age"].fillna(df["Age"].median())
    return df
