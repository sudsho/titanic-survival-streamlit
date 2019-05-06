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
