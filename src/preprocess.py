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


def encode_sex(df):
    df = df.copy()
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    return df


def encode_embarked(df):
    df = df.copy()
    df["Embarked"] = df["Embarked"].fillna("S")
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    return df


def fill_fare(df):
    df = df.copy()
    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    return df


def add_family_size(df):
    df = df.copy()
    if "SibSp" in df.columns and "Parch" in df.columns:
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df


def make_features(df):
    df = basic_clean(df)
    df = fill_age(df)
    df = encode_sex(df)
    df = encode_embarked(df)
    df = fill_fare(df)
    df = add_family_size(df)
    return df
