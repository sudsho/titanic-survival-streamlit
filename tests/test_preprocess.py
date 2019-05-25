import pandas as pd
import numpy as np
from src.preprocess import (
    basic_clean, fill_age, encode_sex, encode_embarked,
    fill_fare, add_family_size, make_features,
)


def small_df():
    return pd.DataFrame({
        "PassengerId": [1, 2, 3],
        "Survived": [0, 1, 1],
        "Pclass": [3, 1, 2],
        "Name": ["a", "b", "c"],
        "Sex": ["male", "female", "female"],
        "Age": [22.0, np.nan, 30.0],
        "SibSp": [1, 0, 0],
        "Parch": [0, 0, 1],
        "Ticket": ["x", "y", "z"],
        "Fare": [7.0, 71.0, np.nan],
        "Cabin": [None, "C85", None],
        "Embarked": ["S", "C", None],
    })


def test_basic_clean_drops_cols():
    df = basic_clean(small_df())
    for c in ("PassengerId", "Name", "Ticket", "Cabin"):
        assert c not in df.columns


def test_fill_age_fills_nans():
    df = fill_age(small_df())
    assert df["Age"].isna().sum() == 0


def test_encode_sex_int():
    df = encode_sex(small_df())
    assert set(df["Sex"].unique()).issubset({0, 1})


def test_encode_embarked_handles_nulls():
    df = encode_embarked(small_df())
    assert df["Embarked"].isna().sum() == 0


def test_fill_fare_fills_nans():
    df = fill_fare(small_df())
    assert df["Fare"].isna().sum() == 0


def test_add_family_size():
    df = add_family_size(small_df())
    assert "FamilySize" in df.columns
    assert df["FamilySize"].iloc[0] == 2


def test_make_features_no_nans():
    df = make_features(small_df())
    assert df.isna().sum().sum() == 0
