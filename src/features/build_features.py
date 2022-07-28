import argparse
import os
import pickle

import pandas as pd
from sklearn.feature_extraction import DictVectorizer

from toolz import compose

from src.data.download import get_datapath as get_datapath
from src.data.download import DataPath


def dump_pickle(obj, filename):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


def read_data(filename):
    """Return processed features dict and target."""

    # Load dataset
    if filename.endswith('parquet'):
        df = pd.read_parquet(filename)
    elif filename.endswith('csv'):
        df = pd.read_csv(filename)
    else:
        raise "Error: not supported file format."

    return df

def extract_target(data: pd.DataFrame, target="Survived"):
    targets = data[target].values
    return targets

def save_preprocessed(df: pd.DataFrame, path):
    df.to_csv(path)

def preprocess_df(df: pd.DataFrame, transforms, categorical, numerical):
    """Return processed features dict and target."""
    # Apply in-between transformations
    df = compose(*transforms[::-1])(df)
    # For dict vectorizer: int = ignored, str = one-hot
    df[categorical] = df[categorical].astype(str)

    return df

def create_feature_dict(df: pd.DataFrame, categorical, numerical):
    # Convert dataframe to feature dictionaries
    return df[categorical + numerical].to_dict(orient='records')


def run(data_root: str, data_path: str):
    transforms = []
    target = 'Survived'
    categorical = ['Sex', 'Pclass', 'Embarked', 'SibSp', 'Parch']
    numerical = ['Fare']

    df = read_data(data_path)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        default='../../data',
        help="The location where the raw Titanic data is downloaded"
    )
    parser.add_argument(
        "--data_path",
        help="the location where the resulting files will be saved."
    )
    args = parser.parse_args()

    run(args.data_root, args.data_path)

if __name__ == '__main__':
    main()

