import argparse
import os
import pickle

import pandas as pd
from sklearn.feature_extraction import DictVectorizer

from sklearn.pipeline import make_pipeline

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

def get_preprocessing_config():
    transforms = []
    target = 'Survived'
    categorical = ['Sex', 'Pclass', 'Embarked', 'SibSp', 'Parch']
    numerical = ['Fare']

    return transforms, target, categorical, numerical

def create_preprocessing_pipeline():
    transforms, target, categorical, numerical = get_preprocessing_config()
    pipeline = make_pipeline(
        DictVectorizer()
    )
    return pipeline


def save_preprocessed(df: pd.DataFrame, path):
    df.to_csv(path)

def prepare_dictionaries(df: pd.DataFrame):

    transforms, target, categorical, numerical = get_preprocessing_config()

    dicts = df[categorical + numerical].to_dict(orient='records')
    return dicts

def preprocess_df(df: pd.DataFrame, transforms, categorical, numerical):
    """Return processed features dict and target."""

    # Apply in-between transformations
    df = compose(*transforms[::-1])(df)
    # For dict vectorizer: int = ignored, str = one-hot
    df[categorical] = df[categorical].fillna(-1).astype("category")
    return df

def preprocess_train(df_train):
    transforms, target, categorical, numerical = get_preprocessing_config()

    df_train = preprocess_df(df_train, transforms, categorical, numerical)

    #
    pipeline = create_preprocessing_pipeline()

    train_dicts = prepare_dictionaries(df_train)
    X_train = pipeline.fit_transform(train_dicts)

    #

    y_train = df_train[target].values

    return X_train, y_train, pipeline

def preprocess_valid(df_val, preprocessing_pipeline):
    transforms, target, categorical, numerical = get_preprocessing_config()

    df_val = preprocess_df(df_val, transforms, categorical, numerical)

    val_dicts = prepare_dictionaries(df_val)
    X_val = preprocessing_pipeline.transform(val_dicts)

    y_val = df_val[target].values

    return X_val, y_val

def preprocess_test(df_test, preprocessing_pipeline):
    transforms, target, categorical, numerical = get_preprocessing_config()

    df_test = preprocess_df(df_test, transforms, categorical, numerical)

    test_dicts = prepare_dictionaries(df_test)
    X_test = preprocessing_pipeline.transform(test_dicts)

    return X_test

def preprocess_all(df_train, df_val):
    X_train, y_train, prep_pipeline = preprocess_train(df_train)
    X_val, y_val = preprocess_valid(df_val, prep_pipeline)

    return X_train, X_val, y_train, y_val, prep_pipeline


def run(data_root: str, output_path: str):

    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        default='../../data',
        help="The location where the raw Titanic data is downloaded"
    )
    parser.add_argument(
        "--output_path",
        help="the location where the resulting files will be saved."
    )
    args = parser.parse_args()

    run(args.data_root, args.output_path)

if __name__ == '__main__':
    main()

