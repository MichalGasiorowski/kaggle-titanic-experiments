import pickle
import argparse

import numpy as np
import pandas as pd
from toolz import compose
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline

# from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer  # StandardScaler,
from sklearn.feature_extraction import DictVectorizer

DEFAULT_TARGET = 'Survived'
DEFAULT_CATEGORICAL = ['Sex', 'Pclass', 'Embarked', 'SibSp', 'Parch']
DEFAULT_NUMERICAL = ['Fare', 'Age']
ID_COLUMN = 'PassengerId'


DEFAULT_ALL_COLUMNS = [ID_COLUMN] + DEFAULT_CATEGORICAL + DEFAULT_NUMERICAL
DEFAULT_TRAIN_ALL_COLUMNS = DEFAULT_CATEGORICAL + DEFAULT_NUMERICAL


def get_id_column():
    return ID_COLUMN


def get_all_columns():
    return DEFAULT_TRAIN_ALL_COLUMNS


def get_all_test_columns():
    return DEFAULT_TRAIN_ALL_COLUMNS


def dump_pickle(obj, filename):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


def extract_target(data: pd.DataFrame, target=DEFAULT_TARGET):
    targets = data[target].values
    return targets


def get_preprocessing_config():
    transforms = []
    target = DEFAULT_TARGET
    categorical = DEFAULT_CATEGORICAL
    numerical = DEFAULT_NUMERICAL

    return transforms, target, categorical, numerical


# try to use FeatureUnion to make more complicated pipeline
# https://github.com/autoreleasefool/rumoureval/blob/042e2a01142391c32f6c3c67f51316ec3fac39e0/rumoureval/classification/sdqc.py
def create_preprocessing_pipeline_for_dict():
    transforms, target, categorical, numerical = get_preprocessing_config()  # pylint: disable=unused-variable)
    pipeline = make_pipeline(DictVectorizer())
    return pipeline


def create_preprocessing_pipeline_for_df():
    transforms, target, categorical, numerical = get_preprocessing_config()  # pylint: disable=unused-variable)

    # numerical pipeline
    cat_pipe = Pipeline(
        [
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False)),
        ]
    )

    # Define numerical pipeline
    num_pipe = Pipeline(
        [
            ("imputation_mean", SimpleImputer(missing_values=np.nan, strategy="mean")),
            ('binner', KBinsDiscretizer(n_bins=6))
            # ('scaler', StandardScaler())
        ]
    )

    preprocessor = ColumnTransformer([('cat', cat_pipe, categorical), ('num', num_pipe, numerical)])

    pipe = Pipeline(
        [
            # ('dv', DictVectorizer()),
            ('preprocessor', preprocessor)
        ]
    )

    return pipe


# def prepare_dictionaries(df: pd.DataFrame):
#
#     transforms, target, categorical, numerical = get_preprocessing_config() #pylint: disable=unused-variable
#
#     dicts = df[categorical + numerical].to_dict(orient='records')
#     return dicts


def preprocess_df(df: pd.DataFrame, transforms, categorical, numerical):  # pylint: disable=unused-argument

    """Return processed features dict and target."""

    # Apply in-between transformations
    df = compose(*transforms[::-1])(df)
    # For dict vectorizer: int = ignored, str = one-hot
    df[categorical] = df[categorical].fillna("-1").astype("str")

    return df


def preprocess_train(df_train):
    transforms, target, categorical, numerical = get_preprocessing_config()

    df_train = preprocess_df(df_train, transforms, categorical, numerical)

    #
    pipeline = create_preprocessing_pipeline_for_df()
    print(pipeline)
    X_train = pipeline.fit_transform(df_train)

    #

    y_train = df_train[target].values

    return X_train, y_train, pipeline


def preprocess_valid(df_val, preprocessing_pipeline):
    transforms, target, categorical, numerical = get_preprocessing_config()  # pylint: disable=unused-variable

    df_val = preprocess_df(df_val, transforms, categorical, numerical)

    # val_dicts = prepare_dictionaries(df_val)
    X_val = preprocessing_pipeline.transform(df_val)

    y_val = df_val[target].values

    return X_val, y_val


def preprocess_test(df_test, preprocessing_pipeline):
    transforms, target, categorical, numerical = get_preprocessing_config()  # pylint: disable=unused-variable)

    df_test = preprocess_df(df_test, transforms, categorical, numerical)

    # test_dicts = prepare_dictionaries(df_test)
    X_test = preprocessing_pipeline.transform(df_test)

    return X_test


def preprocess_all(df_train, df_val):
    X_train, y_train, prep_pipeline = preprocess_train(df_train)
    X_val, y_val = preprocess_valid(df_val, prep_pipeline)

    return X_train, X_val, y_train, y_val, prep_pipeline


def run(data_root: str, output_path: str):
    print(f'data_root: {data_root}, output_path: {output_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root", default='../../data', help="The location where the raw Titanic data is downloaded"
    )
    parser.add_argument("--output_path", help="the location where the resulting files will be saved.")
    args = parser.parse_args()

    run(args.data_root, args.output_path)


if __name__ == '__main__':
    main()
