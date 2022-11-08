from io import BytesIO

import boto3
import pandas as pd

DEFAULT_TARGET = 'Survived'
DEFAULT_CATEGORICAL = ('Sex', 'Pclass', 'Embarked', 'SibSp', 'Parch')
DEFAULT_NUMERICAL = ('Fare', "Age")

DEFAULT_ALL_COLUMNS = DEFAULT_CATEGORICAL + DEFAULT_NUMERICAL


def read_data(filename, columns=DEFAULT_ALL_COLUMNS):
    """Return processed features dict and target."""

    # Load dataset
    if filename.endswith('parquet'):
        df = pd.read_parquet(filename, columns=columns)
    elif filename.endswith('csv'):
        df = pd.read_csv(filename)
    else:
        raise TypeError(f'Error: not supported file format for filename: {filename}')

    return df


def split_s3_path(s3_path):
    path_parts = s3_path.replace("s3://", "").split("/")
    bucket = path_parts.pop(0)
    key = "/".join(path_parts)
    return bucket, key


def load_file_from_s3(s3_path, columns=DEFAULT_ALL_COLUMNS):
    bucket_name, key = split_s3_path(s3_path)
    s3 = boto3.resource('s3')
    obj = s3.Object(bucket_name, key)
    with BytesIO(obj.get()['Body'].read()) as bio:
        if key.endswith('parquet'):
            df = pd.read_parquet(bio, columns=columns)
        elif key.endswith('csv'):
            df = pd.read_csv(bio)
        else:
            raise TypeError(f'Error: not supported file format for filename: {key}')
    return df
