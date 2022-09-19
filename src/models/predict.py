import pickle
from datetime import datetime

import os
import mlflow

import pandas as pd

import boto3
import pickle
from src.features.build_features import preprocess_test
from src.features.build_features import all_columns


DEFAULT_RUN_ID = '21cf301e4e7f459bae86218626159897'
RUN_ID = os.getenv('RUN_ID', DEFAULT_RUN_ID)
BUCKET_NAME = os.getenv('BUCKET_NAME', 'mlflow-enkidupal-experiments')
DEFAULT_EXPERIMENT_NUMBER = '1'
EXPERIMENT_NUMBER = os.getenv('EXPERIMENT_NUMBER', DEFAULT_EXPERIMENT_NUMBER)

model_s3_path = f's3://{BUCKET_NAME}/{EXPERIMENT_NUMBER}/{RUN_ID}/artifacts/model/'

model = mlflow.pyfunc.load_model(model_s3_path)

s3client = boto3.client('s3')

response = s3client.get_object(Bucket=BUCKET_NAME, Key=f'{EXPERIMENT_NUMBER}/{RUN_ID}/artifacts/preprocessor/preprocessor.b')

body = response['Body'].read()
preprocessor = pickle.loads(body)


def load_file_from_s3():
    pass

def read_data(filename):
    """Return processed features dict and target."""

    # Load dataset
    if filename.endswith('parquet'):
        df = pd.read_parquet(filename, columns=all_columns)
    elif filename.endswith('csv'):
        df = pd.read_csv(filename)
    else:
        raise "Error: not supported file format."

    return df

def create_features(json):
    df = pd.json_normalize(json, meta=all_columns)
    return df


def calculate_predict(df: pd.DataFrame):
    X_test = preprocess_test(df, preprocessor)

    print(model)

    predictions = model.predict(X_test).tolist()
    decisions = [1 if p >= 0.5 else 0 for p in predictions ]
    result = {
        'predictions': list(predictions),
        'decisions': list(decisions)
    }

    return result




