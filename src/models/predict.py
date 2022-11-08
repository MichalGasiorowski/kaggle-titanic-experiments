import os
import pickle

import boto3
import mlflow
import pandas as pd

from src.data.read import load_file_from_s3
from src.features.build_features import get_all_columns, preprocess_test

DEFAULT_RUN_ID = '21cf301e4e7f459bae86218626159897'
RUN_ID = os.getenv('RUN_ID', DEFAULT_RUN_ID)
BUCKET_NAME = os.getenv('BUCKET_NAME', 'mlflow-enkidupal-experiments')
DEFAULT_EXPERIMENT_NUMBER = '1'
EXPERIMENT_NUMBER = os.getenv('EXPERIMENT_NUMBER', DEFAULT_EXPERIMENT_NUMBER)

model_s3_path = f's3://{BUCKET_NAME}/{EXPERIMENT_NUMBER}/{RUN_ID}/artifacts/model/'

model = mlflow.pyfunc.load_model(model_s3_path)

s3client = boto3.client('s3')

response = s3client.get_object(
    Bucket=BUCKET_NAME, Key=f'{EXPERIMENT_NUMBER}/{RUN_ID}/artifacts/preprocessor/preprocessor.b'
)

body = response['Body'].read()
preprocessor = pickle.loads(body)


def create_features(json):
    df = pd.json_normalize(json, meta=get_all_columns())
    return df


def create_features_for_s3_path(s3_path):
    df = load_file_from_s3(s3_path, columns=get_all_columns())
    return df


def calculate_predict(df: pd.DataFrame):
    X_test = preprocess_test(df, preprocessor)

    print(model)

    predictions = model.predict(X_test).tolist()
    decisions = [1 if p >= 0.5 else 0 for p in predictions]
    result = {'predictions': list(predictions), 'decisions': list(decisions)}

    return result
