import os
import pickle

import boto3
import numpy as np
import mlflow
import pandas as pd

from src.data.read import load_file_from_s3
import src.features.build_features as build_features


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

ID_COLUMN = build_features.get_id_column()
TARGET_COLUMN = build_features.get_target_column()
PREDICTION_COLUMN = 'Prediction'

def get_model():
    return model


def get_preprocessor():
    return preprocessor


def create_features(json):
    df = pd.json_normalize(json, meta=build_features.get_all_test_columns())
    enrich_df_with_id(df)
    return df


def create_features_for_s3_path(s3_path):
    df = load_file_from_s3(s3_path, columns=build_features.get_all_test_columns())
    enrich_df_with_id(df)
    return df


def calculate_predict_old(df: pd.DataFrame):
    X_test = build_features.preprocess_test(df, preprocessor)

    print(model)

    predictions_np = model.predict(X_test)
    predictions = predictions_np.tolist()

    decisions = [1 if p >= 0.5 else 0 for p in predictions]
    result = {PREDICTION_COLUMN: list(predictions), TARGET_COLUMN: list(decisions)}

    return result


def calculate_predict_df(df: pd.DataFrame):
    X_test = build_features.preprocess_test(df, preprocessor)
    passengerId_np = df.loc[:, ID_COLUMN].to_numpy()
    print(model)

    predictions_np = model.predict(X_test)
    predictions_np_added = np.c_[
        passengerId_np, predictions_np, np.vectorize(lambda x: 1 if x >= 0.5 else 0)(predictions_np)
    ]

    predictions_df = pd.DataFrame(data=predictions_np_added, columns=[ID_COLUMN, PREDICTION_COLUMN, TARGET_COLUMN])
    predictions_df = predictions_df.astype({ID_COLUMN: 'int32', PREDICTION_COLUMN: 'float32', TARGET_COLUMN: 'int32'})
    return predictions_df


def calculate_predict_dict(df: pd.DataFrame):
    predictions_df = calculate_predict_df(df)
    df_dict = predictions_df.to_dict('records')

    return df_dict


def enrich_df_with_id(df: pd.DataFrame):
    if ID_COLUMN not in df:
        df[ID_COLUMN] = -1
    else:
        df[ID_COLUMN].fillna(value=-1, inplace=True)
