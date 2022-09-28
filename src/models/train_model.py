import os
import sys
import json
import pickle
import logging
import argparse

import boto3
import numpy as np
import mlflow
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.data.read import read_data
from src.data.download import DataPath
from src.data.download import run as download_run
from src.data.download import get_datapath
from src.features.build_features import preprocess_all, get_all_columns

MLFLOW_DEFAULT_TRACKING_URI = "http://0.0.0.0:5000"
MLFLOW_DEFAULT_EXPERIMENT = "titanic-train-experiment"

sys.path.append('../')


DEFAULT_RUN_ID = '991e896e9ae742cca6c600b007223523'
RUN_ID = os.getenv('RUN_ID', DEFAULT_RUN_ID)

DEFAULT_BUCKET_NAME = os.getenv('BUCKET_NAME', 'mlflow-enkidupal-experiments')
DEFAULT_KEY = f'1/{RUN_ID}/artifacts/best_hyperparams.json'


def get_params_s3(bucket, key):

    s3client = boto3.client('s3')
    response = s3client.get_object(Bucket=bucket, Key=key)

    body = response['Body'].read()

    params = json.loads(body)
    print(params)
    return params


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def split_train_read(filename: str, val_size=0.2, random_state=42):
    df_train_full = read_data(filename, get_all_columns())

    df_train, df_val = train_test_split(df_train_full, test_size=val_size, random_state=random_state)
    return df_train, df_val


def dump_pickle(obj, filename):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


def fit_booster_model(params, train, valid):
    booster = xgb.train(
        params=params,
        dtrain=train,
        num_boost_round=2000,
        evals=[(valid, 'validation')],
        early_stopping_rounds=50,
        verbose_eval=10,
    )
    return booster


def train_xgb(train, valid, y_val, hyper_params):
    mlflow.xgboost.autolog()
    final_model = fit_booster_model(hyper_params, train, valid)

    y_pred = final_model.predict(valid)
    auc_score = roc_auc_score(y_val, y_pred)
    mlflow.log_metric("valid_auc_score", auc_score)

    return final_model


def run_train(
    datapath: DataPath,
    models_path: str,
    model_type: str,
    hyper_params_path: str,
    hyper_params_bucket_name: str,
    hyper_params_key: str,
):
    external_train_path = datapath.get_train_file_path(datapath.external_train_dirpath)

    df_train, df_val = split_train_read(external_train_path, val_size=0.2, random_state=42)

    X_train, X_val, y_train, y_val, prep_pipeline = preprocess_all(df_train, df_val)

    with open(f'{models_path}/preprocessor.b', "wb") as f_out:
        pickle.dump(prep_pipeline, f_out)
    mlflow.log_artifact(f'{models_path}/preprocessor.b', artifact_path="preprocessor")

    logging.info(f'hyper_params_path: {hyper_params_path}')
    hyper_params = get_params_s3(hyper_params_bucket_name, hyper_params_key)

    if model_type == 'xgboost':
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)
        # train_full = xgb.DMatrix(X_train_full, label=y_train_full)

        final_model = train_xgb(train, valid, y_val, hyper_params)
        return final_model
    if model_type == 'rfc':
        mlflow.sklearn.autolog()
        final_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

        final_model.fit(X_train, y_train)

        mlflow.sklearn.log_model(final_model, artifact_path="models_pickle")

        y_pred = final_model.predict(X_val)

        accuracy = np.round(accuracy_score(y_val, y_pred), 4)
        mlflow.log_metric("accuracy", accuracy)

        print(accuracy)
        return final_model

    raise TypeError(f'Unrecognized model type for training: {model_type}')


def run(
    data_root: str,
    mlflow_tracking_uri: str,
    mlflow_experiment: str,
    models_path: str,
    model: str,
    hyper_params_path: str,
    hyper_params_bucket_name: str,
    hyper_params_key: str,
):
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    print(f"tracking URI: '{mlflow.get_tracking_uri()}'")
    mlflow.set_experiment(mlflow_experiment)

    with mlflow.start_run():
        datapath = get_datapath(data_root)
        download_run(data_root, 'titanic')

        run_train(
            datapath=datapath,
            models_path=models_path,
            model_type=model,
            hyper_params_path=hyper_params_path,
            hyper_params_bucket_name=hyper_params_bucket_name,
            hyper_params_key=hyper_params_key,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root", default='../../data', help="The location where the external Titanic data is downloaded"
    )
    parser.add_argument("--models_path", default='../../models', help="models_path")
    parser.add_argument("--mlflow_tracking_uri", default=MLFLOW_DEFAULT_TRACKING_URI, help="Mlflow tracking uri")
    parser.add_argument("--mlflow_experiment", default=MLFLOW_DEFAULT_EXPERIMENT, help="Mlflow experiment")
    parser.add_argument("--model", default="xgboost", help="Model for Training")
    parser.add_argument("--hyper_params_path", default='', help="Hyper Params Path")
    parser.add_argument("--hyper_params_bucket_name", default=DEFAULT_BUCKET_NAME, help="Hyper Params Bucket Name")
    parser.add_argument("--hyper_params_key", default=DEFAULT_KEY, help="Hyper Parmas for Training")

    args = parser.parse_args()

    run(
        data_root=args.data_root,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment=args.mlflow_experiment,
        models_path=args.models_path,
        model=args.model,
        hyper_params_path=args.hyper_params_path,
        hyper_params_bucket_name=args.hyper_params_bucket_name,
        hyper_params_key=args.hyper_params_key,
    )
