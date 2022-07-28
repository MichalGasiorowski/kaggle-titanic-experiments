import argparse
import os
import sys

import time

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

import xgboost as xgb

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

import mlflow

import pandas as pd
import numpy as np
from toolz import compose
import pickle

from sklearn.model_selection import train_test_split

MLFLOW_DEFAULT_TRACKING_URI="http://0.0.0.0:5000"
MLFLOW_DEFAULT_EXPERIMENT="titanic-experiment"

sys.path.append('../')

from src.data.download import run as download_run
from src.data.download import get_datapath as get_datapath
from src.data.download import DataPath

from src.features.build_features import preprocess_df

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

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

def split_train_read(filename: str, val_size=0.2, random_state=42):
    df_train_full = read_data(filename)

    df_train, df_val = train_test_split(df_train_full, test_size=val_size, random_state=random_state)
    return df_train, df_val

def dump_pickle(obj, filename):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)

def preprocess_df(df: pd.DataFrame, transforms, categorical, numerical):
    """Return processed features dict and target."""
    # Apply in-between transformations
    df = compose(*transforms[::-1])(df)
    # For dict vectorizer: int = ignored, str = one-hot
    df[categorical] = df[categorical].astype("category")

    return df

def preprocess_all(df_train, df_val):
    transforms = []
    target = 'Survived'
    categorical = ['Sex', 'Pclass', 'Embarked', 'SibSp', 'Parch']
    numerical = ['Fare']

    df_train = preprocess_df(df_train, transforms, categorical, numerical)
    df_val = preprocess_df(df_val, transforms, categorical, numerical)

    dv = DictVectorizer()
    train_dicts = df_train[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical + numerical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)

    y_train = df_train[target].values
    y_val = df_val[target].values

    return X_train, X_val, y_train, y_val, dv


def train_model_xgboost_search(train, valid, y_val):

    mlflow.xgboost.autolog(silent=True)

    def objective(params):
        with mlflow.start_run(nested=True):
            mlflow.set_tag("model", "xgboost")
            mlflow.log_params(params)
            # Train model and record run time
            start_time = time.time()

            booster = xgb.train(
                params=params,
                dtrain=train,
                num_boost_round=1000,
                evals=[(valid, 'validation')],
                early_stopping_rounds=50
            )
            run_time = time.time() - start_time
            mlflow.log_metric('runtime', run_time)

            y_pred = booster.predict(valid)
            auc_score = roc_auc_score(y_val, y_pred)
            mlflow.log_metric("auc_score", auc_score)

        return {'loss': -auc_score, 'status': STATUS_OK, 'booster': booster.attributes()}

    search_space = {
        'learning_rate': hp.loguniform('learning_rate', -7, 0),
        'max_depth': scope.int(hp.uniform('max_depth', 1, 100)),
        'min_child_weight': hp.loguniform('min_child_weight', -2, 3),
        'subsample': hp.uniform('subsample', 0.5, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
        'gamma': hp.loguniform('gamma', -10, 10),
        'alpha': hp.loguniform('alpha', -10, 10),
        'lambda': hp.loguniform('lambda', -10, 10),
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'seed': 42,
    }

    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=25,
        trials=Trials()
    )
    print(f'best_result: {best_result}')
    return best_result

def run(data_root: str, mlflow_tracking_uri: str, mlflow_experiment: str, models_path: str):
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    print(f"tracking URI: '{mlflow.get_tracking_uri()}'")
    mlflow.set_experiment(mlflow_experiment)
    mlflow.sklearn.autolog()

    with mlflow.start_run():
        datapath = get_datapath(data_root)
        download_run(data_root, 'titanic')

        external_train_path=datapath.get_train_file_path(datapath._external_train_dirpath)

        df_train, df_val = split_train_read(external_train_path, val_size=0.2, random_state=42)

        X_train, X_val, y_train, y_val, dv = preprocess_all(df_train, df_val)
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        train_model_xgboost_search(train, valid, y_val)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        default='../../data',
        help="The location where the external Titanic data is downloaded"
    )
    parser.add_argument(
        "--models_path",
        default='../../models',
        help="models_path"
    )
    parser.add_argument(
        "--mlflow_tracking_uri",
        default=MLFLOW_DEFAULT_TRACKING_URI,
        help="Mlflow tracking uri"
    )
    parser.add_argument(
        "--mlflow_experiment",
        default="titanic-hpo",
        help="Mlflow experiment"
    )
    parser.add_argument(
        "--model",
        default="xgboost",
        help="Model for HyperOptimization"
    )

    args = parser.parse_args()

    run(args.data_root, args.mlflow_tracking_uri, args.mlflow_experiment, args.models_path)
