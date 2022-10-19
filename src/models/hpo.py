import json
import time
import sys
import os

sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))

# from toolz import compose
import pickle
import logging
import argparse

import numpy as np
import mlflow
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, hp, tpe, fmin, space_eval
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.data.read import read_data
from src.data.download import run as download_run
from src.data.download import get_datapath
from src.util.json_encoder import NpEncoder
from src.features.build_features import preprocess_all, get_all_columns

# from hyperopt.pyll import scope

sys.path.append('../')


MLFLOW_DEFAULT_TRACKING_URI = "http://0.0.0.0:5000"
MLFLOW_DEFAULT_EXPERIMENT = "titanic-experiment"


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def split_train_read(filename: str, val_size=0.2, random_state=42):
    df_train_full = read_data(filename, columns=get_all_columns())

    df_train, df_val = train_test_split(df_train_full, test_size=val_size, random_state=random_state)
    return df_train, df_val


def dump_pickle(obj, filename):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


def fit_rfc_model(params, X_train, y_train):
    n_estimators = int(params['n_estimators'])
    max_depth = int(params['max_depth'])
    min_samples_leaf = int(params['min_samples_leaf'])
    min_samples_split = int(params['min_samples_split'])
    criterion = params['criterion']
    max_features = params['max_features']
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        criterion=criterion,
        max_features=max_features,
        n_jobs=1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_model_rfc_search(
    X_train, y_train, X_valid, y_val, X_train_full, max_evals, models_path
):  # pylint: disable=unused-argument
    mlflow.sklearn.autolog(disable=True)

    def objective(params):
        with mlflow.start_run(nested=True):
            mlflow.set_tag("model", "rfc")
            mlflow.log_params(params)

            # Train model and record run time
            start_time = time.time()
            model = fit_rfc_model(params, X_train, y_train)
            run_time = time.time() - start_time

            mlflow.log_metric('runtime', run_time)

            y_pred = model.predict(X_valid)
            auc_score_calculated: float = roc_auc_score(y_val, y_pred)
            mlflow.log_metric("auc_score", auc_score_calculated)
            # acc = accuracy_score(y_val, y_pred)
            # mlflow.log_metric("accuracy", acc)

        return {'loss': auc_score_calculated * (-1), 'status': STATUS_OK}

    search_space = {
        'n_estimators': hp.randint('n_estimators', 100, 1000),
        'max_depth': hp.randint('max_depth', 5, 40),
        'min_samples_split': hp.uniform('min_samples_split', 2, 8),
        'min_samples_leaf': hp.randint('min_samples_leaf', 1, 10),
        'criterion': hp.choice('criterion', ['gini', 'entropy']),
        'max_features': hp.choice('max_features', ['sqrt', 'log2']),
    }

    best_params = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=int(max_evals),
        rstate=np.random.default_rng(42),
        # early_stop_fn=hyperopt.early_stop.no_progress_loss(20)
        trials=Trials(),
    )
    print(f'best_params: {best_params}')

    sp_eval = space_eval(search_space, best_params)
    print(f'sp_eval: {sp_eval}')

    with open(f'{models_path}/best_params.b', "wb") as f_out:
        pickle.dump(best_params, f_out)
    mlflow.log_artifact(f'{models_path}/best_params.b', artifact_path="best_params")

    # dict_best_params = dict(best_params)
    # print(f'dict_best_params: {dict_best_params}')

    # mlflow.log_dict(dict_best_params, "best_params.json")

    mlflow.sklearn.autolog()
    best_model = fit_rfc_model(sp_eval, X_train, y_train)
    mlflow.sklearn.log_model(best_model, artifact_path="models_pickle")

    y_pred = best_model.predict(X_valid)
    auc_score = roc_auc_score(y_val, y_pred)
    mlflow.log_metric("valid_auc_score", auc_score)

    acc = accuracy_score(y_val, y_pred)
    mlflow.log_metric("accuracy", acc)

    return best_params


def fit_booster_model(params, train, valid, early_stopping_rounds=3, num_boost_round=1000):
    es = xgb.callback.EarlyStopping(
        rounds=early_stopping_rounds,
        min_delta=1e-3,
        save_best=True,
        maximize=True,
        data_name="validation",
        metric_name="auc",
    )

    booster = xgb.train(
        params=params,
        dtrain=train,
        num_boost_round=num_boost_round,
        evals=[(valid, 'validation')],
        # early_stopping_rounds=early_stopping_rounds,
        verbose_eval=5,
        callbacks=[es],
    )
    return booster


def fit_booster_model_no_validation(params, train, num_boost_round=1000):
    booster = xgb.train(
        params=params,
        dtrain=train,
        num_boost_round=num_boost_round,
        # early_stopping_rounds=early_stopping_rounds,
        verbose_eval=5,
    )
    return booster


class SaveBestModel(xgb.callback.TrainingCallback):
    def __init__(self, cvboosters):
        self._cvboosters = cvboosters

    def after_training(self, model):
        self._cvboosters[:] = [cvpack.bst for cvpack in model.cvfolds]
        return model


def fit_cv_booster_model(params, full_train, nfold=5, early_stopping_rounds=10):
    mlflow.xgboost.autolog()
    cvboosters = []
    eval_history = xgb.cv(
        params=params,
        dtrain=full_train,
        nfold=nfold,
        num_boost_round=1000,
        metrics=["auc"],
        shuffle=True,
        verbose_eval=5,
        maximize=True,
        early_stopping_rounds=early_stopping_rounds,
        callbacks=[SaveBestModel(cvboosters)],
    )
    return cvboosters, eval_history


def train_model_xgboost_search(train, valid, y_val, train_full, max_evals, models_path):
    mlflow.xgboost.autolog(disable=True)

    train_eval_params = {'objective': 'binary:logistic', 'eval_metric': 'auc'}
    active_run = mlflow.active_run()

    def objective(params):
        with mlflow.start_run(nested=True):
            mlflow.set_tag("model", "xgboost")
            mlflow.set_tag("kind", "hpo")
            mlflow.set_tag("uber_run_id", active_run.info.run_id)

            mlflow.log_params(params)
            # Train model and record run time
            start_time = time.time()

            booster = fit_booster_model(params, train, valid, early_stopping_rounds=10)

            run_time = time.time() - start_time
            mlflow.log_metric('runtime', run_time)

            y_pred = booster.predict(valid)
            auc_score_calculated: float = roc_auc_score(y_val, y_pred)
            mlflow.log_metric("auc_score", auc_score_calculated)

        return {'loss': auc_score_calculated * (-1), 'status': STATUS_OK, 'booster': booster.attributes()}

    search_space = {
        # hp.choice('max_depth', np.arange(1, 14, dtype=int))
        'learning_rate': hp.loguniform('learning_rate', -7, 0),
        # 'n_round': scope.int(hp.quniform('n_round', 200, 3000, 100)),
        'max_depth': hp.choice('max_depth', np.arange(1, 6, dtype=int)),
        'min_child_weight': hp.loguniform('min_child_weight', -2, 3),
        'subsample': hp.uniform('subsample', 0.5, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
        'gamma': hp.loguniform('gamma', -10, 10),
        'alpha': hp.loguniform('alpha', -10, 10),
        'lambda': hp.loguniform('lambda', -10, 10),
        # 'objective': 'binary:logistic',
        # 'eval_metric': 'auc',
        'seed': 42,
    }
    search_space = search_space | train_eval_params

    # spark_trials = SparkTrials()
    trials = Trials()

    best_hyperparams = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=int(max_evals),
        # early_stop_fn=hyperopt.early_stop.no_progress_loss(10),
        trials=trials,
        max_queue_len=5,
    )

    print(f'best_hyperparams: {best_hyperparams}')
    # mlflow.log_dict(best_params, "best_params.json")
    best_hyperparams_extra = best_hyperparams.copy()
    best_hyperparams_extra = best_hyperparams_extra | train_eval_params

    with open(f'{models_path}/best_hyperparams.json', 'w', encoding='utf-8') as f_out:
        json.dump(best_hyperparams_extra, f_out, ensure_ascii=False, indent=4, cls=NpEncoder)
    mlflow.log_artifact(f'{models_path}/best_hyperparams.json')

    # obtain idea numer of iteration for final training, using K-Fold Training
    with mlflow.start_run(nested=True):
        mlflow.set_tag("model", "xgboost")
        mlflow.set_tag("kind", "cv")
        mlflow.set_tag("uber_run_id", active_run.info.run_id)
        cv_early_stopping_rounds = 5
        nfold = 5

        _, eval_history = fit_cv_booster_model(
            best_hyperparams_extra, train_full, nfold=nfold, early_stopping_rounds=cv_early_stopping_rounds
        )

        best_nrounds = eval_history.shape[0] - cv_early_stopping_rounds
        best_nrounds = int(best_nrounds / (1 - 1 / nfold))

        logging.info(
            f'besteval_history.shape[0]: {eval_history.shape[0]}, '
            f'cv_early_stopping_rounds: {cv_early_stopping_rounds}'
        )
        logging.info(f'best_nrounds: {best_nrounds}')

    # train final model with correct number of rounds, no Early stoppping full train dataset
    mlflow.xgboost.autolog()

    final_model = fit_booster_model_no_validation(best_hyperparams_extra, train_full, num_boost_round=best_nrounds)

    print(final_model)
    print(final_model.attributes())

    y_pred = final_model.predict(valid)
    auc_score = roc_auc_score(y_val, y_pred)
    mlflow.log_metric("valid_auc_score", auc_score)

    # acc = accuracy_score(y_val, y_pred)
    # mlflow.log_metric("accuracy", acc)

    return best_hyperparams


def run(data_root: str, mlflow_tracking_uri: str, mlflow_experiment: str, models_path: str, model: str, max_evals: str):
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    print(f"tracking URI: '{mlflow.get_tracking_uri()}'")
    mlflow.set_experiment(mlflow_experiment)

    with mlflow.start_run():
        datapath = get_datapath(data_root)
        download_run(data_root, 'titanic')

        external_train_path = datapath.get_train_file_path(datapath.external_train_dirpath)

        df_train_full = read_data(external_train_path, get_all_columns())
        # df_train, df_val = split_train_read(external_train_path, val_size=0.2, random_state=42)
        df_train, df_val = train_test_split(df_train_full, test_size=0.2, random_state=42)

        X_train, X_val, y_train, y_val, prep_pipeline = preprocess_all(df_train, df_val)
        # X_train_full, y_train_full, prep_pipeline_full = preprocess_valid(df_train_full)
        X_train_full = np.concatenate([X_train, X_val], axis=0)
        y_train_full = np.concatenate([y_train, y_val], axis=0)

        with open(f'{models_path}/preprocessor.b', "wb") as f_out:
            pickle.dump(prep_pipeline, f_out)
        mlflow.log_artifact(f'{models_path}/preprocessor.b', artifact_path="preprocessor")

        if model == 'xgboost':
            train = xgb.DMatrix(X_train, label=y_train)
            valid = xgb.DMatrix(X_val, label=y_val)
            train_full = xgb.DMatrix(X_train_full, label=y_train_full)

            train_model_xgboost_search(train, valid, y_val, train_full, max_evals, models_path)
        elif model == 'rfc':
            train_model_rfc_search(X_train, y_train, X_val, y_val, X_train_full, max_evals, models_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default='../../data', help="The location where the external data is downloaded")
    parser.add_argument("--models_path", default='../../models', help="models_path")
    parser.add_argument("--mlflow_tracking_uri", default=MLFLOW_DEFAULT_TRACKING_URI, help="Mlflow tracking uri")
    parser.add_argument("--mlflow_experiment", default="titanic-hpo", help="Mlflow experiment")
    parser.add_argument("--model", default="xgboost", help="Model for HPO")
    parser.add_argument("--max_evals", default="50", help="Maximum number of evaluations for HPO")

    args = parser.parse_args()

    run(args.data_root, args.mlflow_tracking_uri, args.mlflow_experiment, args.models_path, args.model, args.max_evals)
