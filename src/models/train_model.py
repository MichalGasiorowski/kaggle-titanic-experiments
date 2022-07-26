import argparse
import os

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import mlflow

import pandas as pd
import numpy as np
from toolz import compose
import pickle

from sklearn.model_selection import train_test_split

MLFLOW_DEFAULT_TRACKING_URI="http://0.0.0.0:5000"
MLFLOW_DEFAULT_EXPERIMENT="titanic-experiment"

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

def download_and_copy_data(data_root, train_filepath, test_filepath):
    os.system(f'kaggle competitions download -c titanic -p {data_root} --force')
    os.system(f'unzip -o {data_root}/"titanic.zip" -d {data_root}')
    os.system(f'cp {data_root}/train.csv {train_filepath}')
    os.system(f'cp {data_root}/test.csv {test_filepath}')

    # clean up
    os.system(f'rm {data_root}/*.csv {data_root}/*.zip')


def extract_target(data: pd.DataFrame, target="Survived"):
    targets = data[target].values
    return targets

def preprocess_df(df: pd.DataFrame, transforms, categorical, numerical):
    """Return processed features dict and target."""
    # Apply in-between transformations
    df = compose(*transforms[::-1])(df)
    # For dict vectorizer: int = ignored, str = one-hot
    df[categorical] = df[categorical].astype(str)

    # Convert dataframe to feature dictionaries
    feature_dicts = df[categorical + numerical].to_dict(orient='records')

    return feature_dicts


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

def preprocess(df: pd.DataFrame, dv: DictVectorizer, fit_dv: bool = False):
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    if fit_dv:
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    return X, dv

def split_train_read(filename: str, val_size=0.2, random_state=42):
    df_train_full = read_data(filename)

    df_train, df_val = train_test_split(df_train_full, test_size=val_size, random_state=random_state)
    return df_train, df_val

def save_preprocessed(df: pd.DataFrame, path):
    df.to_csv(path)

def dump_pickle(obj, filename):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)

class TrainPath:
    def __init__(self, _data_root: str):
        self._data_root = _data_root
        self._data_root_raw = os.path.join(_data_root, 'raw')

        self._data_root_raw = os.path.join(_data_root, 'raw')
        self._data_root_processed =  os.path.join(_data_root, 'processed')

        self._train_dirpath = os.path.join(self._data_root_raw, "train")
        self._train_filepath = os.path.join(self._train_dirpath, "train.csv")
        self._test_dirpath = os.path.join(self._data_root_raw, "test")
        self._test_filepath = os.path.join(self._test_dirpath, "test.csv")

        self._train_processed_dirpath = os.path.join(self._data_root_processed, "train")
        self._train_processed_filepath = os.path.join(self._data_root_processed, "train.csv")
        self._valid_processed_dirpath = os.path.join(self._data_root_processed, "valid")
        self._valid_processed_filepath = os.path.join(self._data_root_processed, "valid.csv")
        self._test_processed_dirpath = os.path.join(self._data_root_processed, "test")
        self._test_processed_filepath = os.path.join(self._data_root_processed, "test.csv")

def get_paths(_data_root):
    return TrainPath(_data_root=_data_root)

def train(train_path: str, models_path: str):
    transforms = []
    target = 'Survived'
    categorical = ['Sex', 'Pclass', 'Embarked', 'SibSp', 'Parch']
    numerical = ['Fare']

    df_train, df_val = split_train_read(train_path._train_filepath, val_size=0.2, random_state=42)

    train_dicts, y_train = preprocess_df(df_train, transforms, categorical, numerical), extract_target(df_train)
    val_dicts, y_val = preprocess_df(df_val, transforms, categorical, numerical), extract_target(df_val)

    df_test = read_data(train_path._test_filepath)
    test_dicts = preprocess_df(df_test, transforms, categorical, numerical)

    # Fit all possible categories
    dv = DictVectorizer()
    dv.fit(train_dicts)

    X_train = dv.transform(train_dicts)
    X_val = dv.transform(val_dicts)
    X_test = dv.transform(test_dicts)

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

    model.fit(X_train, y_train)

    with open(f'{models_path}/preprocessor.b', "wb") as f_out:
        pickle.dump(dv, f_out)
    mlflow.log_artifact(f'{models_path}/preprocessor.b', artifact_path="preprocessor")
    mlflow.sklearn.log_model(model, artifact_path="models_pickle")

    y_pred = model.predict(X_val)

    accuracy = np.round(accuracy_score(y_val, y_pred), 4)
    mlflow.log_metric("accuracy", accuracy)

    print(accuracy)

def run(data_root: str, mlflow_tracking_uri: str, mlflow_experiment: str, models_path: str):
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    print(f"tracking URI: '{mlflow.get_tracking_uri()}'")
    mlflow.set_experiment(mlflow_experiment)
    mlflow.sklearn.autolog()

    with mlflow.start_run():
        train_path = get_paths(data_root)
        #def download_copy_data(data_root, train_filepath, test_filepath):

        download_and_copy_data(data_root=train_path._data_root, train_filepath=train_path._train_filepath,
                           test_filepath=train_path._test_filepath)
        train(train_path=train_path, models_path=models_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        default='../../data',
        help="The location where the raw Titanic data is downloaded"
    )
    parser.add_argument(
        "--models_path",
        default='../../models',
        help="The location where the raw Titanic data was saved"
    )
    parser.add_argument(
        "--dest_path",
        help="The location where the resulting files will be saved."
    )
    parser.add_argument(
        "--mlflow_tracking_uri",
        default=MLFLOW_DEFAULT_TRACKING_URI,
        help="Mlflow tracking uri"
    )
    parser.add_argument(
        "--mlflow_experiment",
        default=MLFLOW_DEFAULT_EXPERIMENT,
        help="Mlflow experiment"
    )

    args = parser.parse_args()

    run(args.data_root, args.mlflow_tracking_uri, args.mlflow_experiment, args.models_path)