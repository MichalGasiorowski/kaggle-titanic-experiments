import argparse
import os

import mlflow

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

def download_and_copy_data(data_root, train_filepath, test_filepath, kaggle_competition: str):
    os.system(f'kaggle competitions download -c {kaggle_competition} -p {data_root} --force')
    os.system(f'unzip -o {data_root}/"{kaggle_competition}.zip" -d {data_root}')
    os.system(f'cp {data_root}/train.csv {train_filepath}')
    os.system(f'cp {data_root}/test.csv {test_filepath}')

    # clean up
    os.system(f'rm {data_root}/*.csv {data_root}/*.zip')

def run(data_root: str, kaggle_competition: str):
    paths = get_paths(data_root)
    download_and_copy_data(data_root=paths._data_root, train_filepath=paths._train_filepath,
                       test_filepath=paths._test_filepath, kaggle_competition=kaggle_competition)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kaggle_competition",
        default='titanic',
        help="Kaggle competition"
    )
    parser.add_argument(
        "--data_root",
        default='../../data',
        help="The location where the raw data is downloaded"
    )

    args = parser.parse_args()

    run(data_root=args.data_root, kaggle_competition=args.kaggle_competition)