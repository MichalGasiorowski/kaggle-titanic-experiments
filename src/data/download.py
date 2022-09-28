import os
import os.path
import argparse

DEFAULT_KAGGLE_COMPETITION = 'titanic'


class DataPath:
    def __init__(self, _data_root: str):
        self._data_root = _data_root

        self._data_root_external = os.path.join(_data_root, 'external')
        self._data_root_interim = os.path.join(_data_root, 'interim')
        self._data_root_raw = os.path.join(_data_root, 'raw')
        self._data_root_processed = os.path.join(_data_root, 'processed')

        self._external_train_dirpath = os.path.join(self._data_root_external, "train")
        self._external_test_dirpath = os.path.join(self._data_root_external, "test")

        self._raw_train_dirpath = os.path.join(self._data_root_raw, "train")
        self._raw_valid_dirpath = os.path.join(self._data_root_raw, "valid")
        self._raw_test_dirpath = os.path.join(self._data_root_raw, "test")

        self._processed_train_dirpath = os.path.join(self._data_root_processed, "train")
        self._processed_valid_dirpath = os.path.join(self._data_root_processed, "valid")
        self._processed_test_dirpath = os.path.join(self._data_root_processed, "test")

    @property
    def data_root(self):
        return self._data_root

    @property
    def data_root_external(self):
        return self._data_root_external

    @property
    def data_root_interim(self):
        return self._data_root_interim

    @property
    def data_root_raw(self):
        return self._data_root_raw

    @property
    def data_root_processed(self):
        return self._data_root_processed

    @property
    def external_train_dirpath(self):
        return self._external_train_dirpath

    @property
    def external_test_dirpath(self):
        return self._external_test_dirpath

    @property
    def raw_train_dirpath(self):
        return self._raw_train_dirpath

    @property
    def raw_valid_dirpath(self):
        return self._raw_valid_dirpath

    @property
    def raw_test_dirpath(self):
        return self._raw_test_dirpath

    @property
    def processed_train_dirpath(self):
        return self._processed_train_dirpath

    @property
    def processed_valid_dirpath(self):
        return self._processed_valid_dirpath

    @property
    def processed_test_dirpath(self):
        return self._processed_test_dirpath

    def make_dirs(self):
        for directory in (
            self.data_root_interim,
            self.external_train_dirpath,
            self.external_test_dirpath,
            self.raw_train_dirpath,
            self.raw_valid_dirpath,
            self.raw_test_dirpath,
            self.processed_train_dirpath,
            self.processed_valid_dirpath,
            self.processed_test_dirpath,
        ):
            os.makedirs(directory, exist_ok=True)

    def get_train_file_path(self, dir_root):
        return f'{dir_root}/train.csv'

    def get_valid_file_path(self, dir_root):
        return f'{dir_root}/valid.csv'

    def get_test_file_path(self, dir_root):
        return f'{dir_root}/test.csv'


def get_datapath(_data_root):
    return DataPath(_data_root=_data_root)


def download_and_copy_data(data_path: DataPath, kaggle_competition: str, force: bool = False):
    if os.path.exists(f'{data_path.external_train_dirpath}/train.csv') and not force:
        return
    os.system(f'kaggle competitions download -c {kaggle_competition} -p {data_path.data_root_external} --force')
    os.system(f'unzip -o {data_path.data_root_external}/"{kaggle_competition}.zip" -d {data_path.data_root_external}')
    os.system(f'cp {data_path.data_root_external}/train.csv {data_path.external_train_dirpath}')
    os.system(f'cp {data_path.data_root_external}/test.csv {data_path.external_test_dirpath}')

    # clean up
    os.system(f'rm {data_path.data_root_external}/*.csv {data_path.data_root_external}/*.zip')


def run(data_root: str, kaggle_competition: str):
    data_path = get_datapath(data_root)
    data_path.make_dirs()

    download_and_copy_data(data_path=data_path, kaggle_competition=kaggle_competition)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle_competition", default=DEFAULT_KAGGLE_COMPETITION, help="Kaggle competition")
    parser.add_argument("--data_root", default='../../data', help="The location where the raw data is downloaded")

    args = parser.parse_args()

    run(data_root=args.data_root, kaggle_competition=args.kaggle_competition)
