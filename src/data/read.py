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