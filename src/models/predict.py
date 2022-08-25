import pickle
from datetime import datetime

import os
import mlflow

import pandas as pd
from flask import Flask, request, jsonify

import boto3
import pickle
from src.features.build_features import preprocess_test
from src.features.build_features import all_columns

app = Flask('titanic-survivorship-prediction')

DEFAULT_RUN_ID = '236d76d507b343b69bc755385d9a017f'
RUN_ID = os.getenv('RUN_ID', DEFAULT_RUN_ID)
BUCKET_NAME = os.getenv('BUCKET_NAME', 'mlflow-enkidupal-experiments')


model_s3_path = f's3://{BUCKET_NAME}/1/{RUN_ID}/artifacts/models_pickle/'

model = mlflow.pyfunc.load_model(model_s3_path)

s3client = boto3.client('s3')

response = s3client.get_object(Bucket=BUCKET_NAME, Key=f'1/{RUN_ID}/artifacts/preprocessor/preprocessor.b')

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

    return jsonify(result)


@app.route('/predict', methods=['POST'])
def predict():
    json = request.get_json()
    print(json)
    features = create_features(json)
    return calculate_predict(features)

@app.route('/predict_from_path', methods=['POST'])
def predict_from_path():
    json = request.get_json()
    print(json)
    path = json['path']
    df = read_data(path)

    return calculate_predict(df)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)


