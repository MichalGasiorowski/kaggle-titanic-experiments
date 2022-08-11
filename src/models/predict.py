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

app = Flask('survivorship-prediction')

DEFAULT_RUN_ID = 'd6dc868dda9e41ecaf536a91f41e059e'
RUN_ID = os.getenv('RUN_ID', DEFAULT_RUN_ID)
BUCKET_NAME = os.getenv('BUCKET_NAME', 'mlflow-enkidupal-experiments')


model_s3_path = f's3://{BUCKET_NAME}/1/{RUN_ID}/artifacts/models_pickle/'

model = mlflow.pyfunc.load_model(model_s3_path)

s3client = boto3.client('s3')

response = s3client.get_object(Bucket=BUCKET_NAME, Key=f'1/{RUN_ID}/artifacts/preprocessor/preprocessor.b')

body = response['Body'].read()
preprocessor = pickle.loads(body)


def create_features(json):
    df = pd.json_normalize(json, meta=all_columns)
    return df


def calculate_predict(df: pd.DataFrame):
    preprocessed = preprocess_test(df, preprocessor)

    predictions = model.predict(preprocessed).tolist()
    result = {
        'predictions': list(predictions)
    }

    return jsonify(result)



@app.route('/predict', methods=['POST'])
def predict():
    json = request.get_json()
    features = create_features(json)
    return calculate_predict(features)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)



