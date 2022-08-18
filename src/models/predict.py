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

DEFAULT_RUN_ID = '236d76d507b343b69bc755385d9a017f'
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


#def calculate_predict(df: pd.DataFrame):
def calculate_predict(raw_dicts, df: pd.DataFrame):
    X_test = preprocess_test(df, preprocessor)

    ###
    # returns dictionary, since model expects its
    # and has dv as first pipeline step
    #dicts = preprocess_test_no_pipeline(df)
    ###
    print(model)

    print(f'raw_dicts:{raw_dicts}', type(raw_dicts))

    predictions = model.predict(X_test).tolist()
    result = {
        'predictions': list(predictions)
    }

    return jsonify(result)



@app.route('/predict', methods=['POST'])
def predict():
    json = request.get_json()
    print(json, type(json))
    features = create_features(json)
    return calculate_predict(json, features)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)



