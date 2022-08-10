import pickle
from datetime import datetime

import os
import mlflow

import pandas as pd
from flask import Flask, request, jsonify

import boto3
import pickle
from src.features.build_features import preprocess_test

app = Flask('survivorship-prediction')

RUN_ID = 'd6dc868dda9e41ecaf536a91f41e059e'

model_s3 = f's3://mlflow-enkidupal-experiments/1/{RUN_ID}/artifacts/models_pickle/'

model = mlflow.pyfunc.load_model(model_s3)

s3client = boto3.client('s3' )

response = s3client.get_object(Bucket='mlflow-enkidupal-experiments', Key=f'1/{RUN_ID}/artifacts/preprocessor/preprocessor.b')

body = response['Body'].read()
preprocessor = pickle.loads(body)

def create_feature(json):
    df = pd.DataFrame(json, index=[0])
    return df

@app.route('/predict', methods=['POST'])
def predict():
    json = request.get_json()
    print('json', json)
    features = create_feature(json)
    print('features', features)

    return calculate_predict(features)

def calculate_predict(df: pd.DataFrame):
    preprocessed = preprocess_test(df, preprocessor)

    prediction = model.predict(preprocessed)

    result = {
        'prediction': float(prediction[0])
    }
    print(result)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)



