import json

from flask import Flask, request

from src.data.read import read_data
from src.models.predict import create_features, calculate_predict, create_features_for_s3_path
from src.features.build_features import get_all_columns

app = Flask('titanic-survivorship-prediction')


@app.route('/predict', methods=['POST'])
def predict():
    json = request.get_json()
    print(json)
    features = create_features(json)
    predictions = calculate_predict(features)
    pred_json = json.dumps(predictions)
    return pred_json


@app.route('/predict_from_s3_path', methods=['POST'])
def predict_from_path():
    json = request.get_json()
    print(json)
    s3_path = json['s3_path']
    features = create_features_for_s3_path(s3_path)

    predictions = calculate_predict(features)
    pred_json = json.dumps(predictions)
    return pred_json


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
