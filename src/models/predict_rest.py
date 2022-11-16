import json

from flask import Flask, request

from src.data.read import read_data
from src.models.predict import create_features, calculate_predict_dict, create_features_for_s3_path
from src.features.build_features import get_all_columns

app = Flask('titanic-survivorship-prediction')


@app.route('/predict', methods=['POST'])
def predict():
    request_json = request.get_json()
    print(f"request_json: {request_json}")
    features = create_features(request_json)
    predictions = calculate_predict_dict(features)
    pred_json = json.dumps(predictions)
    return pred_json


@app.route('/predict_from_s3_path', methods=['POST'])
def predict_from_path():
    request_json = request.get_json()
    print(f"request_json: {request_json}")
    s3_path = request_json['s3_path']
    features = create_features_for_s3_path(s3_path)

    predictions = calculate_predict_dict(features)
    pred_json = json.dumps(predictions)
    return pred_json


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
