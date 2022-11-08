from flask import Flask, request

from src.data.read import read_data, load_file_from_s3
from src.models.predict import create_features, calculate_predict
from src.features.build_features import get_all_columns

app = Flask('titanic-survivorship-prediction')


@app.route('/predict', methods=['POST'])
def predict():
    json = request.get_json()
    print(json)
    features = create_features(json)
    predictions = calculate_predict(features)
    return dict(predictions)


@app.route('/predict_from_s3_path', methods=['POST'])
def predict_from_path():
    json = request.get_json()
    print(json)
    s3_path = json['s3_path']
    df = load_file_from_s3(s3_path, get_all_columns())

    return calculate_predict(df)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
