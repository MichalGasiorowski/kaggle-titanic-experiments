
from flask import Flask, request, jsonify

from src.models.predict import calculate_predict
from src.models.predict import read_data
from src.models.predict import create_features

app = Flask('titanic-survivorship-prediction')


@app.route('/predict', methods=['POST'])
def predict():
    json = request.get_json()
    print(json)
    features = create_features(json)
    predictions = calculate_predict(features)
    return dict(predictions)

@app.route('/predict_from_path', methods=['POST'])
def predict_from_path():
    json = request.get_json()
    print(json)
    path = json['path']
    df = read_data(path)

    return calculate_predict(df)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)