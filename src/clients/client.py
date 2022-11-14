from urllib.parse import urljoin

import requests

SERVICE_URL = 'http://localhost:9696/predict'
LAMBDA_URL = 'http://localhost:9000/2015-03-31/functions/function/invocations'

lambda_json_s3_path = {"s3_path": "s3://mlflow-enkidupal-experiments/data/external/test/test.csv"}


class PredictClient(object):
    def __init__(self, host='localhost:9696'):
        self.host = host
        self.url = None

    def post_request(self, json):
        response = requests.post(self.url, json=json, timeout=30)
        return response.json()

    def predict(self, data):
        self.url = urljoin(self.host, 'predict')
        return self.post_request(data)

    def lambda_predict(self, data):
        self.url = urljoin(self.host, '2015-03-31/functions/function/invocations')
        json = {"data": [data]}
        return self.post_request(json)

    def predict_s3path(self, s3_path):
        self.url = urljoin(self.host, 'predict')
        json = {"s3_path": f"{s3_path}"}
        return self.post_request(json)

    def lambda_predict_s3path(self, s3_path):
        self.url = urljoin(self.host, '2015-03-31/functions/function/invocations')
        json = {"s3_path": f"{s3_path}"}
        return self.post_request(json)


class KaggleResponse(object):
    def __init__(self):
        pass

    def make_prediction(self, s3_path):
        pass
