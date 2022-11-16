from urllib.parse import urljoin

import requests

SERVICE_URL_SUFFIX = 'predict'
LAMBDA_URL_SUFFIX = '2015-03-31/functions/function/invocations'

SERVICE_URL = 'http://localhost:9696/predict'
LAMBDA_URL = 'http://localhost:9000/2015-03-31/functions/function/invocations'


lambda_json_s3_path = {"s3_path": "s3://mlflow-enkidupal-experiments/data/external/test/test.csv"}


class PredictServiceClient(object):
    def __init__(self, host='localhost:9696', suffix=SERVICE_URL_SUFFIX):
        self.host = host
        self.set_client_url(suffix=suffix)

    def set_client_url(self, suffix=LAMBDA_URL_SUFFIX):
        self.url = urljoin(self.host, suffix)

    def post_request(self, json):
        response = requests.post(self.url, json=json, timeout=30)
        return response.json()

    def predict(self, data):
        return self.post_request(data)

    def predict_s3path(self, s3_path):
        json = {"s3_path": f"{s3_path}"}
        return self.post_request(json)


class PredictLambdaClient(object):
    def __init__(self, host='localhost:9000', suffix=LAMBDA_URL_SUFFIX):
        self.host = host
        self.set_client_url(suffix=suffix)

    def set_client_url(self, suffix=LAMBDA_URL_SUFFIX):
        self.url = urljoin(self.host, suffix)

    def post_request(self, json):
        response = requests.post(self.url, json=json, timeout=30)
        return response.json()

    def predict(self, data):
        return self.post_request(data)

    def lambda_predict(self, data):
        json = {"data": [data]}
        return self.post_request(json)

    def predict_s3path(self, s3_path):
        json = {"s3_path": f"{s3_path}"}
        return self.post_request(json)


class KaggleSubmissionClient(object):
    def __init__(self, client):
        self.client = client

    def make_kaggle_submission_s3_path(self, s3_path):
        response = self.client.predict_s3path(s3_path)

        return response
