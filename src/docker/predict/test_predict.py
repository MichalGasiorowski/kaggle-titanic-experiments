import argparse

# import curlify
import requests
from deepdiff import DeepDiff

SERVICE_URL = 'http://localhost:9696/predict'
LAMBDA_URL = 'http://localhost:9000/2015-03-31/functions/function/invocations'


def curlify_request(req):
    command = "curl -X {method} -H {headers} -d '{data}' '{uri}'"
    method = req.method
    uri = req.url
    data = req.body
    headers = ['"{0}: {1}"'.format(k, v) for k, v in req.headers.items()]  # pylint: disable=consider-using-f-string
    headers = " -H ".join(headers)
    return command.format(method=method, headers=headers, data=data, uri=uri)


passenger_X = {"Age": 45, "Sex": 'female', "Pclass": 0, "Embarked": 'C', "SibSp": 1, "Parch": 4, "Fare": 1111}
two_passengers = [
    {"Age": 123, "Sex": 'female', "Pclass": 0, "Embarked": 'C', "SibSp": 1, "Parch": 10, "Fare": 69.34},
    {"Age": 13, "Sex": 'male', "Pclass": 3, "Embarked": 'C', "SibSp": 0, "Parch": 2, "Fare": 8},
]
service_json_s3_path = {"s3_path": "s3://mlflow-enkidupal-experiments/data/external/test/test.csv"}

lambda_passenger_X = {"data": [passenger_X]}
lambda_two_passengers = {"data": two_passengers}
lambda_json_s3_path = {"s3_path": "s3://mlflow-enkidupal-experiments/data/external/test/test.csv"}

expected_response_single = [{"PassengerId": ".*", 'Prediction': [0.795], 'Survived': 1}]
expected_response_multi = [
    {"PassengerId": ".*", 'Prediction': [0.84], 'Survived': 1},
    {"PassengerId": ".*", 'Prediction': [0.495], 'Survived': 0},
]


def get_request_json(scenario):
    json = None
    if scenario == 'single_service':
        json = [passenger_X]
    elif scenario == 'multi_service':
        json = two_passengers
    elif scenario == 's3_path_service':
        json = service_json_s3_path
    elif scenario == 'single_lambda':
        json = lambda_passenger_X
    elif scenario == 'multi_lambda':
        json = lambda_two_passengers
    elif scenario == 's3_path_lambda':
        json = lambda_json_s3_path
    return json


def test_json_response(scenario, response_json):
    if scenario == 's3_path_service':
        assert response_json['Prediction'] is not None
        predictions = response_json['Prediction']
        assert len(predictions) == 418
        assert all(v >= 0.0 and v <= 1.0 for v in predictions)

        assert response_json['Survived'] is not None
        decisions = response_json['Survived']
        assert len(response_json['Survived']) == 418
        assert all(v in (0, 1) for v in decisions)
        return
    elif scenario == 's3_path_lambda':
        assert response_json['Prediction'] is not None
        predictions = response_json['Prediction']
        assert len(predictions) == 418
        assert all(v >= 0.0 and v <= 1.0 for v in predictions)

        assert response_json['Survived'] is not None
        decisions = response_json['Survived']
        assert len(response_json['Survived']) == 418
        assert all(v in (0, 1) for v in decisions)
        return

    expected_json = None
    if scenario == 'single_service':
        expected_json = expected_response_single
    elif scenario == 'multi_service':
        expected_json = expected_response_single
    elif scenario == 'single_lambda':
        expected_json = expected_response_single
    elif scenario == 'multi_lambda':
        expected_json = expected_response_multi

    diff = DeepDiff(response_json, expected_json, significant_digits=1)
    print(f'diff={diff}')

    assert 'type_changes' not in diff
    assert 'values_changed' not in diff


def get_expected_response(scenario):
    json = None
    if scenario == 'single_service':
        json = expected_response_single
    elif scenario == 'multi_service':
        json = expected_response_single
    elif scenario == 'single_lambda':
        json = expected_response_single
    elif scenario == 'multi_lambda':
        json = expected_response_multi
    return json

def send_request(url, scenario):
    '''
    python test_predict.py --url 'http://localhost:9696/predict' --scenario 'single_service'
    python test_predict.py --url 'http://localhost:9000/2015-03-31/functions/function/invocations' --scenario 'single_lambda' #pylint: disable=line-too-long
    '''
    json = get_request_json(scenario)

    response = requests.post(url, json=json, timeout=30)  # 1-element list, since the list is expected
    response_json = response.json()

    req = response.request
    curlified_request = curlify_request(req)

    print(f'curlified_request:\n {curlified_request}')

    print(f'response.json(): \n {response_json}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=SERVICE_URL, help="url to test prediction")
    parser.add_argument("--scenario", default='scenario', help="scenario to test")

    args = parser.parse_args()

    send_request(args.url, args.scenario)
