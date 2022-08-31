import requests

passenger_X = {
    "Age": 45,
    "Sex": 'female',
    "Pclass": 0,
    "Embarked": 'C',
    "SibSp": 1,
    "Parch": 4,
    "Fare": 1111
}

two_passengers = [
    {
        "Age": 123,
        "Sex": 'female',
        "Pclass": 0,
        "Embarked": 'C',
        "SibSp": 1,
        "Parch": 10,
        "Fare": 69.34
    },
    {
    "Age": 13,
    "Sex": 'male',
    "Pclass": 3,
    "Embarked": 'C',
    "SibSp": 0,
    "Parch": 2,
    "Fare": 8
}
]


url = 'http://localhost:9696/predict'

response = requests.post(url, json=[passenger_X]) # 1-element list, since the list is expected
print(response.json())

response = requests.post(url, json=two_passengers)
print(response.json())


test_path_json = {
    'path': '../../data/external/test/test.csv'
}

#path_url = 'http://localhost:9696/predict_from_path'

#response = requests.post(path_url, json=test_path_json)
#print(response.json())
