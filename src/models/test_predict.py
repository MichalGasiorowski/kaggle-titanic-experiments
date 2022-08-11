import requests

passenger_X = {
    "Sex": 'female',
    "Pclass": 0,
    "Embarked": 'C',
    "SibSp": 1,
    "Parch": 10,
    "Fare": 69.34
}

two_passengers = [
    {
        "Sex": 'female',
        "Pclass": 0,
        "Embarked": 'C',
        "SibSp": 1,
        "Parch": 10,
        "Fare": 69.34
    },
    {
    "Sex": 'male',
    "Pclass": 1,
    "Embarked": 'C',
    "SibSp": 0,
    "Parch": 2,
    "Fare": 8
}
]

url = 'http://localhost:9696/predict'
response = requests.post(url, json=two_passengers)
print(response.json())

response = requests.post(url, json=[passenger_X]) # 1-element list, since the list is expected
print(response.json())
