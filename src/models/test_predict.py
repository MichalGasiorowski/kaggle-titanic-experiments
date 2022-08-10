import requests

passenger_X = {
    "Sex": ['female'],
    "Pclass": [0],
    "Embarked": ['C'],
    "SibSp": [1],
    "Parch": [10],
    "Fare": [69.34]
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=passenger_X)
print(response.json())
