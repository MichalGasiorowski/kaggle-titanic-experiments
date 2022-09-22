
from src.models.predict import calculate_predict
from src.models.predict import read_data
from src.models.predict import create_features


def lambda_handler(event, context):
    # pylint: disable=unused-argument
    query = event['data']
    features = create_features(query)
    predictions = calculate_predict(features)

    return dict(predictions)


