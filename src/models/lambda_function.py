from src.models.predict import create_features, calculate_predict


def lambda_handler(event, context):
    # pylint: disable=unused-argument
    query = event['data']
    features = create_features(query)
    predictions = calculate_predict(features)

    return dict(predictions)
