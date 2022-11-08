from src.models.predict import create_features, calculate_predict, create_features_for_s3_path


def lambda_handler(event, context):
    if event['s3_path'] is not None:
        query = event['s3_path']
        features = create_features_for_s3_path(query)
    elif event['data'] is not None:
        query = event['data']
        features = create_features(query)
    # pylint: disable=unused-argument

    predictions = calculate_predict(features)

    return dict(predictions)
