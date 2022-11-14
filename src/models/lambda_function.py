import json

from src.models.predict import create_features, calculate_predict, create_features_for_s3_path


def lambda_handler(event, context):
    if "s3_path" in event:
        query = event["s3_path"]
        features = create_features_for_s3_path(query)
    elif "data" in event:
        query = event["data"]
        features = create_features(query)
    # pylint: disable=unused-argument

    predictions = calculate_predict(features)
    pred_json = json.dumps(predictions)
    # return dict(predictions)
    return pred_json
