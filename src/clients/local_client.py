import pandas as pd

import src.models.predict as predict

test_s3_path = {"s3_path": "s3://mlflow-enkidupal-experiments/data/external/test/test.csv"}


class LocalPredictClient(object):
    def __init__(self):
        pass

    def predict(self, df):
        return predict.calculate_predict_df(df)

    def predict_s3path(self, s3_path):
        features = predict.create_features_for_s3_path(s3_path)
        return predict.calculate_predict_df(features)


class LocalKaggleSubmissionClient(object):
    def __init__(self, local_client):
        self.local_client = LocalPredictClient()

    def make_kaggle_submission_s3_path(self, s3_path):
        df: pd.DataFrame = self.local_client.predict_s3path(s3_path=s3_path)

        return df.to_csv(columns=['PassengerId', 'Survived'], index=False)
