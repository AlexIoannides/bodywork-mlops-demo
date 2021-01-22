"""
This module defines what will happen in 'stage-4-test-model-scoring-service':

- downloads the most recent CSV data file, representing new incoming data;
- scores the new data by sending requests to the model-scoring service;
- compares scores with labels to compute test metrics; and,
- persists metrics to cloud storage (AWS S3).
"""
import re
from datetime import date, datetime
from io import BytesIO
from time import time
from typing import Dict, Tuple

import boto3 as aws
import pandas as pd
import requests
from botocore.exceptions import ClientError
from requests.exceptions import ConnectionError, Timeout

AWS_S3_BUCKET = 'bodywork-ml-ops-project'
MODEL_SCORING_SERVICE_URL = 'http://bodywork-mlops-demo--stage-2-serve-model:5000/score/v1'


def main() -> None:
    """Main script to be executed."""
    test_data, test_data_date = download_latest_data_file(AWS_S3_BUCKET)
    test_results = generate_model_test_results(MODEL_SCORING_SERVICE_URL, test_data)
    test_metrics = compute_test_metrics(test_results, test_data_date)
    persist_test_metrics(test_metrics, test_data_date, AWS_S3_BUCKET)


def download_latest_data_file(aws_bucket: str) -> Tuple[pd.DataFrame, date]:
    """Get latest model from AWS S3 bucket."""
    def _date_from_object_key(key: str) -> date:
        """Extract date from S3 file object key."""
        date_string = re.findall('20[2-9][0-9]-[0-1][0-9]-[0-3][0-9]', key)[0]
        file_date = datetime.strptime(date_string, '%Y-%m-%d').date()
        return file_date

    print(f'downloading latest data file from s3://{aws_bucket}/datasets')
    try:
        s3_client = aws.client('s3')
        s3_objects = s3_client.list_objects(Bucket=aws_bucket, Prefix='datasets/')
        object_keys_and_dates = [
            (obj['Key'], _date_from_object_key(obj['Key']))
            for obj in s3_objects['Contents']
        ]
        latest_file_obj = sorted(object_keys_and_dates, key=lambda e: e[1])[-1]
        latest_file_obj_key = latest_file_obj[0]
        object_data = s3_client.get_object(Bucket=aws_bucket, Key=latest_file_obj_key)
        data = pd.read_csv(BytesIO(object_data['Body'].read()))
        dataset_date = latest_file_obj[1]
    except ClientError:
        print(f'failed to data file from s3://{aws_bucket}/datasets')
    return (data, dataset_date)


def generate_model_test_results(url: str, test_data: pd.DataFrame) -> pd.DataFrame:
    """Get test results for all test data."""
    def get_model_score_timed(
        url: str,
        features: Dict[str, float]
    ) -> Tuple[float, float]:
        """Request score from REST API for a single instance of data."""
        session = requests.Session()
        session.mount(url, requests.adapters.HTTPAdapter(max_retries=3))
        start_time = time()
        try:
            response = session.post(url, json=features)
            time_taken_to_respond = time() - start_time
            if response.ok:
                return (response.json()['prediction'], time_taken_to_respond)
            else:
                return (-1, time_taken_to_respond)
        except (ConnectionError, Timeout):
            return (-1, -1)

    def _analyse_model_score(score: float, label: float) -> Tuple[float, float, float]:
        """Compute performance metrics for model score."""
        absolute_percentage_error = abs(score / label - 1)
        return (score, label, absolute_percentage_error)

    def _single_test_result(X: float, label: float) -> Tuple[float, float, float, float]:
        score, response_time = get_model_score_timed(url, {'X': X})
        test_result = _analyse_model_score(score, label)
        return (*test_result, response_time)

    test_data = [_single_test_result(row.X, row.y) for row in test_data.itertuples()]
    return pd.DataFrame(test_data, columns=['score', 'label', 'APE', 'response_time'])


def compute_test_metrics(test_results: pd.DataFrame, results_date: date) -> pd.DataFrame:
    MAPE = test_results.APE.mean()
    r_squared = test_results.score.corr(test_results.label)
    max_residual = test_results.APE.max()
    mean_response_time = test_results.response_time.mean()
    results_record = pd.DataFrame({
        'date': [results_date],
        'MAPE': [MAPE],
        'r_squared': [r_squared],
        'max_residual': [max_residual],
        'mean_response_time': [mean_response_time]
    })
    return results_record


def persist_test_metrics(
    test_metrics: pd.DataFrame,
    test_data_date: date,
    aws_bucket: str
) -> None:
    """Upload model metrics to AWS S3."""
    test_metrics_filename = f'regressor-test-results-{test_data_date}.csv'
    test_metrics.to_csv(test_metrics_filename, header=True, index=False)
    try:
        s3_client = aws.client('s3')
        s3_client.upload_file(
            test_metrics_filename,
            aws_bucket,
            f'test-metrics/{test_metrics_filename}'
        )
        print(f'uploaded {test_metrics_filename} to s3://{aws_bucket}/test-metrics/')
    except ClientError:
        print('could not upload metrics to S3 - check AWS credentials')


if __name__ == '__main__':
    main()
