"""
This module defines what will happen in 'stage-1-train-model':

- download latest dataset;
- train machine learning model and compute metrics; and,
- save model and metrics to cloud storage (AWS S3).
"""
import re
from datetime import datetime, date
from typing import Tuple

import boto3 as aws
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError
from joblib import dump
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, max_error, r2_score
from sklearn.model_selection import train_test_split

AWS_S3_BUCKET = 'bodywork-ml-ops-project'


def main() -> None:
    """Main script to be executed."""
    data, data_date = download_latest_dataset(AWS_S3_BUCKET)
    model, metrics = train_model(data)
    persist_model(model, data_date)
    persist_metrics(metrics, data_date)


def download_latest_dataset(aws_bucket: str) -> Tuple[pd.DataFrame, date]:
    """Get all available data from AWS S3 bucket.
    
    This function reads all CSV files from an AWS S3 bucket and then
    combines them into a single Pandas DataFrame object.
    """
    def _date_from_object_key(key: str) -> date:
        """Extract date from S3 file object key."""
        date_string = re.findall('20[2-9][0-9]-[0-1][0-9]-[0-3][0-9]', key)[0]
        file_date = datetime.strptime(date_string, '%Y-%m-%d').date()
        return file_date

    def _load_dataset_from_aws_s3(s3_obj_key: str) -> pd.DataFrame:
        """Load CSV datafile from AWS S3 into DataFrame."""
        object_data = s3_client.get_object(
            Bucket=aws_bucket,
            Key=s3_obj_key
        )
        return pd.read_csv(object_data['Body'])

        
    print(f'downloading all available training data from s3://{aws_bucket}/datasets')
    try:
        s3_client = aws.client('s3')
        s3_objects = s3_client.list_objects(Bucket=aws_bucket, Prefix='datasets/')
        object_keys_and_dates = [
            (obj['Key'], _date_from_object_key(obj['Key']))
            for obj in s3_objects['Contents']
        ]
        ordered_dataset_objs = sorted(object_keys_and_dates, key=lambda e: e[1])
        dataset = pd.concat(
            _load_dataset_from_aws_s3(obj_key[0])
            for obj_key in ordered_dataset_objs
        )
    except ClientError as e:
        print(f'failed to download training data from s3://{aws_bucket}/datasets')
    most_recent_date = object_keys_and_dates[-1][1]
    return (dataset, most_recent_date)


def model_metrics(y_actual, y_predicted) -> pd.DataFrame:
    """Return regression metrics record."""
    mape = mean_absolute_percentage_error(y_actual, y_predicted)
    r_squared = r2_score(y_actual, y_predicted)
    max_residual = max_error(y_actual, y_predicted)
    metrics_record = pd.DataFrame({
        'MAPE': [mape],
        'R2': [r_squared],
        'MR': [max_residual]
    })
    return metrics_record


def train_model(data:pd.DataFrame) -> Tuple[BaseEstimator, pd.DataFrame]:
    """Train regression model and compute metrics."""
    X = data['X'].values.reshape(-1, 1)
    y = data['y'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    ols_regressor = LinearRegression(fit_intercept=True)
    ols_regressor.fit(X_train, y_train)
    metrics = model_metrics(y_test, ols_regressor.predict(X_test))
    return (ols_regressor, metrics)


def persist_model(model: BaseEstimator, data_date: date) -> None:
    """Upload trained model to AWS S3."""
    model_filename = f'regressor-{data_date}.joblib'
    dump(model, model_filename)
    try:
        s3_client = aws.client('s3')
        s3_client.upload_file(
            model_filename,
            AWS_S3_BUCKET,
            f'models/{model_filename}'
        )
        print(f'uploaded {model_filename} to s3://{AWS_S3_BUCKET}/models/')
    except ClientError:
        print('could not upload model to S3 - check AWS credentials')


def persist_metrics(metrics: pd.DataFrame, data_date: date) -> None:
    """Upload model metrics to AWS S3."""
    metrics_filename = f'regressor-{data_date}.csv'
    metrics.to_csv(metrics_filename, header=True, index=False)
    try:
        s3_client = aws.client('s3')
        s3_client.upload_file(
            metrics_filename,
            AWS_S3_BUCKET,
            f'model-metrics/{metrics_filename}'
        )
        print(f'uploaded {metrics_filename} to s3://{AWS_S3_BUCKET}/model-metrics/')
    except ClientError:
        print('could not upload model metrics to S3 - check AWS credentials')


if __name__ == '__main__':
    main()
