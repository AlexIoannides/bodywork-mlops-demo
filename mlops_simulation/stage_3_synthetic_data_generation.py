"""
This module generates synthetic training data for use in
stage-1-train-model and stage-4-test-model-scoring-service.

The logs are monitored by Sentry.
"""
import logging
import os
import sys
from datetime import date

import boto3 as aws
import numpy as np
import pandas as pd
import sentry_sdk
from botocore.exceptions import ClientError

AWS_S3_BUCKET = 'bodywork-mlops-project'
N = 24 * 60


def main() -> None:
    """Main script to be executed."""
    dataset = generate_dataset(N)
    persist_dataset(dataset, AWS_S3_BUCKET)


def generate_dataset(n: int) -> pd.DataFrame:
    """Create synthetic regression data using linear model with Gaussian noise."""

    def alpha(day_of_year: int, f: float, kappa: float, A: float) -> float:
        """Return alpha on a given day of the year from a sinusoid model."""
        return kappa + A * np.sin(2 * np.pi * f * (day_of_year - 1) / 364)

    today = date.today()
    beta = 0.5
    sigma = 10
    alpha_now = alpha(today.timetuple().tm_yday, 6, 1, 0.5)
    X = np.random.uniform(0, 100, n)
    epsilon = np.random.normal(0, 1, n)
    y = alpha_now + beta * X + sigma * epsilon
    dataset = pd.DataFrame({'date': np.full(n, str(today)), 'y': y, 'X': X})
    return dataset.query('y >= 0')


def persist_dataset(dataset: pd.DataFrame, aws_bucket: str) -> None:
    """Upload dataset metrics to AWS S3."""
    data_date = date.today()
    dataset_filename = f'regression-dataset-{data_date}.csv'
    dataset.to_csv(dataset_filename, header=True, index=False)
    try:
        s3_client = aws.client('s3')
        s3_client.upload_file(
            dataset_filename,
            aws_bucket,
            f'datasets/{dataset_filename}'
        )
        log.info(f'uploaded {dataset_filename} to s3://{aws_bucket}/datasets/')
    except ClientError as e:
        log.error(e)
        raise RuntimeError('could not upload dataset to S3 - check AWS credentials')


def configure_logger() -> logging.Logger:
    """Configure a logger that will write to stdout."""
    log_handler = logging.StreamHandler(sys.stdout)
    log_format = logging.Formatter(
        '%(asctime)s - '
        '%(levelname)s - '
        '%(module)s.%(funcName)s - '
        '%(message)s'
    )
    log_handler.setFormatter(log_format)
    log = logging.getLogger(__name__)
    log.addHandler(log_handler)
    log.setLevel(logging.INFO)
    return log


def get_sentry_dsn() -> str:
    """Get Sentry DSN from SENTRY_DSN environment variable."""
    sentry_dsn = os.environ.get('SENTRY_DSN')
    if sentry_dsn:
        return sentry_dsn
    else:
        raise RuntimeError('cannot find SENTRY_DSN environment variable')


if __name__ == '__main__':
    sentry_sdk.init(get_sentry_dsn(), traces_sample_rate=1.0)
    sentry_sdk.set_tag('stage', 'stage-3-generate-next-dataset')
    try:
        log = configure_logger()
        main()
    except Exception as e:
        log.error(e)
        sys.exit(1)