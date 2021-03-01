"""
This module generates synthetic training data for use in
stage-1-train-model and stage-4-test-model-scoring-service.
"""
from datetime import date

import boto3 as aws
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError

AWS_S3_BUCKET = 'bodywork-ml-ops-project'
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
        print(f'uploaded {dataset_filename} to s3://{aws_bucket}/datasets/')
    except ClientError:
        print('could not upload dataset to S3 - check AWS credentials')


if __name__ == '__main__':
    main()
