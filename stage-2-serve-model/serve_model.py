"""
This module defines what will happen in 'stage-2-deploy-scoring-service':

- download the latest ML model and load into memory;
- define ML scoring REST API endpoint; and,
- start service.

When running the script locally, the scoring service can be tested from
the command line using,

curl http://0.0.0.0:5000/score/v1 \
    --request POST \
    --header "Content-Type: application/json" \
    --data '{"X": 50}'

The expected response should be,

{
    "prediction": 54.57560049377929,
    "model_info": "LinearRegression()"
}
"""
import logging
import os
import re
import sys
from datetime import datetime, date
from io import BytesIO
from typing import Tuple

import boto3 as aws
import numpy as np
import sentry_sdk
from botocore.exceptions import ClientError
from flask import Flask, jsonify, make_response, request, Response
from joblib import load
from sklearn.base import BaseEstimator

AWS_S3_BUCKET = 'bodywork-mlops-project'

app = Flask(__name__)


def download_latest_model(aws_bucket: str) -> Tuple[BaseEstimator, date]:
    """Get latest model from AWS S3 bucket."""
    def _date_from_object_key(key: str) -> date:
        """Extract date from S3 file object key."""
        date_string = re.findall('20[2-9][0-9]-[0-1][0-9]-[0-3][0-9]', key)[0]
        file_date = datetime.strptime(date_string, '%Y-%m-%d').date()
        return file_date

    log.info(f'downloading latest model data from s3://{aws_bucket}/models')
    try:
        s3_client = aws.client('s3')
        s3_objects = s3_client.list_objects(Bucket=aws_bucket, Prefix='models/')
        object_keys_and_dates = [
            (obj['Key'], _date_from_object_key(obj['Key']))
            for obj in s3_objects['Contents']
        ]
        latest_model_obj = sorted(object_keys_and_dates, key=lambda e: e[1])[-1]
        latest_model_obj_key = latest_model_obj[0]
        object_data = s3_client.get_object(Bucket=aws_bucket, Key=latest_model_obj_key)
        model = load(BytesIO(object_data['Body'].read()))
        dataset_date = latest_model_obj[1]
    except ClientError as e:
        log.error(e)
        raise RuntimeError(f'failed to download model from s3://{aws_bucket}/models')
    return (model, dataset_date)


@app.route('/score/v1', methods=['POST'])
def score_data_instance() -> Response:
    """Score incoming data instance using loaded model."""
    features = request.json['X']
    X = np.array(features, ndmin=2)
    prediction = model.predict(X)
    response_data = jsonify({'prediction': prediction[0], 'model_info': str(model)})
    return make_response(response_data)


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
    sentry_sdk.set_tag('stage', 'stage-2-serve-model')
    try:
        log = configure_logger()
        model, model_date = download_latest_model(AWS_S3_BUCKET)
        log.info(f'loaded model={model} trained on {model_date}')
        log.info('starting API server')
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        log.error(e)
        sys.exit(1)