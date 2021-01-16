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
import re
from datetime import datetime, date
from io import BytesIO
from typing import Tuple

import boto3 as aws
import numpy as np
from botocore.exceptions import ClientError
from flask import Flask, jsonify, make_response, request, Response
from joblib import load
from sklearn.base import BaseEstimator

AWS_S3_BUCKET = 'bodywork-ml-ops-project'

app = Flask(__name__)


def download_latest_model(aws_bucket: str) -> Tuple[BaseEstimator, date]:
    """Get latest model from AWS S3 bucket."""
    def _date_from_object_key(key: str) -> date:
        """Extract date from S3 file object key."""
        date_string = re.findall('20[2-9][0-9]-[0-1][0-9]-[0-3][0-9]', key)[0]
        file_date = datetime.strptime(date_string, '%Y-%m-%d').date()
        return file_date

    print(f'downloading latest model data from s3://{aws_bucket}/models')
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
    except ClientError:
        print(f'failed to download model from s3://{aws_bucket}/models')
    return (model, dataset_date)


@app.route('/score/v1', methods=['POST'])
def score_data_instance() -> Response:
    """Score incoming data instance using loaded model."""
    features = request.json['X']
    X = np.array(features, ndmin=2)
    prediction = model.predict(X)
    response_data = jsonify({'prediction': prediction[0], 'model_info': str(model)})
    return make_response(response_data)


if __name__ == '__main__':
    model, model_date = download_latest_model(AWS_S3_BUCKET)
    print(f'loaded model={model} trained on {model_date}')
    print('starting API server')
    app.run(host='0.0.0.0', port=5000)
