import boto3
import os
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

class S3Client:
    def __init__(self):
        self.client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        self.bucket_name = os.getenv('S3_BUCKET_NAME', 'pavliukmmlops')
    
    def download_training_data(self, file_key='train-00000-of-00001.parquet'):
        """Download training data from S3"""
        logger.info(f"Downloading {file_key} from S3 bucket {self.bucket_name}")
        response = self.client.get_object(Bucket=self.bucket_name, Key=file_key)
        return BytesIO(response['Body'].read())
    
    def upload_to_s3(self, data, key):
        """Upload data to S3"""
        self.client.put_object(Bucket=self.bucket_name, Key=key, Body=data)
        return f"s3://{self.bucket_name}/{key}"