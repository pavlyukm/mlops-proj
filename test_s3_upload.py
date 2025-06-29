#!/usr/bin/env python3
"""Test S3 upload for MLflow artifacts"""
import boto3
import os
from dotenv import load_dotenv

load_dotenv()

def test_s3_mlflow_upload():
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION', 'us-east-1')
    )
    
    bucket_name = os.getenv('S3_BUCKET_NAME', 'pavliukmmlops')
    
    # Test creating MLflow directory structure
    test_content = b"Test MLflow artifact"
    test_key = "mlflow/test/artifact.txt"
    
    try:
        s3.put_object(
            Bucket=bucket_name,
            Key=test_key,
            Body=test_content
        )
        print(f"Successfully uploaded test file to s3://{bucket_name}/{test_key}")
        
        # List objects in mlflow prefix
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix="mlflow/")
        
        if 'Contents' in response:
            print("\nObjects in mlflow/ prefix:")
            for obj in response['Contents']:
                print(f"  - {obj['Key']} (Size: {obj['Size']} bytes)")
        else:
            print("\nNo objects found in mlflow/ prefix")
            
        # Clean up test file
        s3.delete_object(Bucket=bucket_name, Key=test_key)
        print(f"\nCleaned up test file")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_s3_mlflow_upload()