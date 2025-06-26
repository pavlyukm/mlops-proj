import os
import mlflow

def configure_mlflow_s3():
    """Configure MLflow to use S3 for artifact storage"""
    # Set environment variables for S3
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv('MLFLOW_S3_ENDPOINT_URL', 'https://s3.amazonaws.com')
    os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID', '')
    os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY', '')
    os.environ['AWS_DEFAULT_REGION'] = os.getenv('AWS_REGION', 'us-east-1')
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
    
    # Get S3 bucket
    bucket_name = os.getenv('S3_BUCKET_NAME', 'pavliukmmlops')
    
    # Set experiment with S3 artifact location
    experiment_name = "customer_support_ticket_classification"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        mlflow.create_experiment(
            experiment_name,
            artifact_location=f"s3://{bucket_name}/mlflow"
        )
    
    mlflow.set_experiment(experiment_name)
    
    print(f"MLflow configured with S3 artifact storage at: s3://{bucket_name}/mlflow")

if __name__ == "__main__":
    configure_mlflow_s3()