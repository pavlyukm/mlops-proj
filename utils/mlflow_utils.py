import mlflow
import mlflow.pyfunc
import os
import logging
import time
import tempfile
import numpy as np

logger = logging.getLogger(__name__)

mlflow_available = False

def setup_mlflow_with_s3():
    """Configure MLflow to use S3 as artifact store"""
    global mlflow_available
    
    mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Set S3 as default artifact root
    s3_bucket = os.getenv('S3_BUCKET_NAME', 'pavliukmmlops')
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Create or get experiment
            experiment = mlflow.get_experiment_by_name("customer_support_ticket_classification")
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    "customer_support_ticket_classification",
                    artifact_location=f"s3://{s3_bucket}/mlflow"
                )
                mlflow.set_experiment(experiment_id=experiment_id)
            else:
                mlflow.set_experiment("customer_support_ticket_classification")
            
            mlflow_available = True
            logger.info(f"Successfully connected to MLflow with S3 backend at s3://{s3_bucket}/mlflow")
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"MLflow connection attempt {attempt + 1} failed: {e}")
                time.sleep(2)
            else:
                logger.error(f"Failed to connect to MLflow after {max_retries} attempts: {e}")
                return False


class KerasModelWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper class for Keras models to work with MLflow"""
    
    def load_context(self, context):
        """Load the Keras model from artifacts"""
        import tensorflow as tf
        model_path = context.artifacts["keras_model"]
        self.model = tf.keras.models.load_model(model_path)
        logger.info(f"Loaded Keras model from {model_path}")
    
    def predict(self, context, model_input):
        """Make predictions using the loaded model"""
        return self.model.predict(model_input)


def register_champion_model(model, model_name, metrics):
    """Register the champion model in MLflow Model Registry"""
    if not mlflow_available:
        logger.error("MLflow is not available")
        return None
    
    try:
        bucket_name = os.getenv('S3_BUCKET_NAME', 'pavliukmmlops')
        
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save the Keras model
            model_path = os.path.join(tmp_dir, "model.keras")
            model.save(model_path)
            logger.info(f"Saved Keras model to {model_path}")
            
            # Define artifacts
            artifacts = {"keras_model": model_path}
            
            # Define conda environment
            conda_env = {
                "channels": ["defaults"],
                "dependencies": [
                    "python=3.9",
                    "pip",
                    {
                        "pip": [
                            "mlflow",
                            "tensorflow==2.16.2",
                            "numpy",
                            "pandas",
                            "scikit-learn"
                        ]
                    }
                ]
            }
            
            # Log the model using pyfunc
            logger.info("Logging model to MLflow")
            model_info = mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=KerasModelWrapper(),
                artifacts=artifacts,
                conda_env=conda_env,
                registered_model_name=model_name
            )
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Get the latest model version
            client = mlflow.tracking.MlflowClient()
            model_version = client.get_latest_versions(model_name, stages=["None"])[0]
            
            # Transition to production
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Production",
                archive_existing_versions=True
            )
            
            logger.info(f"Successfully registered model {model_name} version {model_version.version} as Production")
            logger.info(f"Model URI: {model_info.model_uri}")
            logger.info(f"Run ID: {mlflow.active_run().info.run_id}")
            logger.info(f"Artifacts stored in S3 at: s3://{bucket_name}/mlflow")
            
            # Save locally as backup
            model.save('model/best_model.keras')
            logger.info("Model also saved locally as backup")
            
            return model_version
            
    except Exception as e:
        logger.error(f"Error registering model: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Fallback: save locally
        try:
            model.save('model/best_model.keras')
            logger.info("Model saved locally as fallback")
        except Exception as local_e:
            logger.error(f"Failed to save model locally: {local_e}")
        
        return None


def load_production_model(model_name):
    """Load the production model from MLflow Model Registry"""
    try:
        client = mlflow.tracking.MlflowClient()
        
        # Get the latest production model
        model_versions = client.get_latest_versions(model_name, stages=["Production"])
        if not model_versions:
            logger.error(f"No production model found for {model_name}")
            return None
        
        model_version = model_versions[0]
        model_uri = f"models:/{model_name}/{model_version.version}"
        
        logger.info(f"Loading model {model_name} version {model_version.version} from {model_uri}")
        
        # Load as pyfunc model
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        
        # Extract the Keras model
        # The wrapper stores the model as self.model after loading
        if hasattr(loaded_model, '_model_impl') and hasattr(loaded_model._model_impl, 'python_model'):
            wrapper = loaded_model._model_impl.python_model
            if hasattr(wrapper, 'model'):
                logger.info("Successfully extracted Keras model from wrapper")
                return wrapper.model
        
        # If extraction fails, try loading the Keras model directly from artifacts
        import tensorflow as tf
        run_id = model_version.run_id
        artifact_uri = f"runs:/{run_id}/model/artifacts/keras_model"
        local_path = mlflow.artifacts.download_artifacts(artifact_uri)
        keras_model = tf.keras.models.load_model(local_path)
        logger.info("Loaded Keras model directly from artifacts")
        return keras_model
        
    except Exception as e:
        logger.error(f"Error loading production model: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None