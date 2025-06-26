import mlflow
import os
import logging
import time
import tempfile
import shutil

logger = logging.getLogger(__name__)

mlflow_available = False

def setup_mlflow_with_s3():
    """Configure MLflow to use S3 as artifact store"""
    global mlflow_available
    
    mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Set S3 as default artifact root
    s3_bucket = os.getenv('S3_BUCKET_NAME', 'pavliukmmlops')
    os.environ['MLFLOW_DEFAULT_ARTIFACT_ROOT'] = f"s3://{s3_bucket}/mlflow"
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Create or get experiment
            experiment = mlflow.get_experiment_by_name("customer_support_ticket_classification")
            if experiment is None:
                mlflow.create_experiment(
                    "customer_support_ticket_classification",
                    artifact_location=f"s3://{s3_bucket}/mlflow"
                )
            else:
                mlflow.set_experiment("customer_support_ticket_classification")
            
            mlflow_available = True
            logger.info("Successfully connected to MLflow with S3 backend")
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"MLflow connection attempt {attempt + 1} failed: {e}")
                time.sleep(2)
            else:
                logger.error(f"Failed to connect to MLflow after {max_retries} attempts: {e}")
                return False

def register_champion_model(model, model_name, metrics):
    """Register the champion model in MLflow Model Registry"""
    if not mlflow_available:
        return None
    
    try:
        bucket_name = os.getenv('S3_BUCKET_NAME', 'pavliukmmlops')
        logger.info(f"Registering model to MLflow with S3 backend: s3://{bucket_name}/mlflow")
        
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save the model in Keras format only
            keras_path = os.path.join(tmp_dir, "model.keras")
            
            # Save in Keras format
            model.save(keras_path)
            logger.info("Saved model in .keras format")
            
            # Log the model file as artifact
            mlflow.log_artifact(keras_path, "model_keras")
            
            # Create a simple model wrapper for MLflow
            import mlflow.pyfunc
            
            class KerasModelWrapper(mlflow.pyfunc.PythonModel):
                def load_context(self, context):
                    import tensorflow as tf
                    self.model = tf.keras.models.load_model(context.artifacts["model"])
                
                def predict(self, context, model_input):
                    return self.model.predict(model_input)
            
            # Log the model using pyfunc
            artifacts = {"model": keras_path}
            conda_env = {
                "channels": ["defaults"],
                "dependencies": [
                    "python=3.9",
                    "pip",
                    {
                        "pip": [
                            "tensorflow==2.16.2",
                            "mlflow",
                            "numpy",
                            "pandas"
                        ]
                    }
                ]
            }
            
            mlflow.pyfunc.log_model(
                artifact_path="pyfunc_model",
                python_model=KerasModelWrapper(),
                artifacts=artifacts,
                conda_env=conda_env
            )
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Get the run ID
        run_id = mlflow.active_run().info.run_id
        
        # Register using the pyfunc model
        model_uri = f"runs:/{run_id}/pyfunc_model"
        
        # Register the model
        mv = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
        
        # Transition to production
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage="Production",
            archive_existing_versions=True
        )
        
        logger.info(f"Successfully registered model {model_name} version {mv.version} as Production")
        logger.info(f"Model artifacts saved to S3 at: s3://{bucket_name}/mlflow/{run_id}/artifacts/")
        
        # Save locally as backup
        model.save('model/best_model.keras')
        logger.info("Model also saved locally as backup")
        
        return mv
        
    except Exception as e:
        logger.error(f"Error registering model: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Fallback: at least save the model locally
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
        model_version = client.get_latest_versions(model_name, stages=["Production"])[0]
        
        # Load as pyfunc model
        model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}/{model_version.version}"
        )
        
        # Extract the actual Keras model
        if hasattr(model, '_model_impl'):
            if hasattr(model._model_impl, 'model'):
                return model._model_impl.model
        
        # If that doesn't work, load the keras artifact directly
        run_id = model_version.run_id
        import tensorflow as tf
        
        # Try different paths
        for artifact_path in ["model_keras/model.keras", "pyfunc_model/artifacts/model"]:
            try:
                artifact_uri = f"runs:/{run_id}/{artifact_path}"
                local_path = mlflow.artifacts.download_artifacts(artifact_uri)
                keras_model = tf.keras.models.load_model(local_path)
                logger.info(f"Loaded Keras model from {artifact_path}")
                return keras_model
            except:
                continue
        
        logger.error("Could not extract Keras model from MLflow")
        return None
        
    except Exception as e:
        logger.error(f"Error loading production model: {e}")
        return None