import mlflow
import logging
import numpy as np
import tensorflow as tf
from data_loading_and_processing.data_loading_and_processing import load_and_preprocess_data
from model.model_builder import train_with_hyperparameter_tuning
from utils.s3_utils import S3Client
from utils.ml_utils import save_encoders
from utils.mlflow_utils import register_champion_model
from sklearn.metrics import accuracy_score, classification_report

# Enable eager execution
tf.config.run_functions_eagerly(True)

logger = logging.getLogger(__name__)

class TrainingService:
    def __init__(self):
        self.s3_client = S3Client()
    
    def train_model(self, file_key='train-00000-of-00001.parquet'):
        """Execute full training pipeline with hyperparameter tuning"""
        # Ensure MLflow is configured with S3
        from utils.mlflow_utils import setup_mlflow_with_s3
        setup_mlflow_with_s3()
        
        with mlflow.start_run():
            try:
                # Set S3 artifact location for this run
                mlflow.set_tag("mlflow.artifactLocation", f"s3://{self.s3_client.bucket_name}/mlflow")
                
                # Download data from S3
                data = self.s3_client.download_training_data(file_key)
                
                # Log data source
                mlflow.log_param("data_source", "S3")
                mlflow.log_param("bucket_name", self.s3_client.bucket_name)
                mlflow.log_param("file_key", file_key)
                
                # Load and preprocess data
                X_train, X_test, y_train, y_test, tfidf_vectorizer, product_encoder, priority_encoder, ticket_type_encoder = load_and_preprocess_data(data)
                
                # Ensure data is in numpy format
                import numpy as np
                X_train = np.array(X_train)
                X_test = np.array(X_test)
                y_train = np.array(y_train)
                y_test = np.array(y_test)
                
                # Save encoders
                encoders = {
                    'tfidf': tfidf_vectorizer,
                    'product': product_encoder,
                    'priority': priority_encoder,
                    'ticket_type': ticket_type_encoder
                }
                save_encoders(encoders)
                
                # Log dataset info
                mlflow.log_param("train_samples", len(X_train))
                mlflow.log_param("test_samples", len(X_test))
                mlflow.log_param("features", X_train.shape[1])
                
                # Define hyperparameter grid for tuning
                hyperparams_grid = [
                    {'learning_rate': 0.001, 'epochs': 10, 'batch_size': 32},
                    {'learning_rate': 0.0001, 'epochs': 15, 'batch_size': 64},
                    {'learning_rate': 0.0005, 'epochs': 10, 'batch_size': 16},
                    {'learning_rate': 0.002, 'epochs': 8, 'batch_size': 32},
                ]
                
                # Train with hyperparameter tuning
                best_model, best_accuracy, best_params = train_with_hyperparameter_tuning(
                    X_train, y_train, X_test, y_test, hyperparams_grid
                )
                
                # Evaluate best model
                y_pred = best_model.predict(X_test)
                y_pred_classes = np.argmax(y_pred, axis=1)
                y_test_classes = np.argmax(y_test, axis=1)
                
                accuracy = accuracy_score(y_test_classes, y_pred_classes)
                report = classification_report(y_test_classes, y_pred_classes)
                
                # Log final metrics
                mlflow.log_metric("final_accuracy", accuracy)
                mlflow.log_text(report, "classification_report.txt")
                mlflow.log_params(best_params)
                
                # Log encoder files as artifacts (NOT as sklearn models)
                mlflow.log_artifact('tfidf_vectorizer.pkl', "encoders")
                mlflow.log_artifact('product_encoder.pkl', "encoders")
                mlflow.log_artifact('priority_encoder.pkl', "encoders")
                mlflow.log_artifact('ticket_type_encoder.pkl', "encoders")
                
                # Register champion model
                model_version = register_champion_model(
                    best_model,
                    "ticket_classification_model",
                    {"accuracy": accuracy}
                )
                
                # Save model locally as backup
                best_model.save('model/best_model.keras')
                
                return {
                    "accuracy": accuracy,
                    "report": report,
                    "best_params": best_params,
                    "model_version": model_version.version if model_version else None
                }
                
            except Exception as e:
                logger.error(f"Error in training pipeline: {e}")
                mlflow.log_param("error", str(e))
                raise