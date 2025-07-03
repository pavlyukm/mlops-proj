import tensorflow as tf
import mlflow
import mlflow.keras
import os
import logging
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import tempfile

logger = logging.getLogger(__name__)

# Enable eager execution
tf.config.run_functions_eagerly(True)

def setup_mlflow():
    """Setup MLflow with S3 backend"""
    try:
        mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        mlflow.set_tracking_uri(mlflow_uri)
        
        # Set experiment
        experiment_name = "customer_support_classification_v2"
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            logger.warning(f"MLflow experiment setup warning: {e}")
            
        logger.info(f"MLflow configured with tracking URI: {mlflow_uri}")
        return True
    except Exception as e:
        logger.error(f"MLflow setup failed: {e}")
        return False

def create_model(input_dim, num_classes):
    """Create a simple neural network model"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(X_train, X_test, y_train, y_test, num_classes):
    """Train the model with MLflow tracking"""
    
    if not setup_mlflow():
        logger.warning("MLflow not available, training without tracking")
    
    with mlflow.start_run() as run:
        try:
            # Convert sparse matrices to dense
            X_train_dense = X_train.toarray()
            X_test_dense = X_test.toarray()
            
            # Log parameters
            mlflow.log_param("input_dim", X_train_dense.shape[1])
            mlflow.log_param("num_classes", num_classes)
            mlflow.log_param("train_samples", len(X_train_dense))
            mlflow.log_param("test_samples", len(X_test_dense))
            
            # Create model
            model = create_model(X_train_dense.shape[1], num_classes)
            
            # Callbacks
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            )
            
            # Train model
            history = model.fit(
                X_train_dense, y_train,
                validation_data=(X_test_dense, y_test),
                epochs=20,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=1
            )
            
            # Evaluate model
            y_pred = model.predict(X_test_dense)
            y_pred_classes = np.argmax(y_pred, axis=1)
            
            accuracy = accuracy_score(y_test, y_pred_classes)
            
            # Log metrics
            mlflow.log_metric("final_accuracy", accuracy)
            mlflow.log_metric("final_loss", history.history['loss'][-1])
            mlflow.log_metric("final_val_accuracy", history.history['val_accuracy'][-1])
            mlflow.log_metric("final_val_loss", history.history['val_loss'][-1])
            
            # Log model
            model_path = "model"
            mlflow.keras.log_model(
                model, 
                model_path,
                registered_model_name="customer_support_classifier"
            )
            
            # Save model locally
            os.makedirs('model', exist_ok=True)
            model.save('model/classifier.keras')
            
            logger.info(f"Training completed. Final accuracy: {accuracy:.4f}")
            logger.info(f"Model saved to MLflow and locally")
            
            return model, accuracy, run.info.run_id
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            mlflow.log_param("error", str(e))
            raise

def load_production_model():
    """Load the latest production model"""
    try:
        # Try to load from MLflow first
        try:
            client = mlflow.tracking.MlflowClient()
            model_versions = client.get_latest_versions(
                "customer_support_classifier", 
                stages=["Production", "Staging"]
            )
            
            if model_versions:
                model_version = model_versions[0]
                model_uri = f"models:/customer_support_classifier/{model_version.version}"
                model = mlflow.keras.load_model(model_uri)
                logger.info(f"Loaded model version {model_version.version} from MLflow")
                return model
        except Exception as e:
            logger.warning(f"Could not load from MLflow: {e}")
        
        # Fallback to local model
        if os.path.exists('model/classifier.keras'):
            model = tf.keras.models.load_model('model/classifier.keras')
            logger.info("Loaded model from local file")
            return model
        
        logger.error("No model found")
        return None
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def promote_model_to_production(run_id):
    """Promote a model to production stage"""
    try:
        client = mlflow.tracking.MlflowClient()
        
        # Get the model version from the run
        model_versions = client.search_model_versions(
            f"run_id='{run_id}'"
        )
        
        if model_versions:
            model_version = model_versions[0]
            
            # Transition to production
            client.transition_model_version_stage(
                name="customer_support_classifier",
                version=model_version.version,
                stage="Production",
                archive_existing_versions=True
            )
            
            logger.info(f"Model version {model_version.version} promoted to Production")
            return model_version.version
        else:
            logger.warning("No model version found for run")
            return None
            
    except Exception as e:
        logger.error(f"Error promoting model: {e}")
        return None