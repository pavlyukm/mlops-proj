import tensorflow as tf
import mlflow
import mlflow.keras
import mlflow.xgboost
import os
import logging
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import tempfile
import xgboost as xgb

logger = logging.getLogger(__name__)

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

def create_xgboost_model(num_classes, **hyperparams):
    """Create XGBoost model with custom hyperparameters"""
    default_params = {
        'objective': 'multi:softprob',
        'num_class': num_classes,
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'eval_metric': 'mlogloss'
    }
    
    # Override with provided hyperparameters
    default_params.update(hyperparams)
    
    model = xgb.XGBClassifier(**default_params)
    return model

def train_xgboost_model(X_train, X_test, y_train, y_test, num_classes, **hyperparams):
    """Train XGBoost model with MLflow tracking"""
    
    if not setup_mlflow():
        logger.warning("MLflow not available, training without tracking")
    
    with mlflow.start_run() as run:
        try:
            # Convert sparse matrices to dense
            X_train_dense = X_train.toarray()
            X_test_dense = X_test.toarray()
            
            # Create model with hyperparameters
            model = create_xgboost_model(num_classes, **hyperparams)
            
            # Log all parameters
            mlflow.log_param("model_type", "xgboost")
            mlflow.log_param("input_dim", X_train_dense.shape[1])
            mlflow.log_param("num_classes", num_classes)
            mlflow.log_param("train_samples", len(X_train_dense))
            mlflow.log_param("test_samples", len(X_test_dense))
            
            # Log hyperparameters
            for param, value in hyperparams.items():
                mlflow.log_param(f"hp_{param}", value)
            
            # Log default parameters that weren't overridden
            for param, value in model.get_params().items():
                mlflow.log_param(param, value)
            
            # Train model
            model.fit(
                X_train_dense, y_train,
                eval_set=[(X_test_dense, y_test)],
                verbose=True
            )
            
            # Evaluate model
            y_pred = model.predict(X_test_dense)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Get probabilities for more detailed evaluation
            y_pred_proba = model.predict_proba(X_test_dense)
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("test_accuracy", accuracy)
            
            # Log classification report
            report = classification_report(y_test, y_pred)
            mlflow.log_text(report, "classification_report.txt")
            
            # Log model to MLflow
            mlflow.xgboost.log_model(
                model, 
                "model",
                registered_model_name="customer_support_classifier"
            )
            
            # Save model locally as backup
            os.makedirs('model', exist_ok=True)
            model.save_model('model/classifier.json')
            
            logger.info(f"XGBoost model training completed. Accuracy: {accuracy:.4f}")
            
            return model, accuracy, run.info.run_id
            
        except Exception as e:
            logger.error(f"XGBoost model training failed: {e}")
            mlflow.log_param("error", str(e))
            raise

def get_current_champion():
    """Get current champion model info from MLflow model registry"""
    try:
        client = mlflow.tracking.MlflowClient()
        
        # Get all versions of the model
        model_versions = client.get_latest_versions(
            "customer_support_classifier",
            stages=["Production"]
        )
        
        if model_versions:
            champion_version = model_versions[0]
            return {
                'version': champion_version.version,
                'run_id': champion_version.run_id,
                'stage': champion_version.current_stage,
                'accuracy': None  # We'll get this from the run
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Error getting current champion: {e}")
        return None

def get_model_accuracy_from_run(run_id):
    """Get accuracy metric from MLflow run"""
    try:
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        
        # Try different metric names
        for metric_name in ['accuracy', 'test_accuracy', 'final_accuracy']:
            if metric_name in run.data.metrics:
                return run.data.metrics[metric_name]
        
        return None
        
    except Exception as e:
        logger.error(f"Error getting accuracy from run {run_id}: {e}")
        return None

def promote_model_to_champion(run_id, accuracy):
    """Promote model to champion if it's better than current champion"""
    try:
        client = mlflow.tracking.MlflowClient()
        
        # Get current champion
        current_champion = get_current_champion()
        
        # Get model version from run
        model_versions = client.search_model_versions(
            f"run_id='{run_id}'"
        )
        
        if not model_versions:
            logger.error(f"No model version found for run {run_id}")
            return False
        
        new_model_version = model_versions[0]
        
        # Compare with current champion
        should_promote = False
        promotion_reason = ""
        
        if current_champion is None:
            should_promote = True
            promotion_reason = "No current champion"
        else:
            # Get current champion's accuracy
            current_accuracy = get_model_accuracy_from_run(current_champion['run_id'])
            
            if current_accuracy is None:
                should_promote = True
                promotion_reason = "Cannot determine current champion accuracy"
            elif accuracy > current_accuracy:
                should_promote = True
                promotion_reason = f"New model accuracy ({accuracy:.4f}) > Champion accuracy ({current_accuracy:.4f})"
            else:
                promotion_reason = f"New model accuracy ({accuracy:.4f}) <= Champion accuracy ({current_accuracy:.4f})"
        
        if should_promote:
            # Demote current champion to challenger
            if current_champion:
                try:
                    client.transition_model_version_stage(
                        name="customer_support_classifier",
                        version=current_champion['version'],
                        stage="Staging"
                    )
                    
                    # Update tags
                    client.set_model_version_tag(
                        name="customer_support_classifier",
                        version=current_champion['version'],
                        key="role",
                        value="challenger"
                    )
                    
                    logger.info(f"Demoted version {current_champion['version']} to challenger")
                    
                except Exception as e:
                    logger.warning(f"Failed to demote current champion: {e}")
            
            # Promote new model to champion
            client.transition_model_version_stage(
                name="customer_support_classifier",
                version=new_model_version.version,
                stage="Production"
            )
            
            # Update tags
            client.set_model_version_tag(
                name="customer_support_classifier",
                version=new_model_version.version,
                key="role",
                value="champion"
            )
            
            client.set_model_version_tag(
                name="customer_support_classifier",
                version=new_model_version.version,
                key="accuracy",
                value=str(accuracy)
            )
            
            logger.info(f"Promoted version {new_model_version.version} to champion")
            logger.info(f"Promotion reason: {promotion_reason}")
            
            return {
                'promoted': True,
                'version': new_model_version.version,
                'reason': promotion_reason,
                'accuracy': accuracy
            }
        else:
            # Tag as challenger
            client.set_model_version_tag(
                name="customer_support_classifier",
                version=new_model_version.version,
                key="role",
                value="challenger"
            )
            
            client.set_model_version_tag(
                name="customer_support_classifier",
                version=new_model_version.version,
                key="accuracy",
                value=str(accuracy)
            )
            
            logger.info(f"Model version {new_model_version.version} tagged as challenger")
            logger.info(f"Not promoted: {promotion_reason}")
            
            return {
                'promoted': False,
                'version': new_model_version.version,
                'reason': promotion_reason,
                'accuracy': accuracy
            }
            
    except Exception as e:
        logger.error(f"Error in model promotion: {e}")
        return {
            'promoted': False,
            'error': str(e)
        }

def load_champion_model():
    """Load the current champion model"""
    try:
        client = mlflow.tracking.MlflowClient()
        
        # Get production model (champion)
        model_versions = client.get_latest_versions(
            "customer_support_classifier",
            stages=["Production"]
        )
        
        if model_versions:
            champion_version = model_versions[0]
            model_uri = f"models:/customer_support_classifier/{champion_version.version}"
            
            # Load model
            model = mlflow.xgboost.load_model(model_uri)
            
            logger.info(f"Loaded champion model version {champion_version.version}")
            return model
        
        # Fallback to local model
        if os.path.exists('model/classifier.json'):
            model = xgb.XGBClassifier()
            model.load_model('model/classifier.json')
            logger.info("Loaded model from local file")
            return model
        
        logger.error("No champion model found")
        return None
        
    except Exception as e:
        logger.error(f"Error loading champion model: {e}")
        return None

def get_model_registry_info():
    """Get information about models in the registry"""
    try:
        client = mlflow.tracking.MlflowClient()
        
        # Get all versions
        all_versions = client.get_latest_versions("customer_support_classifier")
        
        info = {
            'model_name': 'customer_support_classifier',
            'total_versions': len(all_versions),
            'champion': None,
            'challenger': None,
            'all_versions': []
        }
        
        for version in all_versions:
            version_info = {
                'version': version.version,
                'stage': version.current_stage,
                'run_id': version.run_id,
                'tags': dict(version.tags) if version.tags else {}
            }
            
            # Get accuracy from run
            accuracy = get_model_accuracy_from_run(version.run_id)
            if accuracy:
                version_info['accuracy'] = accuracy
            
            info['all_versions'].append(version_info)
            
            # Identify champion and challenger
            if version.current_stage == "Production":
                info['champion'] = version_info
            elif version.current_stage == "Staging":
                info['challenger'] = version_info
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting model registry info: {e}")
        return {'error': str(e)}

# Backward compatibility functions
def train_model(X_train, X_test, y_train, y_test, num_classes, **hyperparams):
    """Train model and handle promotion automatically"""
    model, accuracy, run_id = train_xgboost_model(
        X_train, X_test, y_train, y_test, num_classes, **hyperparams
    )
    
    # Attempt automatic promotion
    promotion_result = promote_model_to_champion(run_id, accuracy)
    
    return model, accuracy, run_id, promotion_result

def load_production_model():
    """Load the production model (champion)"""
    return load_champion_model()

def create_model(input_dim, num_classes, **hyperparams):
    """Create model with hyperparameters"""
    return create_xgboost_model(num_classes, **hyperparams)