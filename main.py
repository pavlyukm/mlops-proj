from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import os
from dotenv import load_dotenv
from predict import PredictionService

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Customer Support Ticket Classifier",
    description="API for classifying customer support tickets with automatic champion/challenger promotion",
    version="2.0.0"
)

# Global prediction service
prediction_service = None

class PredictionRequest(BaseModel):
    subject: str
    body: str
    answer: str = None  # Optional answer field
    type: str = None
    priority: str = None
    language: str = None
    tag_1: str = None
    tag_2: str = None
    tag_3: str = None
    tag_4: str = None
    tag_5: str = None
    tag_6: str = None
    tag_7: str = None
    tag_8: str = None

class TrainingRequest(BaseModel):
    # XGBoost hyperparameters
    max_depth: int = 6
    learning_rate: float = 0.1
    n_estimators: int = 100
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0
    reg_lambda: float = 1
    min_child_weight: int = 1

class TrainingResponse(BaseModel):
    message: str
    accuracy: float
    run_id: str
    num_classes: int
    promotion_result: dict

@app.on_event("startup")
async def startup_event():
    """Initialize the prediction service on startup"""
    global prediction_service
    
    # Wait for MLflow to be ready
    import time
    import requests
    
    mlflow_url = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow-service:5000')
    max_retries = 30
    retry_delay = 2
    
    logger.info(f"Waiting for MLflow to be ready at {mlflow_url}...")
    
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{mlflow_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("MLflow is ready!")
                break
        except Exception as e:
            logger.info(f"MLflow not ready yet (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(retry_delay)
    else:
        logger.warning("MLflow may not be ready, continuing anyway...")
    
    # Try to initialize prediction service
    try:
        prediction_service = PredictionService()
        logger.info("Prediction service initialized successfully")
    except Exception as e:
        logger.warning(f"Could not initialize prediction service: {e}")
        logger.info("You may need to train a model first using POST /train")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Customer Support Ticket Classifier API v2.0",
        "description": "XGBoost-based classifier with automatic champion/challenger promotion",
        "endpoints": {
            "POST /train": "Train a new XGBoost model with custom hyperparameters",
            "POST /predict": "Classify a ticket using the current champion model",
            "GET /model-info": "Get information about the current champion model",
            "GET /model-registry": "Get information about all models in the registry",
            "GET /health": "Health check"
        },
        "kubeflow_integration": "Available - add Kubeflow endpoints after fixing dependencies"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_loaded = prediction_service is not None and prediction_service.model is not None
    
    return {
        "status": "healthy",
        "champion_model_loaded": model_loaded,
        "prediction_service_ready": prediction_service is not None,
        "mlflow_tracking_uri": os.getenv('MLFLOW_TRACKING_URI'),
        "message": "Basic API working - Kubeflow integration pending dependency fix"
    }

@app.post("/train", response_model=TrainingResponse)
async def train_model(hyperparams: TrainingRequest = TrainingRequest()):
    """Train a new XGBoost model with custom hyperparameters"""
    try:
        logger.info("Starting XGBoost model training...")
        
        # Import training modules
        from data_utils import DataProcessor
        from model_utils import train_model
        
        # Initialize data processor
        data_processor = DataProcessor()
        
        # Load and preprocess data
        logger.info("Loading data from S3...")
        df = data_processor.load_data_from_s3()
        
        logger.info("Preprocessing data...")
        X_train, X_test, y_train, y_test, num_classes = data_processor.preprocess_data(df)
        
        # Save preprocessors
        data_processor.save_preprocessors()
        logger.info("Preprocessors saved to encoders/ folder")
        
        # Convert hyperparameters to dict
        hyperparams_dict = hyperparams.dict()
        
        # Train model with custom hyperparameters
        logger.info(f"Training XGBoost model with hyperparameters: {hyperparams_dict}")
        model, accuracy, run_id, promotion_result = train_model(
            X_train, X_test, y_train, y_test, num_classes, **hyperparams_dict
        )
        
        logger.info("Training completed successfully!")
        
        # Reload prediction service if model was promoted
        if promotion_result.get('promoted', False):
            global prediction_service
            try:
                prediction_service = PredictionService()
                logger.info("Prediction service reloaded with new champion model")
            except Exception as e:
                logger.warning(f"Could not reload prediction service: {e}")
        
        return TrainingResponse(
            message=f"Training completed. Model {'promoted to champion' if promotion_result.get('promoted') else 'added as challenger'}",
            accuracy=float(accuracy),
            run_id=run_id,
            num_classes=int(num_classes),
            promotion_result=promotion_result
        )
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

@app.post("/predict")
async def predict_ticket(request: PredictionRequest):
    """Classify a customer support ticket using the current champion model"""
    try:
        if prediction_service is None or prediction_service.model is None:
            raise HTTPException(
                status_code=503, 
                detail="Champion model not loaded. Please train a model first using POST /train"
            )
        
        result = prediction_service.predict(
            subject=request.subject,
            body=request.body,
            answer=request.answer,
            ticket_type=request.type,
            priority=request.priority,
            language=request.language,
            tag_1=request.tag_1,
            tag_2=request.tag_2,
            tag_3=request.tag_3,
            tag_4=request.tag_4,
            tag_5=request.tag_5,
            tag_6=request.tag_6,
            tag_7=request.tag_7,
            tag_8=request.tag_8
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get information about the current champion model"""
    try:
        if prediction_service is None:
            return {
                "champion_model_loaded": False,
                "message": "Prediction service not initialized"
            }
        
        return prediction_service.get_model_info()
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@app.get("/model-registry")
async def get_model_registry():
    """Get information about all models in the MLflow registry"""
    try:
        from model_utils import get_model_registry_info
        
        registry_info = get_model_registry_info()
        return registry_info
        
    except Exception as e:
        logger.error(f"Error getting model registry info: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting model registry info: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)