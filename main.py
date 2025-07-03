from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import os
from dotenv import load_dotenv
from predict import PredictionService

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Customer Support Ticket Classifier",
    description="API for classifying customer support tickets",
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

class TrainingResponse(BaseModel):
    message: str
    accuracy: float
    run_id: str
    num_classes: int

@app.on_event("startup")
async def startup_event():
    """Initialize the prediction service on startup"""
    global prediction_service
    
    # Wait for MLflow to be ready
    import time
    import requests
    
    mlflow_url = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
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
        "endpoints": {
            "POST /train": "Train a new model",
            "POST /predict": "Classify a support ticket",
            "GET /model-info": "Get model information",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": prediction_service is not None and prediction_service.model is not None
    }

@app.post("/train", response_model=TrainingResponse)
async def train_model():
    """Trigger model training"""
    try:
        logger.info("Starting training process...")
        
        # Import training modules directly
        from data_utils import DataProcessor
        from model_utils import train_model as train_model_func, promote_model_to_production
        
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
        
        # Train model
        logger.info("Training model...")
        model, accuracy, run_id = train_model_func(X_train, X_test, y_train, y_test, num_classes)
        
        # Promote to production if accuracy is good enough
        if accuracy > 0.7:  # 70% threshold
            logger.info(f"Model accuracy {accuracy:.4f} is above threshold, promoting to production...")
            version = promote_model_to_production(run_id)
            if version:
                logger.info(f"Model version {version} promoted to production")
        else:
            logger.warning(f"Model accuracy {accuracy:.4f} is below threshold (0.7), not promoting to production")
        
        logger.info("Training pipeline completed successfully!")
        
        # Reload prediction service with new model
        global prediction_service
        try:
            prediction_service = PredictionService()
            logger.info("Prediction service reloaded successfully")
        except Exception as e:
            logger.warning(f"Could not reload prediction service: {e}")
        
        return TrainingResponse(
            message="Training completed successfully. Model reloaded.",
            accuracy=float(accuracy),
            run_id=run_id,
            num_classes=int(num_classes)
        )
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

@app.post("/predict")
async def predict_ticket(request: PredictionRequest):
    """Classify a customer support ticket"""
    try:
        if prediction_service is None or prediction_service.model is None:
            raise HTTPException(
                status_code=503, 
                detail="Model not loaded. Please train a model first using POST /train"
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
    """Get information about the current model"""
    try:
        if prediction_service is None:
            return {"model_loaded": False, "message": "Prediction service not initialized"}
        
        return prediction_service.get_model_info()
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)