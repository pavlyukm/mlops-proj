from fastapi import FastAPI, Request
from pydantic import BaseModel
import mlflow
import logging
from dotenv import load_dotenv
import tensorflow as tf
import os
import sys

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Enable eager execution at the very beginning
tf.config.run_functions_eagerly(True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

# Lazy load services to avoid initialization issues
training_service = None
prediction_service = None

def get_training_service():
    global training_service
    if training_service is None:
        from services.training_service import TrainingService
        training_service = TrainingService()
    return training_service

def get_prediction_service():
    global prediction_service
    if prediction_service is None:
        from services.prediction_service import PredictionService
        prediction_service = PredictionService()
    return prediction_service

@app.on_event("startup")
async def startup_event():
    """Initialize MLflow on startup"""
    from utils.mlflow_utils import setup_mlflow_with_s3
    setup_mlflow_with_s3()

@app.get("/")
async def root():
    return {"message": "Customer Support Ticket Classification API. Visit /docs for documentation."}

@app.post("/train")
async def train(request: Request):
    logger.info("Received request at /train endpoint")
    
    try:
        service = get_training_service()
        result = service.train_model()
        return {
            "message": "Training completed successfully",
            "accuracy": result["accuracy"],
            "best_params": result["best_params"],
            "model_version": result.get("model_version"),
            "report": result["report"]
        }
    except Exception as e:
        logger.error(f"Error in /train endpoint: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"error": str(e)}

@app.get("/labels")
async def get_labels():
    try:
        service = get_prediction_service()
        return service.get_labels()
    except Exception as e:
        logger.error(f"Error in /labels endpoint: {e}")
        return {"error": str(e)}

class PredictionInput(BaseModel):
    customer_email: str
    product_purchased: str
    ticket_subject: str
    ticket_priority: str
    combined_text: str

@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        logger.info("Received request at /predict endpoint")
        service = get_prediction_service()
        
        from utils.mlflow_utils import mlflow_available
        if mlflow_available:
            with mlflow.start_run():
                return service.predict(input_data)
        else:
            return service.predict(input_data)
            
    except Exception as e:
        logger.error(f"Error in /predict endpoint: {e}")
        return {"error": str(e)}