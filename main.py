from fastapi import FastAPI, Request
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import boto3
from io import BytesIO
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv
from data_loading_and_processing.data_loading_and_processing import load_and_preprocess_data
from evalute_data.evalute_data import evaluate_model
from model.model import train_model
import pickle
import logging
import mlflow
import mlflow.tensorflow
import mlflow.sklearn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize MLflow with lazy connection
mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
mlflow.set_tracking_uri(mlflow_tracking_uri)

# Global flag to track if MLflow is available
mlflow_available = False

def ensure_mlflow_experiment():
    """Ensure MLflow experiment exists, with retry logic"""
    global mlflow_available
    if mlflow_available:
        return True
        
    import time
    max_retries = 5
    for attempt in range(max_retries):
        try:
            mlflow.set_experiment("customer_support_ticket_classification")
            mlflow_available = True
            logger.info("Successfully connected to MLflow")
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"MLflow connection attempt {attempt + 1} failed: {e}")
                time.sleep(2)
            else:
                logger.error(f"Failed to connect to MLflow after {max_retries} attempts: {e}")
                logger.warning("Continuing without MLflow tracking")
                return False

app = FastAPI()

@app.get("/")
async def root():
    return {"message": """
    Welcome to my crappy ML app :) Visit /docs for Swagger UI documentation.
    """}


# model and vectorizer
model = load_model('model/best_model.h5')
try:
    with open('vectorizer/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    logger.info("TF-IDF vectorizer loaded successfully")
except Exception as e:
    logger.error("Error loading TF-IDF vectorizer: %s", e)
    tfidf_vectorizer = TfidfVectorizer()

# label encoders
try:
    with open('encoders/product_encoder.pkl', 'rb') as f:
        product_encoder = pickle.load(f)
    logger.info("Product encoder loaded successfully")
except Exception as e:
    logger.error("Error loading product encoder: %s", e)
    product_encoder = LabelEncoder()

try:
    with open('encoders/priority_encoder.pkl', 'rb') as f:
        priority_encoder = pickle.load(f)
    logger.info("Priority encoder loaded successfully")
except Exception as e:
    logger.error("Error loading priority encoder: %s", e)
    priority_encoder = LabelEncoder()

try:
    with open('encoders/ticket_type_encoder.pkl', 'rb') as f:
        ticket_type_encoder = pickle.load(f)
    logger.info("Ticket type encoder loaded successfully")
except Exception as e:
    logger.error("Error loading ticket type encoder: %s", e)
    ticket_type_encoder = LabelEncoder()

# AWS
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')

s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY, region_name=AWS_REGION)

@app.post("/train")
async def train(request: Request):
    logger.info("Received request at /train endpoint")
    
    if ensure_mlflow_experiment():
        with mlflow.start_run():
            try:
                bucket_name = 'pavliukmmlops'
                file_key = 'train-00000-of-00001.parquet' # if we get more training data need to change the file name or
                # automate it somehow

                logger.info("Downloading file from S3")
                response = s3.get_object(Bucket=bucket_name, Key=file_key)
                data = response['Body'].read()

                # Log parameters
                mlflow.log_param("bucket_name", bucket_name)
                mlflow.log_param("file_key", file_key)
                mlflow.log_param("data_source", "S3")

                logger.info("Loading and preprocessing data")
                X_train, X_test, y_train, y_test, tfidf_vectorizer, product_encoder, priority_encoder, ticket_type_encoder = load_and_preprocess_data(BytesIO(data))

                # Log dataset info
                mlflow.log_param("train_samples", len(X_train))
                mlflow.log_param("test_samples", len(X_test))
                mlflow.log_param("features", X_train.shape[1])

                logger.info("Training model with new data")
                model, accuracy, report = train_model('model/best_model.h5', X_train, y_train, X_test, y_test)

                # Log metrics and artifacts
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_text(report, "classification_report.txt")
                
                # Log model
                mlflow.tensorflow.log_model(model, "model")
                
                # Log encoders
                mlflow.sklearn.log_model(tfidf_vectorizer, "tfidf_vectorizer")
                mlflow.sklearn.log_model(product_encoder, "product_encoder")
                mlflow.sklearn.log_model(priority_encoder, "priority_encoder")
                mlflow.sklearn.log_model(ticket_type_encoder, "ticket_type_encoder")

                # save the updated model
                model.save('best_model.h5')

                mlflow.log_artifact('best_model.h5', "saved_model")

                return {"message": "Training completed successfully", "accuracy": accuracy, "report": report}
            except Exception as e:
                logger.error("Error in /train endpoint: %s", str(e))
                mlflow.log_param("error", str(e))
                return {"error": str(e)}
    else:
        # Fallback without MLflow
        try:
            bucket_name = 'pavliukmmlops'
            file_key = 'train-00000-of-00001.parquet'

            logger.info("Downloading file from S3 (without MLflow tracking)")
            response = s3.get_object(Bucket=bucket_name, Key=file_key)
            data = response['Body'].read()

            logger.info("Loading and preprocessing data")
            X_train, X_test, y_train, y_test, tfidf_vectorizer, product_encoder, priority_encoder, ticket_type_encoder = load_and_preprocess_data(BytesIO(data))

            logger.info("Training model with new data")
            model, accuracy, report = train_model('model/best_model.h5', X_train, y_train, X_test, y_test)

            # save the updated model
            model.save('best_model.h5')

            return {"message": "Training completed successfully (without MLflow tracking)", "accuracy": accuracy, "report": report}
        except Exception as e:
            logger.error("Error in /train endpoint: %s", str(e))
            return {"error": str(e)}

@app.get("/labels")
async def get_labels():
    try:
        # helpful endpoint for building /predict payloads -- shows which products and priorities are avialable (and
        # prediction classes)
        return {
            "product_purchased_labels": product_encoder.classes_.tolist(),
            "ticket_priority_labels": priority_encoder.classes_.tolist(),
            "ticket_type_labels": ticket_type_encoder.classes_.tolist()
        }
    except Exception as e:
        logger.error("Error in /labels endpoint: %s", str(e))
        return {"error": str(e)}

class PredictionInput(BaseModel):
    customer_email: str
    product_purchased: str
    ticket_subject: str
    ticket_priority: str
    combined_text: str

@app.post("/predict")
async def predict(input_data: PredictionInput):
    if ensure_mlflow_experiment():
        with mlflow.start_run():
            try:
                logger.info("Received request at /predict endpoint")
                
                # Log input parameters
                mlflow.log_param("customer_email", input_data.customer_email)
                mlflow.log_param("product_purchased", input_data.product_purchased)
                mlflow.log_param("ticket_priority", input_data.ticket_priority)
                
                df = pd.DataFrame([{
                    'Customer Email': input_data.customer_email,
                    'Product Purchased': input_data.product_purchased,
                    'Ticket Subject': input_data.ticket_subject,
                    'Ticket Priority': input_data.ticket_priority,
                    'Combined Text': input_data.combined_text
                }])

                logger.info("DataFrame before transformation: %s", df)
                logger.info("Data types: %s", df.dtypes)

                try:
                    df['Product Purchased'] = product_encoder.transform([df['Product Purchased'][0]])[0]
                except ValueError as ve:
                    logger.error("Error encoding 'Product Purchased': %s", str(ve))
                    mlflow.log_param("encoding_error", f"Product Purchased: {str(ve)}")
                    return {"error": f"Unseen label in 'Product Purchased': {df['Product Purchased'][0]}"}

                try:
                    df['Ticket Priority'] = priority_encoder.transform([df['Ticket Priority'][0]])[0]
                except ValueError as ve:
                    logger.error("Error encoding 'Ticket Priority': %s", str(ve))
                    mlflow.log_param("encoding_error", f"Ticket Priority: {str(ve)}")
                    return {"error": f"Unseen label in 'Ticket Priority': {df['Ticket Priority'][0]}"}

                logger.info("DataFrame after encoding: %s", df)
                logger.info("Data types after encoding: %s", df.dtypes)

                X_text = tfidf_vectorizer.transform(df['Combined Text'] + " " + df['Ticket Subject'])

                X = pd.concat([
                    df[['Product Purchased', 'Ticket Priority']].reset_index(drop=True),
                    pd.DataFrame(X_text.toarray())
                ], axis=1)

                X.columns = X.columns.astype(str)

                logger.info("DataFrame after transformation: %s", X)
                logger.info("Data types after transformation: %s", X.dtypes)

                y_pred = model.predict(X)
                y_pred_classes = np.argmax(y_pred, axis=1)

                y_pred_labels = ticket_type_encoder.inverse_transform(y_pred_classes)

                # Log prediction results
                mlflow.log_param("predicted_class", y_pred_labels[0])
                mlflow.log_metric("prediction_confidence", float(np.max(y_pred)))

                return {"predictions": y_pred_labels.tolist()}
            except Exception as e:
                logger.error("Error in /predict endpoint: %s", str(e))
                mlflow.log_param("prediction_error", str(e))
                return {"error": str(e)}
    else:
        # Fallback without MLflow
        try:
            logger.info("Received request at /predict endpoint (without MLflow tracking)")
            
            df = pd.DataFrame([{
                'Customer Email': input_data.customer_email,
                'Product Purchased': input_data.product_purchased,
                'Ticket Subject': input_data.ticket_subject,
                'Ticket Priority': input_data.ticket_priority,
                'Combined Text': input_data.combined_text
            }])

            logger.info("DataFrame before transformation: %s", df)
            logger.info("Data types: %s", df.dtypes)

            try:
                df['Product Purchased'] = product_encoder.transform([df['Product Purchased'][0]])[0]
            except ValueError as ve:
                logger.error("Error encoding 'Product Purchased': %s", str(ve))
                return {"error": f"Unseen label in 'Product Purchased': {df['Product Purchased'][0]}"}

            try:
                df['Ticket Priority'] = priority_encoder.transform([df['Ticket Priority'][0]])[0]
            except ValueError as ve:
                logger.error("Error encoding 'Ticket Priority': %s", str(ve))
                return {"error": f"Unseen label in 'Ticket Priority': {df['Ticket Priority'][0]}"}

            logger.info("DataFrame after encoding: %s", df)
            logger.info("Data types after encoding: %s", df.dtypes)

            X_text = tfidf_vectorizer.transform(df['Combined Text'] + " " + df['Ticket Subject'])

            X = pd.concat([
                df[['Product Purchased', 'Ticket Priority']].reset_index(drop=True),
                pd.DataFrame(X_text.toarray())
            ], axis=1)

            X.columns = X.columns.astype(str)

            logger.info("DataFrame after transformation: %s", X)
            logger.info("Data types after transformation: %s", X.dtypes)

            y_pred = model.predict(X)
            y_pred_classes = np.argmax(y_pred, axis=1)

            y_pred_labels = ticket_type_encoder.inverse_transform(y_pred_classes)

            return {"predictions": y_pred_labels.tolist()}
        except Exception as e:
            logger.error("Error in /predict endpoint: %s", str(e))
            return {"error": str(e)}