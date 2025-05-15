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
from load_data.load_data import load_and_preprocess_data
from process_data.process_data import evaluate_model
from model.model import train_model
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

@app.get("/")
async def root():
    return {"message": """
    Welcome to my crappy ML app :)

    Visit /docs for Swagger UI documentation.
    """}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Load the pre-trained model
model = load_model('best_model.h5')

# Load the TF-IDF vocabulary
try:
    tfidf_vocab = np.load('tfidf_vocabulary.npy', allow_pickle=True)
    logger.info("File loaded successfully: %s", tfidf_vocab)
    tfidf_vectorizer = TfidfVectorizer(vocabulary=tfidf_vocab)
except Exception as e:
    logger.error("Error loading file: %s", e)
    tfidf_vectorizer = TfidfVectorizer()  # Initialize without vocabulary if loading fails

# Load the label encoder classes
try:
    label_encoder_classes = np.load('label_encoder_classes.npy', allow_pickle=True)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = label_encoder_classes
except Exception as e:
    logger.error("Error loading label encoder classes: %s", e)
    label_encoder = LabelEncoder()  # Initialize without classes if loading fails

# Load environment variables for AWS
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')

# Initialize S3 client
s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY, region_name=AWS_REGION)

@app.post("/train")
async def train(request: Request):
    logger.info("Received request at /train endpoint")
    try:
        bucket_name = 'pavliukmmlops'
        file_key = 'train-00000-of-00001.parquet'

        logger.info("Downloading file from S3")
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        data = response['Body'].read()

        logger.info("Loading and preprocessing data")
        X_train, X_test, y_train, y_test, tfidf_vectorizer, label_encoder = load_and_preprocess_data(BytesIO(data))

        logger.info("Training model with new data")
        model, accuracy, report = train_model('best_model.h5', X_train, y_train, X_test, y_test)

        # Save the updated model
        model.save('best_model.h5')

        return {"message": "Training completed successfully", "accuracy": accuracy, "report": report}
    except Exception as e:
        logger.error("Error in /train endpoint: %s", str(e))
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
        # Preprocess the input data
        df = pd.DataFrame([{
            'Customer Email': input_data.customer_email,
            'Product Purchased': input_data.product_purchased,
            'Ticket Subject': input_data.ticket_subject,
            'Ticket Priority': input_data.ticket_priority,
            'Combined Text': input_data.combined_text
        }])

        # Vectorize text features
        X_text = tfidf_vectorizer.transform(df['Combined Text'] + " " + df['Ticket Subject'])

        # Combine features
        X = pd.concat([
            df[['Product Purchased', 'Ticket Priority']].reset_index(drop=True),
            pd.DataFrame(X_text.toarray())
        ], axis=1)

        X.columns = X.columns.astype(str)

        # Make predictions
        y_pred = model.predict(X)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Decode the predictions
        y_pred_labels = label_encoder.inverse_transform(y_pred_classes)

        return {"predictions": y_pred_labels.tolist()}
    except Exception as e:
        logger.error("Error in /predict endpoint: %s", str(e))
        return {"error": str(e)}
