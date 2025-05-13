from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Load the pre-trained model
model = load_model('best_model.h5')

# Load the TF-IDF vocabulary
try:
    tfidf_vocab = np.load('tfidf_vocabulary.npy', allow_pickle=True)
    print("File loaded successfully:", tfidf_vocab)
    tfidf_vectorizer = TfidfVectorizer(vocabulary=tfidf_vocab)
except Exception as e:
    print("Error loading file:", e)
    tfidf_vectorizer = TfidfVectorizer()  # Initialize without vocabulary if loading fails

# Load the label encoder classes
try:
    label_encoder_classes = np.load('label_encoder_classes.npy', allow_pickle=True)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = label_encoder_classes
except Exception as e:
    print("Error loading label encoder classes:", e)
    label_encoder = LabelEncoder()  # Initialize without classes if loading fails

class InputData(BaseModel):
    file_path: str

@app.post("/predict")
async def predict(input_data: InputData):
    file_path = input_data.file_path

    # Load and preprocess the input data
    df = pd.read_parquet(file_path)
    df['Product Purchased'] = label_encoder.transform(df['Product Purchased'])
    df['Ticket Priority'] = label_encoder.transform(df['Ticket Priority'])

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
