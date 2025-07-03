import pandas as pd
import boto3
import os
import logging
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        self.bucket_name = os.getenv('S3_BUCKET_NAME', 'pavliukmmlops')
        self.label_encoder = LabelEncoder()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        
    def load_data_from_s3(self, file_key='dataset-tickets-multi-lang-4-20k.csv'):
        """Load dataset from S3"""
        try:
            logger.info(f"Loading data from s3://{self.bucket_name}/{file_key}")
            
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_key)
            csv_content = response['Body'].read().decode('utf-8')
            df = pd.read_csv(StringIO(csv_content))
            
            logger.info(f"Loaded dataset with shape: {df.shape}")
            logger.info(f"Columns: {df.columns.tolist()}")
            
            return df
        except Exception as e:
            logger.error(f"Error loading data from S3: {e}")
            raise
    
    def preprocess_data(self, df):
        """Preprocess the customer support ticket dataset"""
        try:
            logger.info(f"Dataset columns: {df.columns.tolist()}")
            logger.info(f"Dataset shape: {df.shape}")
            logger.info(f"Sample data preview:\n{df.head(2)}")
            
            # Define text columns to vectorize
            text_columns = ['subject', 'body', 'answer']
            
            # Define categorical columns to encode  
            categorical_columns = ['type', 'priority', 'language', 'tag_1', 'tag_2', 'tag_3', 'tag_4', 'tag_5', 'tag_6', 'tag_7', 'tag_8']
            
            # Target column
            target_column = 'queue'
            
            # Check if required columns exist
            required_cols = ['subject', 'body', target_column]  # answer might be optional
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Clean the data - remove rows where target is null
            df = df.dropna(subset=[target_column])
            logger.info(f"After removing null targets: {df.shape}")
            
            # Process text columns and combine them
            text_data = []
            used_columns = []
            
            for col in text_columns:
                if col in df.columns:
                    # Fill missing values and convert to string
                    df[col] = df[col].fillna('').astype(str)
                    text_data.append(df[col].values)  # Get numpy array of strings
                    used_columns.append(col)
                    logger.info(f"Processed text column: {col}")
            
            # Combine text arrays element-wise
            if text_data:
                import numpy as np
                combined_text_array = []
                for i in range(len(df)):
                    # Combine text from all columns for each row
                    row_text = ' '.join([text_col[i] for text_col in text_data])
                    combined_text_array.append(row_text)
                
                # Create combined text column
                df['combined_text'] = combined_text_array
            else:
                raise ValueError("No text columns found to process")
            
            logger.info(f"Combined text from columns: {used_columns}")
            
            # Clean categorical columns - fill missing with 'unknown'
            categorical_features = []
            self.categorical_encoders = {}
            
            for col in categorical_columns:
                if col in df.columns:
                    # Fill missing values
                    df[col] = df[col].fillna('unknown').astype(str)
                    
                    # Encode categorical variable
                    encoder = LabelEncoder()
                    encoded_values = encoder.fit_transform(df[col])
                    categorical_features.append(encoded_values)
                    self.categorical_encoders[col] = encoder
                    
                    logger.info(f"Encoded {col}: {len(encoder.classes_)} classes")
            
            # Vectorize combined text
            X_text = self.tfidf_vectorizer.fit_transform(df['combined_text'])
            
            # Combine text features with categorical features
            if categorical_features:
                import numpy as np
                from scipy import sparse
                
                # Stack categorical features
                categorical_matrix = np.column_stack(categorical_features)
                
                # Convert to sparse matrix
                categorical_sparse = sparse.csr_matrix(categorical_matrix)
                
                # Combine text and categorical features
                X_combined = sparse.hstack([X_text, categorical_sparse])
            else:
                X_combined = X_text
            
            # Encode target variable
            y_encoded = self.label_encoder.fit_transform(df[target_column])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_combined, y_encoded, 
                test_size=0.2, 
                random_state=42, 
                stratify=y_encoded
            )
            
            logger.info(f"Training set size: {X_train.shape}")
            logger.info(f"Test set size: {X_test.shape}")
            logger.info(f"Number of queue classes: {len(self.label_encoder.classes_)}")
            logger.info(f"Queue classes: {self.label_encoder.classes_}")
            
            # Log feature breakdown
            text_features = X_text.shape[1]
            cat_features = len(categorical_features)
            logger.info(f"Text features: {text_features}, Categorical features: {cat_features}")
            
            return X_train, X_test, y_train, y_test, len(self.label_encoder.classes_)
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise
    
    def save_preprocessors(self):
        """Save all fitted preprocessors"""
        os.makedirs('encoders', exist_ok=True)
        
        # Save main encoders
        joblib.dump(self.label_encoder, 'encoders/label_encoder.pkl')
        joblib.dump(self.tfidf_vectorizer, 'encoders/tfidf_vectorizer.pkl')
        
        # Save categorical encoders
        joblib.dump(self.categorical_encoders, 'encoders/categorical_encoders.pkl')
        
        logger.info("All preprocessors saved successfully")
    
    def load_preprocessors(self):
        """Load all fitted preprocessors"""
        try:
            self.label_encoder = joblib.load('encoders/label_encoder.pkl')
            self.tfidf_vectorizer = joblib.load('encoders/tfidf_vectorizer.pkl')
            self.categorical_encoders = joblib.load('encoders/categorical_encoders.pkl')
            logger.info("All preprocessors loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading preprocessors: {e}")
            return False
    
    def get_label_info(self):
        """Get information about the labels"""
        try:
            if hasattr(self.label_encoder, 'classes_'):
                return {
                    'classes': self.label_encoder.classes_.tolist(),
                    'n_classes': len(self.label_encoder.classes_)
                }
            else:
                return None
        except Exception as e:
            logger.error(f"Error getting label info: {e}")
            return None