import logging
import numpy as np
from data_utils import DataProcessor
from model_utils import load_production_model

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.model = None
        self.load_model_and_preprocessors()
    
    def load_model_and_preprocessors(self):
        """Load model and preprocessors"""
        try:
            # Load preprocessors
            if not self.data_processor.load_preprocessors():
                raise Exception("Failed to load preprocessors")
            
            # Load model
            self.model = load_production_model()
            if self.model is None:
                raise Exception("Failed to load model")
            
            logger.info("Model and preprocessors loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model and preprocessors: {e}")
            raise
    
    def predict(self, subject, body, answer=None, ticket_type=None, priority=None, language=None, 
                tag_1=None, tag_2=None, tag_3=None, tag_4=None, tag_5=None, 
                tag_6=None, tag_7=None, tag_8=None):
        """Make prediction for a support ticket"""
        try:
            if self.model is None:
                raise Exception("Model not loaded")
            
            # Combine text (including answer if provided)
            text_parts = [subject, body]
            if answer:
                text_parts.append(answer)
            combined_text = ' '.join([part for part in text_parts if part])
            
            # Vectorize text
            text_vectorized = self.data_processor.tfidf_vectorizer.transform([combined_text])
            
            # Prepare categorical features
            categorical_features = []
            categorical_columns = ['type', 'priority', 'language', 'tag_1', 'tag_2', 'tag_3', 'tag_4', 'tag_5', 'tag_6', 'tag_7', 'tag_8']
            categorical_values = [ticket_type, priority, language, tag_1, tag_2, tag_3, tag_4, tag_5, tag_6, tag_7, tag_8]
            
            # Encode categorical features
            for col, value in zip(categorical_columns, categorical_values):
                if col in self.data_processor.categorical_encoders:
                    encoder = self.data_processor.categorical_encoders[col]
                    
                    # Handle unknown values
                    if value is None or value == '':
                        value = 'unknown'
                    
                    try:
                        encoded_value = encoder.transform([str(value)])[0]
                    except ValueError:
                        # If value not seen during training, use first class (or 'unknown' if it exists)
                        if 'unknown' in encoder.classes_:
                            encoded_value = encoder.transform(['unknown'])[0]
                        else:
                            encoded_value = 0  # Use first class as fallback
                    
                    categorical_features.append(encoded_value)
            
            # Combine features
            if categorical_features:
                import numpy as np
                from scipy import sparse
                
                categorical_array = np.array(categorical_features).reshape(1, -1)
                categorical_sparse = sparse.csr_matrix(categorical_array)
                X_combined = sparse.hstack([text_vectorized, categorical_sparse])
            else:
                X_combined = text_vectorized
            
            # Convert to dense for prediction
            X_dense = X_combined.toarray()
            
            # Make prediction
            prediction_probs = self.model.predict(X_dense)
            predicted_class_idx = np.argmax(prediction_probs[0])
            confidence = float(np.max(prediction_probs[0]))
            
            # Get class label
            predicted_label = self.data_processor.label_encoder.inverse_transform([predicted_class_idx])[0]
            
            return {
                "predicted_queue": predicted_label,
                "confidence": confidence,
                "probabilities": {
                    label: float(prob) 
                    for label, prob in zip(
                        self.data_processor.label_encoder.classes_, 
                        prediction_probs[0]
                    )
                }
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def get_model_info(self):
        """Get information about the loaded model"""
        try:
            label_info = self.data_processor.get_label_info()
            return {
                "model_loaded": self.model is not None,
                "classes": label_info['classes'] if label_info else [],
                "n_classes": label_info['n_classes'] if label_info else 0,
                "input_shape": self.model.input_shape if self.model else None
            }
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {"error": str(e)}