import logging
import numpy as np
from data_utils import DataProcessor
from model_utils import load_champion_model, get_model_registry_info

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.model = None
        self.model_version = None
        self.load_model_and_preprocessors()
    
    def load_model_and_preprocessors(self):
        """Load champion model and preprocessors"""
        try:
            # Load preprocessors
            if not self.data_processor.load_preprocessors():
                raise Exception("Failed to load preprocessors")
            
            # Load champion model
            self.model = load_champion_model()
            if self.model is None:
                raise Exception("Failed to load champion model")
            
            # Get model version info
            try:
                registry_info = get_model_registry_info()
                if registry_info and 'champion' in registry_info:
                    self.model_version = registry_info['champion']['version']
                    logger.info(f"Loaded champion model version {self.model_version}")
                else:
                    logger.info("Champion model loaded (version unknown)")
            except Exception as e:
                logger.warning(f"Could not get model version info: {e}")
            
            logger.info("Champion model and preprocessors loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model and preprocessors: {e}")
            raise
    
    def predict(self, subject, body, answer=None, ticket_type=None, priority=None, language=None, 
                tag_1=None, tag_2=None, tag_3=None, tag_4=None, tag_5=None, 
                tag_6=None, tag_7=None, tag_8=None):
        """Make prediction using the champion model"""
        try:
            if self.model is None:
                raise Exception("Champion model not loaded")
            
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
            
            # Convert to dense for XGBoost
            X_dense = X_combined.toarray()
            
            # Make prediction
            prediction_probs = self.model.predict_proba(X_dense)
            predicted_class_idx = np.argmax(prediction_probs[0])
            confidence = float(np.max(prediction_probs[0]))
            
            # Get class label
            predicted_label = self.data_processor.label_encoder.inverse_transform([predicted_class_idx])[0]
            
            return {
                "predicted_queue": predicted_label,
                "confidence": confidence,
                "model_type": "xgboost",
                "model_version": self.model_version,
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
        """Get information about the current champion model"""
        try:
            label_info = self.data_processor.get_label_info()
            
            base_info = {
                "champion_model_loaded": self.model is not None,
                "model_type": "xgboost",
                "model_version": self.model_version,
                "classes": label_info['classes'] if label_info else [],
                "n_classes": label_info['n_classes'] if label_info else 0,
            }
            
            # Add XGBoost-specific info
            if self.model is not None:
                try:
                    base_info.update({
                        "n_estimators": self.model.n_estimators,
                        "max_depth": self.model.max_depth,
                        "learning_rate": self.model.learning_rate,
                        "subsample": self.model.subsample,
                        "colsample_bytree": self.model.colsample_bytree,
                        "reg_alpha": self.model.reg_alpha,
                        "reg_lambda": self.model.reg_lambda,
                        "min_child_weight": self.model.min_child_weight
                    })
                except Exception as e:
                    logger.warning(f"Could not get XGBoost parameters: {e}")
            
            # Add registry info
            try:
                registry_info = get_model_registry_info()
                if registry_info:
                    base_info["registry_info"] = registry_info
            except Exception as e:
                logger.warning(f"Could not get registry info: {e}")
            
            return base_info
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {"error": str(e)}