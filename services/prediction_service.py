import pandas as pd
import numpy as np
import mlflow
import logging
from utils.ml_utils import load_encoders
from utils.mlflow_utils import load_production_model

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self):
        self.encoders = load_encoders()
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the production model from MLflow or fallback to local"""
        self.model = load_production_model("ticket_classification_model")
        if self.model is None:
            # Fallback to loading from local file
            try:
                from tensorflow.keras.models import load_model
                # Try new keras format first
                if os.path.exists('model/best_model.keras'):
                    self.model = load_model('model/best_model.keras')
                    logger.info("Loaded model from local .keras file")
                elif os.path.exists('model/best_model.h5'):
                    self.model = load_model('model/best_model.h5')
                    logger.info("Loaded model from local .h5 file")
                else:
                    logger.error("No model file found")
            except Exception as e:
                logger.error(f"Failed to load any model: {e}")
    
    def predict(self, input_data):
        """Make prediction for a single input"""
        if mlflow.active_run():
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
        
        # Encode categorical features
        try:
            df['Product Purchased'] = self.encoders['product'].transform([df['Product Purchased'][0]])[0]
        except ValueError as ve:
            raise ValueError(f"Unseen label in 'Product Purchased': {df['Product Purchased'][0]}")
        
        try:
            df['Ticket Priority'] = self.encoders['priority'].transform([df['Ticket Priority'][0]])[0]
        except ValueError as ve:
            raise ValueError(f"Unseen label in 'Ticket Priority': {df['Ticket Priority'][0]}")
        
        # Transform text features
        X_text = self.encoders['tfidf'].transform(df['Combined Text'] + " " + df['Ticket Subject'])
        
        # Combine features
        X = pd.concat([
            df[['Product Purchased', 'Ticket Priority']].reset_index(drop=True),
            pd.DataFrame(X_text.toarray())
        ], axis=1)
        
        X.columns = X.columns.astype(str)
        
        # Make prediction
        y_pred = self.model.predict(X)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_pred_labels = self.encoders['ticket_type'].inverse_transform(y_pred_classes)
        
        if mlflow.active_run():
            mlflow.log_param("predicted_class", y_pred_labels[0])
            mlflow.log_metric("prediction_confidence", float(np.max(y_pred)))
        
        return {"predictions": y_pred_labels.tolist()}
    
    def get_labels(self):
        """Get available labels for each encoder"""
        return {
            "product_purchased_labels": self.encoders['product'].classes_.tolist(),
            "ticket_priority_labels": self.encoders['priority'].classes_.tolist(),
            "ticket_type_labels": self.encoders['ticket_type'].classes_.tolist()
        }