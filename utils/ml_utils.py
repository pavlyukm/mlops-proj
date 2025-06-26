import pickle
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

def load_encoders():
    """Load all encoders from pickle files"""
    encoders = {}
    
    encoder_files = {
        'product': 'encoders/product_encoder.pkl',
        'priority': 'encoders/priority_encoder.pkl',
        'ticket_type': 'encoders/ticket_type_encoder.pkl',
        'tfidf': 'vectorizer/tfidf_vectorizer.pkl'
    }
    
    for name, path in encoder_files.items():
        try:
            with open(path, 'rb') as f:
                encoders[name] = pickle.load(f)
            logger.info(f"{name} encoder loaded successfully")
        except Exception as e:
            logger.error(f"Error loading {name} encoder: {e}")
            if name == 'tfidf':
                encoders[name] = TfidfVectorizer()
            else:
                encoders[name] = LabelEncoder()
    
    return encoders

def save_encoders(encoders):
    """Save encoders to pickle files"""
    encoder_files = {
        'product': ('encoders/product_encoder.pkl', encoders['product']),
        'priority': ('encoders/priority_encoder.pkl', encoders['priority']),
        'ticket_type': ('encoders/ticket_type_encoder.pkl', encoders['ticket_type']),
        'tfidf': ('vectorizer/tfidf_vectorizer.pkl', encoders['tfidf'])
    }
    
    for name, (path, encoder) in encoder_files.items():
        with open(path, 'wb') as f:
            pickle.dump(encoder, f)