import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_data(file_path):
    try:
        logger.info("Loading dataset")
        df = pd.read_parquet(file_path)

        logger.info("Preprocessing data")
        product_encoder = LabelEncoder()
        priority_encoder = LabelEncoder()
        ticket_type_encoder = LabelEncoder()

        df['Product Purchased'] = product_encoder.fit_transform(df['Product Purchased'])
        df['Ticket Priority'] = priority_encoder.fit_transform(df['Ticket Priority'])
        df['Ticket Type'] = ticket_type_encoder.fit_transform(df['Ticket Type'])

        tfidf_vectorizer = TfidfVectorizer()
        X_text = tfidf_vectorizer.fit_transform(df['Combined Text'] + " " + df['Ticket Subject'])

        with open('tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(tfidf_vectorizer, f)

        with open('product_encoder.pkl', 'wb') as f:
            pickle.dump(product_encoder, f)
        with open('priority_encoder.pkl', 'wb') as f:
            pickle.dump(priority_encoder, f)
        with open('ticket_type_encoder.pkl', 'wb') as f:
            pickle.dump(ticket_type_encoder, f)

        X = pd.concat([
            df[['Product Purchased', 'Ticket Priority']].reset_index(drop=True),
            pd.DataFrame(X_text.toarray())
        ], axis=1)

        X.columns = X.columns.astype(str)

        y = df['Ticket Type']
        y = to_categorical(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        logger.info("Data preprocessing completed successfully")
        return X_train, X_test, y_train, y_test, tfidf_vectorizer, product_encoder, priority_encoder, ticket_type_encoder
    except Exception as e:
        logger.error("Error in load_and_preprocess_data: %s", str(e))
        raise
