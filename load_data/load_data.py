import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_and_preprocess_data(file_path):
    # Load the dataset
    df = pd.read_parquet(file_path)

    # Preprocess the data
    label_encoder = LabelEncoder()
    df['Product Purchased'] = label_encoder.fit_transform(df['Product Purchased'])
    df['Ticket Priority'] = label_encoder.fit_transform(df['Ticket Priority'])
    df['Ticket Type'] = label_encoder.fit_transform(df['Ticket Type'])

    # Vectorize text features
    tfidf_vectorizer = TfidfVectorizer()
    X_text = tfidf_vectorizer.fit_transform(df['Combined Text'] + " " + df['Ticket Subject'])

    # Combine features
    X = pd.concat([
        df[['Product Purchased', 'Ticket Priority']].reset_index(drop=True),
        pd.DataFrame(X_text.toarray())
    ], axis=1)

    X.columns = X.columns.astype(str)

    y = df['Ticket Type']
    y = to_categorical(y)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, tfidf_vectorizer, label_encoder