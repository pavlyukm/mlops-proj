# This file is now deprecated in favor of model_builder.py
# Kept for backward compatibility if needed

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
import logging

logger = logging.getLogger(__name__)

def train_model(model_path, X_train, y_train, X_test, y_test):
    """Legacy training function - use model_builder.py instead"""
    logger.warning("Using legacy train_model function. Consider using model_builder.py")
    
    try:
        model = load_model(model_path)
        logger.info("Loaded existing model")
    except:
        logger.info("Creating new model")
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(y_train.shape[1], activation='softmax'))
        
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    
    from sklearn.metrics import accuracy_score, classification_report
    import numpy as np
    
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    report = classification_report(y_test_classes, y_pred_classes)
    
    return model, accuracy, report