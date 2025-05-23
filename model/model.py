import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tf.config.run_functions_eagerly(True)

def train_model(model_path, X_train, y_train, X_test, y_test):
    try:
        logger.info("Loading existing model")
        model = load_model(model_path)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        logger.info("Training model with new data")
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

        logger.info("Evaluating model")
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)

        accuracy = accuracy_score(y_test_classes, y_pred_classes)
        report = classification_report(y_test_classes, y_pred_classes)

        logger.info(f"Accuracy: {accuracy}")
        logger.info("Classification Report:")
        logger.info(report)

        return model, accuracy, report
    except Exception as e:
        logger.error("Error in train_model: %s", str(e))
        raise
