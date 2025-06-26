from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, GlobalMaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import mlflow

# Enable eager execution
tf.config.run_functions_eagerly(True)

def get_text_classification_model(input_dim, num_classes=3, learning_rate=0.001):
    """
    Returns a compiled text classification model for TF-IDF input.
    """
    model = Sequential()
    
    # Input layer
    model.add(Dense(256, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(0.3))
    
    # Hidden layers
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    
    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Log model architecture to MLflow
    if mlflow.active_run():
        mlflow.log_param("input_dim", input_dim)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("model_type", "dense_tfidf")
    
    return model

def train_with_hyperparameter_tuning(X_train, y_train, X_test, y_test, hyperparams_grid):
    """
    Train multiple models with different hyperparameters and return the best one.
    Implements champion/challenger approach.
    """
    best_model = None
    best_accuracy = 0
    best_params = None
    
    input_dim = X_train.shape[1]
    num_classes = y_train.shape[1]
    
    for params in hyperparams_grid:
        with mlflow.start_run(nested=True):
            model = get_text_classification_model(
                input_dim=input_dim,
                num_classes=num_classes,
                learning_rate=params.get('learning_rate', 0.001)
            )
            
            # Log hyperparameters
            mlflow.log_params(params)
            
            history = model.fit(
                X_train, y_train,
                epochs=params.get('epochs', 10),
                batch_size=params.get('batch_size', 32),
                validation_data=(X_test, y_test),
                verbose=1
            )
            
            # Evaluate model
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("loss", loss)
            mlflow.log_metric("val_loss", history.history['val_loss'][-1])
            mlflow.log_metric("val_accuracy", history.history['val_accuracy'][-1])
            
            # Champion/Challenger comparison
            if accuracy > best_accuracy:
                best_model = model
                best_accuracy = accuracy
                best_params = params
                mlflow.set_tag("model_type", "champion")
            else:
                mlflow.set_tag("model_type", "challenger")
    
    return best_model, best_accuracy, best_params