from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def train_model(X_train, y_train, X_test, y_test, input_shape, num_classes):
    def build_model(hp):
        model = Sequential()
        model.add(Dense(units=hp.Int('units_1', min_value=32, max_value=512, step=32),
                        activation='relu', input_shape=(input_shape,)))
        model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)))
        model.add(Dense(units=hp.Int('units_2', min_value=32, max_value=256, step=32), activation='relu'))
        model.add(Dropout(rate=hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    model = build_model(hp=None)  # You need to define hp or adjust accordingly

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    report = classification_report(y_test_classes, y_pred_classes)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)

    return model, accuracy, report
