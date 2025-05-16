from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    report = classification_report(y_test_classes, y_pred_classes)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)

    return accuracy, report
