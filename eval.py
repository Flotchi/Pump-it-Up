import joblib
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model_path, preproc_path, X_test, y_test):

    """Evaluate a trained model"""

    model = joblib.load(model_path)
    preproc = joblib.load(preproc_path)
    X_test_proc = preproc.transform(X_test)
    y_pred = model.predict(X_test_proc)

    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
