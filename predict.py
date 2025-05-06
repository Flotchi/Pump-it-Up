import joblib
import pandas as pd
from data.data_cleaning import clean_data

def predict_new_data(model_path, preprocessor_path, new_data):
    """
    Predicts water pump status for new data using our RF model."""

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    new_data = clean_data(new_data)
    X_transformed = preprocessor.transform(new_data)

    predictions = model.predict(X_transformed)

    return predictions
