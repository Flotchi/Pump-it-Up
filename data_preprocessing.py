import pandas as pd
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
import joblib

def preprocess_data(df, save_path='preprocessor.joblib'):
    """
    Preprocesses data by scaling numerical features and encoding categorical features.

    Return:
    ColumnTransformer
        Fitted preprocessor object that can transform new data

    """

    # Identify numerical and categorical columns
    num_cols = df.select_dtypes(include=['int', 'float']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    # Create preprocessing transformer
    preprocessor = ColumnTransformer([
        # Scale numerical features (robust to outliers)
        ('num', RobustScaler(), num_cols),

        # One-hot encode categorical features
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    # Fit to data and save processor
    preprocessor.fit(df)
    joblib.dump(preprocessor, save_path)

    return preprocessor
