import joblib
import pandas as pd
from data.data_cleaning import clean_data

def calculate_priority_scores(df, model_path, preproc_path):
    """
    Calculate priority scores for water pumps based on model predictions and key factors.

    """
    # Load model artifacts
    model = joblib.load(model_path)
    preprocessor = joblib.load(preproc_path)

    # Clean and preprocess data
    clean_df = clean_data(df)
    processed_data = preprocessor.transform(clean_df)

    # Get model predictions
    proba = model.predict_proba(processed_data)
    predictions = model.predict(processed_data)

    # Prepare results
    results = df.copy()
    results['status_group'] = pd.Series(predictions).map({
        2: 'functional',
        1: 'functional needs repair',
        0: 'non functional'
    })

    # Calculate priority factors
    results['failure_prob'] = proba[:, 0]
    results['repair_prob'] = proba[:, 1]
    results['norm_population'] = df['population'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))

    quality_scores = {
        'soft': 1, 'salty': 0.5, 'milky': 0.5,
        'unknown': 0.5, 'coloured': 0.5, 'fluoride': 0.5
    }
    results['quality_score'] = df['water_quality'].map(quality_scores)

    quantity_scores = {
        'enough': 1, 'seasonal': 0.75,
        'insufficient': 0.5, 'dry': 0.2
    }
    results['quantity_score'] = df['quantity'].map(quantity_scores)

    # Calculate final priority score
    results['priority_score'] = (
        results['failure_prob'] +
        results['repair_prob'] +
        results['quality_score'] +
        results['quantity_score'] +
        results['norm_population']
    ) / 5

    # Filter and return
    results.loc[results['status_group'] == 'functional', 'priority_score'] = 0
    return results
