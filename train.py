from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from data.data_preprocessing import preprocess_data
from data.data_cleaning import clean_data
import joblib


def train_model(df, model_path='models/random_forest_model.pkl', random_state=42):
    """
    Model training
    """
    # Clean df
    df = clean_data(df)
    target_map = {'functional': 2, 'functional needs repair': 1, 'non functional': 0}
    df['status_group'] = df['status_group'].map(target_map)

    X = df.drop(columns=['status_group'])
    y = df['status_group']

    # Preprocessing
    preprocessor = preprocess_data(X)
    X_transformed = preprocessor.transform(X)

    # SMOTE Balancing
    smote = SMOTE(random_state=random_state)
    X_sampled, y_sampled = smote.fit_resample(X_transformed, y)

    # Training
    model = RandomForestClassifier(
        n_estimators=100,
        min_samples_split=5,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_sampled, y_sampled)

    # save model
    joblib.dump(model, model_path)

    return model
