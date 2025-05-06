# Water Pump Status Predictor

A machine learning pipeline to predict maintenance needs for Tanzanian water pumps.

## Key Modules

### Data Preparation
- `cleaning.py`: Handles missing data, outliers, and feature selection
- `preprocessing.py`: Feature scaling and encoding

### Machine Learning
- `training.py`:
  - Trains Random Forest classifier
  - Handles class imbalance with SMOTE
  - Saves model
- `prediction.py`: Loads model and makes predictions
- `eval.py`: evaluate model accury

### Scoring formula
- 'scoring.py': assign priority score to new df
