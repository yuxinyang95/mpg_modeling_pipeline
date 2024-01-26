import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from joblib import dump
import json
import ssl
import numpy as np
from datetime import datetime


def train_and_save_ridge_model(metrics_save_path="metrics", model_save_path="ridge_model.joblib"):
    """
    Trains a Ridge Regression model using GridSearchCV and saves the trained model.
    :param model_save_path: Path to save the trained model.
    :param metrics_save_path: Path to save the model metrics and hyperparameters.
    """
    # Load data
    # Bypass ssl verification
    ssl._create_default_https_context = ssl._create_unverified_context
    url = "https://raw.githubusercontent.com/plotly/datasets/master/auto-mpg.csv"
    df = pd.read_csv(url)

    # Remove rows with missing values
    df_cleaned = df.dropna()

    # Preprocess data
    X = df_cleaned.drop("mpg", axis=1)
    y = df_cleaned["mpg"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define model and randomly choose value for hyperparameters
    model = Ridge()
    parameters = {'alpha': [0.001, 0.01, 0.1, 1, 10, 50, 100, 500]}
    clf = GridSearchCV(model, parameters, cv=5)

    # Train model
    clf.fit(X_train, y_train)

    # Save model
    dump(clf, model_save_path)

    # Prepare metrics for storage
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cv_results = {key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in
                  clf.cv_results_.items()}

    cv_r2_scores = cv_results['mean_test_score']

    # Save timestamp, the best alpha, best score, r2 and other info for cv
    metrics = {
        'timestamp': timestamp,
        'best_parameters': clf.best_params_,
        'best_cv_score': clf.best_score_,
        'cv_r2_scores': cv_r2_scores,
        'cv_results': cv_results
    }

    # Append metrics to existing file or create new one
    try:
        with open(metrics_save_path, 'r') as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        existing_data = []

    existing_data.append(metrics)

    with open(metrics_save_path, 'w') as file:
        json.dump(existing_data, file, indent=4)

    # Return train model for unit test or future use
    return clf


if __name__ == '__main__':
    train_and_save_ridge_model()
