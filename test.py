import pytest
from train_model import train_and_save_ridge_model
from server import app
def test_model_training():
    clf = train_and_save_ridge_model()
    # Add tests to check if the model is trained and saved correctly
    print("Best Parameters:", clf.best_params_)

    # Print best score
    print("Best Score:", clf.best_score_)

    # Access and print details of the best estimator
    best_model = clf.best_estimator_
    print("Best Estimator:", best_model)
    feature_names = best_model.feature_names_in_
    print("Feature Names:", feature_names)

    # Print detailed results of the grid search
    print("Grid Search Results:")
    print(clf.cv_results_)

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_predict_endpoint(client):
    return