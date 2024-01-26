import os
import pytest
from train_model import train_and_save_ridge_model
from server import create_app

@pytest.mark.usefixtures("setup_and_teardown")
class TestRidgeModel:
    model_save_path = "test_ridge_model.joblib"
    metrics_save_path = "test_model_metrics.json"

    def test_successful_model_training(self):
        """Test successful model training and saving."""
        clf = train_and_save_ridge_model(model_save_path=self.model_save_path,
                                         metrics_save_path=self.metrics_save_path)

        assert 'alpha' in clf.best_params_
        assert os.path.exists(self.model_save_path)

    def test_metrics_saving(self):
        """Test if metrics are saved correctly."""
        train_and_save_ridge_model(model_save_path=self.model_save_path,
                                   metrics_save_path=self.metrics_save_path)

        assert os.path.exists(self.metrics_save_path)

    @pytest.fixture
    def client(self):
        app = create_app(self.model_save_path)
        with app.test_client() as client:
            yield client

    def test_health_check(self, client):
        """Test the health check endpoint."""
        response = client.get('/')
        assert response.status_code == 200
        assert response.data.decode('utf-8') == "The Server is Up"

    @pytest.fixture(scope="class")
    def setup_and_teardown(self):
        """Setup and teardown for test class."""
        yield
        if os.path.exists(self.model_save_path):
            os.remove(self.model_save_path)
        if os.path.exists(self.metrics_save_path):
            os.remove(self.metrics_save_path)


if __name__ == '__main__':
    TestRidgeModel()
