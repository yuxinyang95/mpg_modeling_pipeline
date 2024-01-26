from flask import Flask, request, jsonify
import pandas as pd
from joblib import load


def create_app(model_path):
    app = Flask(__name__)
    model = load(model_path)

    @app.route('/')
    def healthcheck():
        return "The Server is Up"

    @app.route('/predict', methods=['POST'])
    def predict():
        # Receive JSON array from the request body
        json_data = request.json

        # Convert JSON array to DataFrame
        X_pred = pd.DataFrame(json_data)

        # Make predictions
        predictions = model.predict(X_pred)
        # Return the predictions as a JSON response
        return jsonify({'predictions': predictions.tolist()})

    return app


if __name__ == '__main__':
    app = create_app('ridge_model.joblib')
    app.run(debug=True)
