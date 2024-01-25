from flask import Flask, request, jsonify
from joblib import load
import pandas as pd
app = Flask(__name__)
model = load('ridge_model.joblib')


@app.route('/')
def healthCheck():
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


if __name__ == '__main__':
    app.run(debug=True)
