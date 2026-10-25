# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize the Flask application
superkart_model_api = Flask("SuperKart Sale Revevue Outlets Predictor")

# Load the trained machine learning model
model = joblib.load("superkart_model.joblib")

# Define a route for the home page (GET request)
@superkart_model_api.get('/')
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a simple welcome message.
    """
    return "Welcome to the SuperKart Sales Revenue Outlets Prediction API!"

# Define an endpoint for single property prediction (POST request)
@superkart_model_api.post('/v1/predict')
def predict_sales_revenue():

    product_data = request.get_json()

    required_features = [
        'Product_Weight',
        'Product_Allocated_Area',
        'Product_MRP',
        'Store_Establishment_Year',
        'Product_Sugar_Content',
        'Product_Type',
        'Store_Size',
        'Store_Location_City_Type',
        'Store_Type'
    ]

    if not isinstance(product_data, dict):
        return jsonify({
            'error': 'A JSON object containing product features is required.'
        }), 400

    missing_features = [
        feature for feature in required_features
        if feature not in product_data
    ]

    if missing_features:
        return jsonify({
            'error': 'Missing required features',
            'missing_features': missing_features
        }), 400

    sample = {
        feature: product_data[feature]
        for feature in required_features
    }

    input_data = pd.DataFrame([sample])

    # Predict sales revenue directly
    raw_prediction = float(model.predict(input_data)[0])

    # Ensure the model returned a valid number
    if not np.isfinite(raw_prediction):
        return jsonify({
            'error': 'Model generated an invalid prediction.'
        }), 500

    # Round predicted sales to two decimal places
    predicted_sales = round(raw_prediction, 2)

    return jsonify({
        'Predicted Sales (in dollars)': predicted_sales
})

# Define an endpoint for batch prediction (POST request)
@superkart_model_api.post('/v1/predictbatch')
def predict_sales_batch():
    """
    This function handles POST requests to the '/v1/predictbatch' endpoint.
    It expects a CSV file containing property details for multiple properties
    and returns the predicted rental prices as a dictionary in the JSON response.
    """
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the CSV file into a Pandas DataFrame
    input_data = pd.read_csv(file)

    # Make predictions for all properties in the DataFrame (get log_prices)
    predicted_log_sales = model.predict(input_data).tolist()

    # Calculate actual prices
    predicted_sales = [round(float(np.exp(log_sales)), 2) for log_sales in predicted_log_sales]

    # Create a dictionary of predictions with property IDs as keys
    product_ids = input_data['Product_id'].tolist()  # Assuming 'id' is the product id column
    output_dict = dict(zip(product_ids, predicted_sales))  # Use actual sales

    # Return the predictions dictionary as a JSON response
    return output_dict



# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    superkart_model_api.run(debug=True)
