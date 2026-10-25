# Import necessary libraries
import io
import numpy as np
import joblib
import pandas as pd

from flask import Flask, request, jsonify, send_file


# ---------------------------------------------------------
# Flask Application
# ---------------------------------------------------------

superkart_model_api = Flask("SuperKart Sales Revenue Predictor")


# ---------------------------------------------------------
# Load Trained Model
# ---------------------------------------------------------

model = joblib.load("superkart_model.joblib")


# ---------------------------------------------------------
# Required Model Features
# ---------------------------------------------------------

REQUIRED_FEATURES = [
    "Product_Weight",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Establishment_Year",
    "Product_Sugar_Content",
    "Product_Type",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type"
]


# ---------------------------------------------------------
# Valid Categorical Values
# ---------------------------------------------------------

VALID_CATEGORIES = {

    "Product_Sugar_Content": [
        "Low Sugar",
        "No Sugar",
        "Regular",
        "reg"
    ],

    "Product_Type": [
        "Baking Goods",
        "Breads",
        "Breakfast",
        "Canned",
        "Dairy",
        "Frozen Foods",
        "Fruits and Vegetables",
        "Hard Drinks",
        "Health and Hygiene",
        "Household",
        "Meat",
        "Others",
        "Seafood",
        "Snack Foods",
        "Soft Drinks",
        "Starchy Foods"
    ],

    "Store_Size": [
        "High",
        "Medium",
        "Small"
    ],

    "Store_Location_City_Type": [
        "Tier 1",
        "Tier 2",
        "Tier 3"
    ],

    "Store_Type": [
        "Departmental Store",
        "Food Mart",
        "Supermarket Type1",
        "Supermarket Type2"
    ]
}


NUMERIC_FEATURES = [
    "Product_Weight",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Establishment_Year"
]


# ---------------------------------------------------------
# Validate Prediction Input
# ---------------------------------------------------------

def validate_input(df):

    if df.empty:
        return "Input data is empty."

    missing_columns = [
        column
        for column in REQUIRED_FEATURES
        if column not in df.columns
    ]

    if missing_columns:
        return (
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    if df[REQUIRED_FEATURES].isnull().any().any():
        return "Input contains missing values."

    for column in NUMERIC_FEATURES:

        converted = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        if converted.isnull().any():
            return f"Invalid numeric values in {column}."

        if not np.isfinite(converted.to_numpy(dtype=float)).all():
            return f"Non-finite numeric values in {column}."

        df[column] = converted

    for column, valid_values in VALID_CATEGORIES.items():

        invalid_values = df.loc[
            ~df[column].isin(valid_values),
            column
        ].unique()

        if len(invalid_values) > 0:

            return (
                f"Invalid values in {column}: "
                f"{invalid_values.tolist()}"
            )

    return None


# ---------------------------------------------------------
# Home Endpoint
# ---------------------------------------------------------

@superkart_model_api.get("/")
def home():

    return (
        "Welcome to the SuperKart Sales "
        "Revenue Outlets Prediction API!"
    )


# ---------------------------------------------------------
# Single Prediction Endpoint
# ---------------------------------------------------------

@superkart_model_api.post("/v1/predict")
def predict_sales_revenue():

    product_data = request.get_json(silent=True)

    if not isinstance(product_data, dict):

        return jsonify({
            "error": "A valid JSON object is required."
        }), 400

    input_data = pd.DataFrame([product_data])

    validation_error = validate_input(input_data)

    if validation_error:

        return jsonify({
            "error": validation_error
        }), 400

    try:

        model_input = input_data[REQUIRED_FEATURES]

        prediction = float(
            model.predict(model_input)[0]
        )

        if not np.isfinite(prediction):
            raise ValueError("Non-finite model prediction.")

        predicted_sales = round(prediction, 2)

        return jsonify({
            "Predicted Sales (in dollars)": predicted_sales
        })

    except Exception:

        superkart_model_api.logger.exception(
            "Single prediction failed"
        )

        return jsonify({
            "error": "Prediction failed. Check backend logs."
        }), 500


# ---------------------------------------------------------
# Batch Prediction Endpoint
# ---------------------------------------------------------

@superkart_model_api.post("/v1/predictbatch")
def predict_sales_batch():

    if "file" not in request.files:

        return jsonify({
            "error": "No CSV file uploaded."
        }), 400

    uploaded_file = request.files["file"]

    if not uploaded_file.filename:

        return jsonify({
            "error": "No file selected."
        }), 400

    if not uploaded_file.filename.lower().endswith(".csv"):

        return jsonify({
            "error": "Only CSV files are supported."
        }), 400

    try:

        input_data = pd.read_csv(uploaded_file)

    except (pd.errors.ParserError, UnicodeError, ValueError):

        return jsonify({
            "error": "Unable to read CSV file."
        }), 400

    except pd.errors.EmptyDataError:

        return jsonify({
            "error": "Uploaded CSV file is empty."
        }), 400

    validation_error = validate_input(input_data)

    if validation_error:

        return jsonify({
            "error": validation_error
        }), 400

    try:

        # Select only the nine features used during training
        model_input = input_data[REQUIRED_FEATURES]

        # Predict actual sales revenue directly
        predictions = model.predict(model_input)

        if not np.isfinite(predictions).all():
            raise ValueError("Non-finite model predictions.")

        # Preserve original CSV columns
        results = input_data.copy()

        # Add predicted sales
        results["Predicted_Product_Store_Sales_Total"] = [
            round(float(prediction), 2)
            for prediction in predictions
        ]

        # Generate CSV in memory
        output = io.BytesIO()

        results.to_csv(
            output,
            index=False
        )

        output.seek(0)

        return send_file(
            output,
            mimetype="text/csv",
            as_attachment=True,
            download_name="superkart_batch_predictions.csv"
        )

    except Exception:

        superkart_model_api.logger.exception(
            "Batch prediction failed"
        )

        return jsonify({
            "error": "Batch prediction failed. Check backend logs."
        }), 500


# ---------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------

if __name__ == "__main__":

    superkart_model_api.run(
        host="0.0.0.0",
        port=7860,
        debug=False
    )