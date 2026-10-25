import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend
#BACKEND_URL = "http://backend:7860"
BACKEND_URL = "http://host.docker.internal:7860"
# Set the title of the Streamlit app
st.title("SuperKart Sales Revenue Prediction")

st.subheader("Online Prediction")

Product_Weight = st.number_input(
    "Product Weight",
    min_value=0.0,
    value=10.0,
    step=0.1
)

Product_Allocated_Area = st.number_input(
    "Product Allocated Area",
    min_value=0.0,
    value=0.1,
    step=0.01
)

Product_MRP = st.number_input(
    "Product MRP",
    min_value=0.0,
    value=100.0,
    step=0.1
)

Store_Establishment_Year = st.number_input(
    "Store Establishment Year",
    min_value=1900,
    max_value=2026,
    value=2010,
    step=1
)

Product_Sugar_Content = st.selectbox(
    "Product Sugar Content",
    [
        "Low Sugar",
        "No Sugar",
        "Regular",
        "reg"
    ]
)

Product_Type = st.selectbox(
    "Product Type",
    [
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
    ]
)

Store_Size = st.selectbox(
    "Store Size",
    [
        "High",
        "Medium",
        "Small"
    ]
)

Store_Location_City_Type = st.selectbox(
    "Store Location City Type",
    [
        "Tier 1",
        "Tier 2",
        "Tier 3"
    ]
)

Store_Type = st.selectbox(
    "Store Type",
    [
        "Departmental Store",
        "Food Mart",
        "Supermarket Type1",
        "Supermarket Type2"
    ]
)

input_data = {
    "Product_Weight": Product_Weight,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_MRP": Product_MRP,
    "Store_Establishment_Year": Store_Establishment_Year,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Type": Product_Type,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type
}

if st.button("Predict", type="primary"):

    try:
        response = requests.post(
            f"{BACKEND_URL}/v1/predict",
            json=input_data,
            timeout=120
        )

        if response.status_code == 200:

            prediction = response.json()[
                "Predicted Sales (in dollars)"
            ]

            st.success(
                f"Predicted Product Sales Revenue: ${prediction:,.2f}"
            )

        else:
            st.error(
                f"Prediction failed (HTTP {response.status_code}): "
                f"{response.text}"
            )

    except requests.exceptions.RequestException as e:
        st.error(f"Unable to communicate with prediction API: {e}")
        
# Section for batch prediction
st.subheader("Batch Prediction")

# Allow users to upload a CSV file for batch prediction
uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])

# Make batch prediction when the "Predict Batch" button is clicked
if uploaded_file is not None:
    if st.button("Predict Batch", type="primary"):
        response = requests.post(f"{BACKEND_URL}/v1/predictbatch", files={"file": uploaded_file})  # Send file to Flask API
        if response.status_code == 200:
            predictions = response.json()
            st.success("Batch predictions completed!")
            st.write(predictions)  # Display the predictions
        else:
            st.error("Unable to connect to the prediction API.")
