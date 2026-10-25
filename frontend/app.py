import streamlit as st
import pandas as pd
import requests
import io


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BACKEND_URL = "http://host.docker.internal:7860"

st.set_page_config(
    page_title="SuperKart Sales Prediction",
    page_icon="🛒",
    layout="wide"
)


# ---------------------------------------------------------
# Application Header
# ---------------------------------------------------------

st.title("🛒 SuperKart Sales Revenue Prediction")

st.write(
    "Predict product sales revenue using a trained "
    "Random Forest machine learning model."
)


# ---------------------------------------------------------
# Required Features
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
# Single Prediction
# ---------------------------------------------------------

st.header("Single Product Prediction")

with st.form("single_prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

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

    with col2:

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

    predict_button = st.form_submit_button(
        "Predict Sales Revenue",
        type="primary"
    )


# ---------------------------------------------------------
# Single Prediction Request
# ---------------------------------------------------------

if predict_button:

    payload = {
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

    try:

        with st.spinner("Generating prediction..."):

            response = requests.post(
                f"{BACKEND_URL}/v1/predict",
                json=payload,
                timeout=120
            )

        if response.status_code == 200:

            prediction = response.json()[
                "Predicted Sales (in dollars)"
            ]

            st.success("Prediction completed successfully!")

            st.metric(
                label="Predicted Sales Revenue",
                value=f"${prediction:,.2f}"
            )

        else:

            try:
                error_message = response.json().get(
                    "error",
                    response.text
                )
            except ValueError:
                error_message = response.text

            st.error(
                f"Prediction failed: {error_message}"
            )

    except requests.exceptions.RequestException as error:

        st.error(
            f"Unable to connect to prediction API: {error}"
        )


# ---------------------------------------------------------
# Batch Prediction
# ---------------------------------------------------------

st.divider()

st.header("Batch Prediction")

st.write(
    "Upload a CSV file containing multiple products "
    "to generate sales revenue predictions."
)


# ---------------------------------------------------------
# CSV Template Download
# ---------------------------------------------------------

template_data = pd.DataFrame([
    {
        "Product_Weight": 10.0,
        "Product_Allocated_Area": 0.1,
        "Product_MRP": 100.0,
        "Store_Establishment_Year": 2010,
        "Product_Sugar_Content": "Low Sugar",
        "Product_Type": "Dairy",
        "Store_Size": "Medium",
        "Store_Location_City_Type": "Tier 1",
        "Store_Type": "Supermarket Type1"
    }
])

st.download_button(
    label="Download Sample CSV Template",
    data=template_data.to_csv(index=False),
    file_name="superkart_batch_template.csv",
    mime="text/csv"
)


# ---------------------------------------------------------
# File Upload
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        preview_data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data Preview")

        st.dataframe(
            preview_data.head(10),
            use_container_width=True
        )

        st.write(
            f"Total records uploaded: {len(preview_data)}"
        )

        missing_columns = [
            column
            for column in REQUIRED_FEATURES
            if column not in preview_data.columns
        ]

        if missing_columns:

            st.error(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )

        else:

            st.success(
                "All required model features are present."
            )

            if st.button(
                "Generate Batch Predictions",
                type="primary"
            ):

                uploaded_file.seek(0)

                try:

                    with st.spinner(
                        "Generating batch predictions..."
                    ):

                        response = requests.post(
                            f"{BACKEND_URL}/v1/predictbatch",
                            files={
                                "file": (
                                    uploaded_file.name,
                                    uploaded_file,
                                    "text/csv"
                                )
                            },
                            timeout=120
                        )

                    if response.status_code == 200:

                        results = pd.read_csv(
                            io.BytesIO(response.content)
                        )

                        st.success(
                            "Batch predictions completed successfully!"
                        )

                        st.subheader("Prediction Results")

                        st.dataframe(
                            results,
                            use_container_width=True
                        )

                        st.write(
                            f"Total predictions generated: {len(results)}"
                        )

                        st.download_button(
                            label="Download Prediction Results",
                            data=response.content,
                            file_name="superkart_batch_predictions.csv",
                            mime="text/csv"
                        )

                    else:

                        try:
                            error_message = response.json().get(
                                "error",
                                response.text
                            )
                        except ValueError:
                            error_message = response.text

                        st.error(
                            f"Batch prediction failed: {error_message}"
                        )

                except requests.exceptions.RequestException as error:

                    st.error(
                        f"Unable to connect to batch prediction API: {error}"
                    )

    except Exception as error:

        st.error(
            f"Unable to read uploaded CSV file: {error}"
        )