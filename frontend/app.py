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
# Required Model Features
# ---------------------------------------------------------

REQUIRED_FEATURES = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category"
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
            value=12.66,
            step=0.1
        )

        Product_Allocated_Area = st.number_input(
            "Product Allocated Area",
            min_value=0.0,
            value=0.027,
            step=0.001,
            format="%.3f"
        )

        Product_MRP = st.number_input(
            "Product MRP",
            min_value=0.0,
            value=117.08,
            step=0.01
        )

        Store_Age_Years = st.number_input(
            "Store Age (Years)",
            min_value=0,
            max_value=150,
            value=16,
            step=1
        )

        Product_Sugar_Content = st.selectbox(
            "Product Sugar Content",
            [
                "Low Sugar",
                "No Sugar",
                "Regular"
            ]
        )

    with col2:

        Product_Id_char = st.selectbox(
            "Product ID Category",
            [
                "FD",
                "DR",
                "NC"
            ]
        )

        Product_Type_Category = st.selectbox(
            "Product Type Category",
            [
                "Non Perishables",
                "Perishables"
            ]
        )

        Store_Size = st.selectbox(
            "Store Size",
            [
                "High",
                "Medium",
                "Small"
            ],
            index=1
        )

        Store_Location_City_Type = st.selectbox(
            "Store Location City Type",
            [
                "Tier 1",
                "Tier 2",
                "Tier 3"
            ],
            index=1
        )

        Store_Type = st.selectbox(
            "Store Type",
            [
                "Departmental Store",
                "Food Mart",
                "Supermarket Type1",
                "Supermarket Type2",
                "Supermarket Type3"
            ],
            index=3
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
        "Product_Sugar_Content": Product_Sugar_Content,
        "Product_Allocated_Area": Product_Allocated_Area,
        "Product_MRP": Product_MRP,
        "Store_Size": Store_Size,
        "Store_Location_City_Type": Store_Location_City_Type,
        "Store_Type": Store_Type,
        "Product_Id_char": Product_Id_char,
        "Store_Age_Years": Store_Age_Years,
        "Product_Type_Category": Product_Type_Category
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

            st.success(
                "Prediction completed successfully!"
            )

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
    "Upload a CSV file containing the 10 required "
    "model features to generate sales revenue predictions."
)


# ---------------------------------------------------------
# CSV Template Download
# ---------------------------------------------------------

template_data = pd.DataFrame([
    {
        "Product_Weight": 12.66,
        "Product_Sugar_Content": "Low Sugar",
        "Product_Allocated_Area": 0.027,
        "Product_MRP": 117.08,
        "Store_Size": "Medium",
        "Store_Location_City_Type": "Tier 2",
        "Store_Type": "Supermarket Type2",
        "Product_Id_char": "FD",
        "Store_Age_Years": 16,
        "Product_Type_Category": "Non Perishables"
    }
])

template_data = template_data[REQUIRED_FEATURES]

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

                        # API returns a dictionary:
                        # {"0": 2945.54, "1": 4810.87}

                        predictions = response.json()

                        # Preserve the order of uploaded rows
                        prediction_values = [
                            predictions[str(index)]
                            for index in range(len(preview_data))
                        ]

                        # Append predictions to uploaded data
                        results = preview_data.copy()

                        results[
                            "Predicted_Product_Store_Sales_Total"
                        ] = prediction_values

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

                        # Convert the results to CSV
                        csv_results = results.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            label="Download Prediction Results",
                            data=csv_results,
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

                except (ValueError, KeyError, TypeError) as error:

                    st.error(
                        f"Unable to process prediction results: {error}"
                    )

    except (pd.errors.ParserError, UnicodeError, ValueError) as error:

        st.error(
            f"Unable to read uploaded CSV file: {error}"
        )