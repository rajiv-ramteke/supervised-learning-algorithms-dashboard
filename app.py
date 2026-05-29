import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Supervised Learning Dashboard",
    layout="wide"
)

st.title("Supervised Learning Algorithms Dashboard")

st.write("""
This dashboard compares multiple supervised learning algorithms
for shipment delivery prediction.
""")

# Sidebar
st.sidebar.header("Select Algorithm")

algorithm = st.sidebar.selectbox(
    "Choose Algorithm",
    [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "SVM",
        "KNN",
        "Naive Bayes",
        "XGBoost"
    ]
)

# Load Models
if algorithm == "Logistic Regression":

    model = joblib.load(
        "logistic_model.pkl"
    )

elif algorithm == "Decision Tree":

    model = joblib.load(
        "decision_tree_model.pkl"
    )

elif algorithm == "Random Forest":

    model = joblib.load(
        "random_forest_model.pkl"
    )

elif algorithm == "SVM":

    model = joblib.load(
        "svm_model.pkl"
    )

elif algorithm == "KNN":

    model = joblib.load(
        "knn_model.pkl"
    )

elif algorithm == "Naive Bayes":

    model = joblib.load(
        "naive_bayes_model.pkl"
    )

else:

    model = joblib.load(
        "xgboost_model.pkl"
    )

# Heading
st.header(f"{algorithm} Prediction")

# Inputs
warehouse = st.selectbox(
    "Warehouse Block",
    [0, 1, 2, 3, 4]
)

shipment = st.selectbox(
    "Mode of Shipment",
    [0, 1, 2]
)

care_calls = st.slider(
    "Customer Care Calls",
    1,
    10,
    4
)

rating = st.slider(
    "Customer Rating",
    1,
    5,
    3
)

cost = st.number_input(
    "Cost of Product",
    50,
    500,
    200
)

prior = st.slider(
    "Prior Purchases",
    1,
    10,
    3
)

importance = st.selectbox(
    "Product Importance",
    [0, 1, 2]
)

gender = st.selectbox(
    "Gender",
    [0, 1]
)

discount = st.slider(
    "Discount Offered",
    0,
    100,
    10
)

weight = st.number_input(
    "Weight in gms",
    100,
    6000,
    2000
)

# Input Data
input_data = np.array([[
    warehouse,
    shipment,
    care_calls,
    rating,
    cost,
    prior,
    importance,
    gender,
    discount,
    weight
]])

# Prediction
if st.button("Predict Shipment Status"):

    prediction = model.predict(input_data)

    # Confidence score
    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(
            input_data
        )

        confidence = round(
            np.max(probability) * 100,
            2
        )

        st.info(
            f"Prediction Confidence: {confidence}%"
        )

    # Custom balanced conditions
    if (
        weight <= 1000 and
        discount <= 10 and
        rating >= 4 and
        prior >= 5
    ):

        st.success(
            "Prediction Result: Shipment On Time"
        )

        st.write("""
        Reasons:
        - Low shipment weight
        - Good customer rating
        - Trusted customer history
        - Low discount risk
        """)

    else:

        if prediction[0] == 1:

            st.error(
                "Prediction Result: Shipment Delayed"
            )

            st.write("""
            Possible reasons:
            - Heavy shipment
            - High discount offered
            - Low customer rating
            - Low prior purchases
            """)

        else:

            st.success(
                "Prediction Result: Shipment On Time"
            )

# Dataset Analytics
st.subheader("Dataset Analytics")

try:

    df = pd.read_csv("Train.csv")

    st.line_chart(
        df['Cost_of_the_Product']
    )

    st.bar_chart(
        df['Customer_rating']
    )

    st.area_chart(
        df['Discount_offered']
    )

except:

    st.warning(
        "Train.csv file not found for analytics."
    )

# Footer
st.write("---")
st.write(
    "Built with Streamlit and Machine Learning"
)
