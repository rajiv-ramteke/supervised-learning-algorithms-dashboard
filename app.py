
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Supervised Learning Dashboard",
    layout="wide"
)

st.title("Supervised Learning Algorithms Dashboard")

# Select Algorithm
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

# Load Model
if algorithm == "Logistic Regression":
    model = joblib.load("logistic_model.pkl")

elif algorithm == "Decision Tree":
    model = joblib.load("decision_tree_model.pkl")

elif algorithm == "Random Forest":
    model = joblib.load("random_forest_model.pkl")

elif algorithm == "SVM":
    model = joblib.load("svm_model.pkl")

elif algorithm == "KNN":
    model = joblib.load("knn_model.pkl")

elif algorithm == "Naive Bayes":
    model = joblib.load("naive_bayes_model.pkl")

else:
    model = joblib.load("xgboost_model.pkl")

st.header(f"{algorithm} Prediction")

warehouse = st.selectbox(
    "Warehouse Block",
    [0,1,2,3,4]
)

shipment = st.selectbox(
    "Mode of Shipment",
    [0,1,2]
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
    [0,1,2]
)

gender = st.selectbox(
    "Gender",
    [0,1]
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

if st.button("Predict"):

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Shipment Delayed")
    else:
        st.success("Shipment On Time")
