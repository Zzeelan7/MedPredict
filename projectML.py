import streamlit as st
import joblib
import numpy as np

st.title("Multi-Disease Prediction App")

disease = st.selectbox("Select a disease to predict", 
                       ["Diabetes", "Heart Disease", "Parkinson's"])

if disease == "Diabetes":
    model = joblib.load("Random Forest_diabetes_model.pkl")
    pregnancies = st.number_input("Pregnancies", 0, 20)
    glucose = st.number_input("Glucose", 0, 200)
    bmi = st.number_input("BMI", 0.00, 50.00)
    age = st.number_input("Age", 0, 150)
    if st.button("Predict"):
        features = np.array([[pregnancies, glucose, bmi, age]])
        prediction = model.predict(features)
        st.write("Result:", "Positive" if prediction[0] == 1 else "Negative")

elif disease == "Heart Disease":
    model = joblib.load("Random Forest_heart_disease_model.pkl")
    chl = st.number_input("cholestoral", 0, 500)
    mhr = st.number_input("Max_heart_rate", 0, 200)
    age = st.number_input("age", 0, 150)
    if st.button("Predict"):
        features = np.array([[chl, mhr, age]])
        prediction = model.predict(features)
        st.write("Result:", "Positive" if prediction[0] == 1 else "Negative")

elif disease == "Parkinson's":
    model = joblib.load("Logistic Regression_parkinson_model.pkl")
    mdvpj = st.number_input("MDVP:Jitter(%)", 0.000, 1.000)
    mdvps = st.number_input("MDVP:Shimmer", 0.000, 1.000)
    rpde = st.number_input("RPDE", 0.000, 1.000)
    s1 = st.number_input("spread1", -10.000, 0.000)
    s2 = st.number_input("spread2", 0.000, 10.000)
    d2 = st.number_input("D2", 0.000, 10.000)
    ppe = st.number_input("PPE", 0.000, 1.000)
    if st.button("Predict"):
        features = np.array([[mdvpj, mdvps, rpde, s1, s2, d2, ppe]])
        prediction = model.predict(features)
        st.write("Result:", "Positive" if prediction[0] == 1 else "Negative")