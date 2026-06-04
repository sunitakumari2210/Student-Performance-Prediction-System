import streamlit as st
import joblib

model = joblib.load("student_model.pkl")

st.title("Student Performance Prediction System")

study_hours = st.number_input("Enter Study Hours")

attendance = st.number_input("Enter Attendance Percentage")

previous_score = st.number_input("Enter Previous Score")

if st.button("Predict"):

    prediction = model.predict(
        [[study_hours, attendance, previous_score]]
    )

    if prediction[0] == 1:
        st.success("Student Will Pass")
    else:
        st.error("Student Will Fail")