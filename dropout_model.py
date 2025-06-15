import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the saved model and scaler
try:
    model = joblib.load('model_rf_top_features.joblib')
    scaler = joblib.load('scaler_top_features.joblib')

top_features_list = [
    'Curricular_units_2nd_sem_approved',
    'Curricular_units_2nd_sem_grade',
    'Curricular_units_1st_sem_approved',
    'Curricular_units_1st_sem_grade',
    'Tuition_fees_up_to_date',
    'Age_at_enrollment',
    'Curricular_units_2nd_sem_evaluations',
    'Admission_grade',
    'Course',
    'Curricular_units_1st_sem_evaluations'
]


except FileNotFoundError:
    st.error("Model or scaler file not found. Please ensure 'model_rf_top_features.joblib' and 'scaler_top_features.joblib' exist.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()


# Title of the app
st.title("Student Dropout/Graduation Predictor")

st.write("""
This app predicts whether a student is likely to **Graduate** or **Dropout**
based on key academic and personal factors.
""")

# Sidebar for user input
st.sidebar.header("Input Student Data")

# Dictionary to hold input data
input_data = {}

input_data['Curricular_units_2nd_sem_approved'] = st.sidebar.number_input(
    'Curricular Units Approved in 2nd Sem',
    min_value=0, value=5, step=1
)
input_data['Curricular_units_1st_sem_approved'] = st.sidebar.number_input(
    'Curricular Units Approved in 1st Sem',
    min_value=0, value=5, step=1
)
input_data['Curricular_units_2nd_sem_grade'] = st.sidebar.number_input(
    'Average Grade in 2nd Sem (0-200)',
    min_value=0.0, max_value=200.0, value=120.0, step=0.1
)
input_data['Curricular_units_1st_sem_grade'] = st.sidebar.number_input(
    'Average Grade in 1st Sem (0-200)',
    min_value=0.0, max_value=200.0, value=120.0, step=0.1
)
input_data['Tuition_fees_up_to_date'] = st.sidebar.selectbox(
    'Tuition Fees Up to Date?', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No'
)
input_data['Age_at_enrollment'] = st.sidebar.number_input(
    'Age at Enrollment (Years)',
    min_value=17, value=18, step=1
)
input_data['Admission_grade'] = st.sidebar.number_input(
    'Admission Grade (0-200)',
    min_value=0.0, max_value=200.0, value=130.0, step=0.1
)
input_data['Curricular_units_1st_sem_evaluations'] = st.sidebar.number_input(
    'Curricular Units Evaluated in 1st Sem',
    min_value=0, value=6, step=1
)
input_data['Curricular_units_2nd_sem_evaluations'] = st.sidebar.number_input(
    'Curricular Units Evaluated in 2nd Sem',
    min_value=0, value=6, step=1
)
input_data['Course'] = st.sidebar.number_input(
    'Course (Numerical Code)', min_value=1, value=1, step=1
)


# Create a DataFrame from the input data, ensuring the column order matches the training data
# It's essential that the column order is correct for the scaler and the model
input_df = pd.DataFrame([input_data])

# Ensure the order of columns in input_df matches the top_features_list
input_df = input_df[top_features_list]

# Scale the input data
scaled_input = scaler.transform(input_df)

# Make prediction
if st.sidebar.button('Predict Status'):
    try:
        prediction = model.predict(scaled_input)
        prediction_proba = model.predict_proba(scaled_input)

        # Map prediction output back to original labels
        # Remember: 1 was mapped to 'Graduate', 0 to 'Dropout'
        status_mapping = {1: 'Graduate', 0: 'Dropout'}
        predicted_status = status_mapping[prediction[0]]

        st.subheader("Prediction Result")

        if predicted_status == 'Graduate':
            st.success(f"The student is predicted to **{predicted_status}**.")
            # Show probability for graduate
            prob_graduate = prediction_proba[0][1] # Probability for class 1 (Graduate)
            st.write(f"Confidence: {prob_graduate:.2f}")
        else:
            st.error(f"The student is predicted to **{predicted_status}**.")
            # Show probability for dropout
            prob_dropout = prediction_proba[0][0] # Probability for class 0 (Dropout)
            st.write(f"Confidence: {prob_dropout:.2f}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.markdown("---")
st.write("""
*Note: This prediction is based on a machine learning model trained on a specific dataset.
The results should be interpreted as a probability and not a definitive outcome.*
""")
