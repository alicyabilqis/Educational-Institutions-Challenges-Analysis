# Import libraries
import streamlit as st
import joblib
import pandas as pd

# Load model and scaler
model_rf = joblib.load('model_rf.joblib')
scaler = joblib.load('scaler.joblib')

# Top 10 most important features based on feature importance
top_features = [
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

# App Title
st.title('Student Dropout Risk Prediction')
st.subheader('Early Detection Through Predictive Modeling')

# Introduction
st.write("""
School dropout remains a serious challenge in education. To help address this issue, this prediction model was developed to identify students who are at risk of dropping out. 
By analyzing student data, the model can detect patterns that indicate potential risk early on. The predictions are intended to support schools in taking timely and appropriate actions to help students stay in school and complete their education.
""")

# Input form for the top 10 features
curricular_units_2nd_sem_approved = st.number_input('2nd Semester - Approved Units', min_value=0, step=1)
curricular_units_2nd_sem_grade = st.number_input('2nd Semester - Average Grade', min_value=0.0, step=0.1)
curricular_units_1st_sem_approved = st.number_input('1st Semester - Approved Units', min_value=0, step=1)
curricular_units_1st_sem_grade = st.number_input('1st Semester - Average Grade', min_value=0.0, step=0.1)
tuition_fees_up_to_date = st.selectbox('Tuition Fees Paid (1: Yes, 0: No)', [1, 0])
age_at_enrollment = st.number_input('Age at Enrollment', min_value=0, step=1)
curricular_units_2nd_sem_evaluations = st.number_input('2nd Semester - Evaluations Taken', min_value=0, step=1)
admission_grade = st.number_input('Admission Grade', min_value=0.0, step=0.1)
course = st.number_input('Course Code', min_value=0, step=1)
curricular_units_1st_sem_evaluations = st.number_input('1st Semester - Evaluations Taken', min_value=0, step=1)

# Organize input data
input_data = pd.DataFrame([{
    'Curricular_units_2nd_sem_approved': curricular_units_2nd_sem_approved,
    'Curricular_units_2nd_sem_grade': curricular_units_2nd_sem_grade,
    'Curricular_units_1st_sem_approved': curricular_units_1st_sem_approved,
    'Curricular_units_1st_sem_grade': curricular_units_1st_sem_grade,
    'Tuition_fees_up_to_date': tuition_fees_up_to_date,
    'Age_at_enrollment': age_at_enrollment,
    'Curricular_units_2nd_sem_evaluations': curricular_units_2nd_sem_evaluations,
    'Admission_grade': admission_grade,
    'Course': course,
    'Curricular_units_1st_sem_evaluations': curricular_units_1st_sem_evaluations
}])

# Predict button
if st.button('Predict Status'):
    # Ensure the column order is correct
    input_data = input_data[top_features]

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model_rf.predict(input_scaled)
    status_labels = {1: 'Graduate', 0: 'Dropout'}
    predicted_status = status_labels[prediction[0]]

    st.subheader('Prediction Result:')
    if predicted_status == 'Graduate':
        st.success(f'The student is predicted to: **{predicted_status}**')
    else:
        st.error(f'The student is predicted to: **{predicted_status}**')

    # Optional: probability output
    try:
        prediction_proba = model_rf.predict_proba(input_scaled)
        st.write('Prediction Probabilities:')
        proba_df = pd.DataFrame(prediction_proba, columns=['Dropout Probability', 'Graduate Probability'])
        st.dataframe(proba_df)
    except AttributeError:
        st.info("Probability estimates are not available for this model.")

# Reminder
st.caption("Please ensure all fields are filled before submitting.")

