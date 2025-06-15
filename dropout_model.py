import streamlit as st
import joblib
import pandas as pd

# Load model and scaler
model_rf = joblib.load('model_rf.joblib')
scaler = joblib.load('scaler.joblib')

# Ordered top features based on your input
top_features = [
    'Age_at_enrollment',
    'Admission_grade',
    'Course',
    'Tuition_fees_up_to_date',
    'Curricular_units_1st_sem_approved',
    'Curricular_units_1st_sem_grade',
    'Curricular_units_1st_sem_evaluations',
    'Curricular_units_2nd_sem_approved',
    'Curricular_units_2nd_sem_grade',
    'Curricular_units_2nd_sem_evaluations'
]

# Course options with labels
course_options = {
    33: 'Biofuel Production Technologies',
    171: 'Animation and Multimedia Design',
    8014: 'Social Service (Evening)',
    9003: 'Agronomy',
    9070: 'Communication Design',
    9085: 'Veterinary Nursing',
    9119: 'Informatics Engineering',
    9130: 'Equinculture',
    9147: 'Management',
    9238: 'Social Service',
    9254: 'Tourism',
    9500: 'Nursing',
    9556: 'Oral Hygiene',
    9670: 'Advertising and Marketing Management',
    9773: 'Journalism and Communication',
    9853: 'Basic Education',
    9991: 'Management (Evening)'
}

# App Title
st.title('Predicting Student Dropout Risk Using Key Academic Indicators')

# Introduction
st.write("""
School dropout is a persistent challenge in education systems worldwide.  
This tool leverages machine learning to help identify students who may be at risk of dropping out.  
By focusing on the most relevant academic and financial indicators, schools can take early action to support students.
""")

# Input fields for the 10 features
age_at_enrollment = st.number_input('Age at Enrollment', min_value=0, step=1)
admission_grade = st.number_input('Admission Grade', min_value=0.0, step=0.1)

# Course selection with descriptions
course_label = st.selectbox('Course of Study', options=list(course_options.values()))
course_code = [code for code, name in course_options.items() if name == course_label][0]

tuition_fees_up_to_date = st.selectbox('Tuition Fees Paid (1: Yes, 0: No)', [1, 0])
curricular_units_1st_sem_approved = st.number_input('1st Semester - Approved Units', min_value=0, step=1)
curricular_units_1st_sem_grade = st.number_input('1st Semester - Average Grade', min_value=0.0, step=0.1)
curricular_units_1st_sem_evaluations = st.number_input('1st Semester - Evaluations Taken', min_value=0, step=1)
curricular_units_2nd_sem_approved = st.number_input('2nd Semester - Approved Units', min_value=0, step=1)
curricular_units_2nd_sem_grade = st.number_input('2nd Semester - Average Grade', min_value=0.0, step=0.1)
curricular_units_2nd_sem_evaluations = st.number_input('2nd Semester - Evaluations Taken', min_value=0, step=1)

# Organize inputs into DataFrame
input_data = pd.DataFrame([{
    'Age_at_enrollment': age_at_enrollment,
    'Admission_grade': admission_grade,
    'Course': course_code,
    'Tuition_fees_up_to_date': tuition_fees_up_to_date,
    'Curricular_units_1st_sem_approved': curricular_units_1st_sem_approved,
    'Curricular_units_1st_sem_grade': curricular_units_1st_sem_grade,
    'Curricular_units_1st_sem_evaluations': curricular_units_1st_sem_evaluations,
    'Curricular_units_2nd_sem_approved': curricular_units_2nd_sem_approved,
    'Curricular_units_2nd_sem_grade': curricular_units_2nd_sem_grade,
    'Curricular_units_2nd_sem_evaluations': curricular_units_2nd_sem_evaluations
}])

# Predict button
if st.button('Predict Status'):
    # Ensure column order is correct
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

    # Optional: Show prediction probabilities
    try:
        prediction_proba = model_rf.predict_proba(input_scaled)
        st.write('Prediction Probabilities:')
        proba_df = pd.DataFrame(prediction_proba, columns=['Dropout Probability', 'Graduate Probability'])
        st.dataframe(proba_df)
    except AttributeError:
        st.info("This model does not provide probability estimates.")

# Note
st.caption("Ensure all fields are completed before running the prediction.")
