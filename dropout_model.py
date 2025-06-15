# student_dropout_predictor_app.py

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model and scaler
model_rf = joblib.load('model_rf.joblib')
scaler = joblib.load('scaler.joblib')

# Feature order based on top 10 important features
feature_names = [
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

# Course code descriptions
course_descriptions = {
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

st.set_page_config(page_title="Student Dropout Predictor", layout="centered")
st.title('Early Detection of Student Dropout Risk')

st.write(
    """School dropout remains a serious challenge in education.  
This prediction model was developed to help identify students at risk of dropping out, enabling early intervention and support."""
)

with st.form("dropout_form"):
    st.subheader("📝 Student Information")

    age = st.number_input("Age at Enrollment", min_value=15, max_value=70, step=1)
    admission_grade = st.number_input("Admission Grade (0 - 200)", min_value=0.0, max_value=200.0, step=0.1)

    course_code = st.number_input("Course Code (numeric)", min_value=0, step=1)
    if course_code in course_descriptions:
        st.info(f"**Selected Course:** {course_descriptions[course_code]}")
    else:
        st.warning("⚠️ Unknown Course Code. Please double-check.")

    tuition_paid = st.selectbox("Tuition Fees Up To Date", [1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')

    cu1_approved = st.number_input("1st Sem: Curricular Units Approved", min_value=0, step=1)
    cu1_grade = st.number_input("1st Sem: Average Grade", min_value=0.0, max_value=20.0, step=0.1)
    cu1_evaluations = st.number_input("1st Sem: Number of Evaluations", min_value=0, step=1)

    cu2_approved = st.number_input("2nd Sem: Curricular Units Approved", min_value=0, step=1)
    cu2_grade = st.number_input("2nd Sem: Average Grade", min_value=0.0, max_value=20.0, step=0.1)
    cu2_evaluations = st.number_input("2nd Sem: Number of Evaluations", min_value=0, step=1)

    submitted = st.form_submit_button("🔍 Predict Dropout Status")

    if submitted:
        # Check for missing or invalid values (simple example)
        input_values = [age, admission_grade, course_code, tuition_paid, cu1_approved,
                        cu1_grade, cu1_evaluations, cu2_approved, cu2_grade, cu2_evaluations]

        if any(val is None or (isinstance(val, float) and np.isnan(val)) for val in input_values):
            st.error("❌ Please fill in all fields correctly.")
        else:
            input_dict = {
                'Age_at_enrollment': age,
                'Admission_grade': admission_grade,
                'Course': course_code,
                'Tuition_fees_up_to_date': tuition_paid,
                'Curricular_units_1st_sem_approved': cu1_approved,
                'Curricular_units_1st_sem_grade': cu1_grade,
                'Curricular_units_1st_sem_evaluations': cu1_evaluations,
                'Curricular_units_2nd_sem_approved': cu2_approved,
                'Curricular_units_2nd_sem_grade': cu2_grade,
                'Curricular_units_2nd_sem_evaluations': cu2_evaluations
            }

            input_df = pd.DataFrame([input_dict])[feature_names]
            input_scaled = scaler.transform(input_df)
            prediction = model_rf.predict(input_scaled)[0]

            labels = {1: 'Graduate', 0: 'Dropout'}
            pred_label = labels[prediction]

            st.subheader("🎯 Prediction Result:")
            if pred_label == 'Graduate':
                st.success(f"The student is predicted to **{pred_label}**.")
            else:
                st.error(f"The student is at risk of **{pred_label}**.")

            # Optional: prediction probabilities
            try:
                prob = model_rf.predict_proba(input_scaled)[0]
                st.write("📊 Prediction Probabilities:")
                st.write(f"Dropout: {prob[0]:.2%}, Graduate: {prob[1]:.2%}")
            except Exception:
                pass
