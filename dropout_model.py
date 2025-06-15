import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model & scaler
model_rf = joblib.load('model_rf.joblib')
scaler = joblib.load('scaler.joblib')

st.set_page_config(page_title="Student Dropout Predictor", layout="centered")
st.title('Early Detection of Student Dropout Risk')

st.write("This app predicts the risk of a student dropping out based on key academic indicators.")

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

# 🔹 Display course code with descriptions
st.write("### Available Course Codes")
cols = st.columns(3)
items = list(course_descriptions.items())
for i, col in enumerate(cols):
    with col:
        for j in range(i, len(items), 3):
            code, name = items[j]
            col.caption(f"{code} – {name}")

st.markdown("---")

# Feature order for model
feature_names = [
    'Marital_status',
    'Application_mode',
    'Application_order',
    'Course',
    'Daytime_evening_attendance',
    'Previous_qualification',
    'Previous_qualification_grade',
    'Nacionality',
    'Mothers_qualification',
    'Fathers_qualification',
    'Mothers_occupation',
    'Fathers_occupation',
    'Admission_grade',
    'Displaced',
    'Educational_special_needs',
    'Debtor',
    'Tuition_fees_up_to_date',
    'Gender',
    'Scholarship_holder',
    'Age_at_enrollment',
    'International',
    'Curricular_units_1st_sem_credited',
    'Curricular_units_1st_sem_enrolled',
    'Curricular_units_1st_sem_evaluations',
    'Curricular_units_1st_sem_approved',
    'Curricular_units_1st_sem_grade',
    'Curricular_units_1st_sem_without_evaluations',
    'Curricular_units_2nd_sem_credited',
    'Curricular_units_2nd_sem_enrolled',
    'Curricular_units_2nd_sem_evaluations',
    'Curricular_units_2nd_sem_approved',
    'Curricular_units_2nd_sem_grade',
    'Curricular_units_2nd_sem_without_evaluations',
    'Unemployment_rate',
    'Inflation_rate',
    'GDP'
]


with st.form("student_form"):
    st.subheader("📝 Input Student Information")

    age = st.number_input("Age at Enrollment", min_value=15, max_value=70, step=1)
    admission_grade = st.number_input("Admission Grade (Max 16)", min_value=0.0, max_value=16.0, step=0.1)

    course_code = st.number_input("Course Code", min_value=0, step=1)
    if course_code in course_descriptions:
        st.info(f"Selected Course: **{course_descriptions[course_code]}**")
    else:
        st.warning("⚠️ Unknown Course Code")

    tuition_paid = st.selectbox("Tuition Fees Up To Date", [1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')

    cu1_approved = st.number_input("1st Sem: Units Approved (Max 30)", min_value=0, max_value=30, step=1)
    cu1_grade = st.number_input("1st Sem: Avg Grade (Max 16)", min_value=0.0, max_value=16.0, step=0.1)
    cu1_evaluations = st.number_input("1st Sem: Evaluations (Max 33)", min_value=0, max_value=33, step=1)

    cu2_approved = st.number_input("2nd Sem: Units Approved (Max 30)", min_value=0, max_value=30, step=1)
    cu2_grade = st.number_input("2nd Sem: Avg Grade (Max 16)", min_value=0.0, max_value=16.0, step=0.1)
    cu2_evaluations = st.number_input("2nd Sem: Evaluations (Max 33)", min_value=0, max_value=33, step=1)

    submitted = st.form_submit_button("🔍 Predict Dropout Risk")

    if submitted:
        # Validate
        invalid = False
        messages = []
        if admission_grade > 16: messages.append("Admission grade cannot exceed 16."); invalid = True
        if cu1_grade > 16 or cu2_grade > 16: messages.append("Grades cannot exceed 16."); invalid = True
        if cu1_approved > 30 or cu2_approved > 30: messages.append("Approved units cannot exceed 30."); invalid = True
        if cu1_evaluations > 33 or cu2_evaluations > 33: messages.append("Evaluations cannot exceed 33."); invalid = True

        if invalid:
            for m in messages:
                st.error(m)
        else:
            input_data = pd.DataFrame([[
                age, admission_grade, course_code, tuition_paid,
                cu1_approved, cu1_grade, cu1_evaluations,
                cu2_approved, cu2_grade, cu2_evaluations
            ]], columns=feature_names)

            input_scaled = scaler.transform(input_data)
            prediction = model_rf.predict(input_scaled)[0]
            prob = model_rf.predict_proba(input_scaled)[0]

            st.subheader("🎯 Prediction Result:")
            if prediction == 1:
                st.success("✅ The student is predicted to **Graduate**.")
            else:
                st.error("⚠️ The student is at risk of **Dropping Out**.")

            st.write("📊 Prediction Probabilities:")
            st.write(f"Dropout: {prob[0]:.2%} | Graduate: {prob[1]:.2%}")
