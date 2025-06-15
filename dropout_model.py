import streamlit as st
import joblib
import pandas as pd

# Load model & scaler
model_rf = joblib.load('model_rf.joblib')
scaler = joblib.load('scaler.joblib')

st.set_page_config(page_title="Student Dropout Predictor", layout="centered")
st.title('🎓 Early Detection of Student Dropout Risk')

st.write("This app predicts the risk of a student dropping out based on selected academic indicators.")

# --- Course dictionary ---
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

# --- Feature Names (full list for model input order) ---
feature_names = [
    'Marital_status', 'Application_mode', 'Application_order', 'Course', 'Daytime_evening_attendance',
    'Previous_qualification', 'Previous_qualification_grade', 'Nacionality', 'Mothers_qualification',
    'Fathers_qualification', 'Mothers_occupation', 'Fathers_occupation', 'Admission_grade', 'Displaced',
    'Educational_special_needs', 'Debtor', 'Tuition_fees_up_to_date', 'Gender', 'Scholarship_holder',
    'Age_at_enrollment', 'International', 'Curricular_units_1st_sem_credited', 'Curricular_units_1st_sem_enrolled',
    'Curricular_units_1st_sem_evaluations', 'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade',
    'Curricular_units_1st_sem_without_evaluations', 'Curricular_units_2nd_sem_credited',
    'Curricular_units_2nd_sem_enrolled', 'Curricular_units_2nd_sem_evaluations',
    'Curricular_units_2nd_sem_approved', 'Curricular_units_2nd_sem_grade',
    'Curricular_units_2nd_sem_without_evaluations', 'Unemployment_rate', 'Inflation_rate', 'GDP'
]

# --- Streamlit Form ---
with st.form("student_form"):
    st.subheader("📝 Input Student Data")

    age = st.number_input("Age at Enrollment", min_value=15, max_value=70, value=18, step=1)
    admission_grade = st.number_input("Admission Grade (0–16)", min_value=0.0, max_value=16.0, value=12.0, step=0.1)
    course_code = st.number_input("Course Code", min_value=0, step=1)

    if course_code in course_descriptions:
        st.info(f"Selected Course: **{course_descriptions[course_code]}**")
    else:
        st.warning("⚠️ Unknown Course Code")

    tuition_paid = st.selectbox("Tuition Fees Up To Date", [1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')

    cu1_approved = st.number_input("1st Sem: Units Approved", min_value=0, max_value=30, value=5, step=1)
    cu1_grade = st.number_input("1st Sem: Avg Grade (0–16)", min_value=0.0, max_value=16.0, value=10.0, step=0.1)
    cu1_evaluations = st.number_input("1st Sem: Evaluations Taken", min_value=0, max_value=33, value=6, step=1)

    cu2_approved = st.number_input("2nd Sem: Units Approved", min_value=0, max_value=30, value=4, step=1)
    cu2_grade = st.number_input("2nd Sem: Avg Grade (0–16)", min_value=0.0, max_value=16.0, value=9.0, step=0.1)
    cu2_evaluations = st.number_input("2nd Sem: Evaluations Taken", min_value=0, max_value=33, value=5, step=1)

    submitted = st.form_submit_button("🔍 Predict Dropout Risk")

    if submitted:
        # --- Default values for unused features ---
        default_values = {
            'Marital_status': 1,
            'Application_mode': 1,
            'Application_order': 1,
            'Daytime_evening_attendance': 1,
            'Previous_qualification': 1,
            'Previous_qualification_grade': 12.0,
            'Nacionality': 1,
            'Mothers_qualification': 1,
            'Fathers_qualification': 1,
            'Mothers_occupation': 1,
            'Fathers_occupation': 1,
            'Displaced': 0,
            'Educational_special_needs': 0,
            'Debtor': 0,
            'Gender': 1,
            'Scholarship_holder': 0,
            'International': 0,
            'Curricular_units_1st_sem_credited': 0,
            'Curricular_units_1st_sem_without_evaluations': 0,
            'Curricular_units_2nd_sem_credited': 0,
            'Curricular_units_2nd_sem_without_evaluations': 0,
            'Unemployment_rate': 6.5,
            'Inflation_rate': 1.5,
            'GDP': 2.0
        }

        # --- Input dictionary (merge manual + user inputs) ---
        input_data = {
            'Course': course_code,
            'Admission_grade': admission_grade,
            'Tuition_fees_up_to_date': tuition_paid,
            'Age_at_enrollment': age,
            'Curricular_units_1st_sem_enrolled': cu1_approved + 1,
            'Curricular_units_1st_sem_evaluations': cu1_evaluations,
            'Curricular_units_1st_sem_approved': cu1_approved,
            'Curricular_units_1st_sem_grade': cu1_grade,
            'Curricular_units_2nd_sem_enrolled': cu2_approved + 1,
            'Curricular_units_2nd_sem_evaluations': cu2_evaluations,
            'Curricular_units_2nd_sem_approved': cu2_approved,
            'Curricular_units_2nd_sem_grade': cu2_grade
        }

        input_dict = {**default_values, **input_data}
        input_df = pd.DataFrame([input_dict])[feature_names]  # Ensure column order matches model

        # --- Predict ---
        scaled = scaler.transform(input_df)
        prediction = model_rf.predict(scaled)[0]
        proba = model_rf.predict_proba(scaled)[0]

        st.subheader("🎯 Prediction Result:")
        if prediction == 1:
            st.success("✅ The student is predicted to **Graduate**.")
        else:
            st.error("⚠️ The student is at risk of **Dropping Out**.")

        st.write("📊 Prediction Probabilities:")
        st.write(f"Dropout: {proba[0]:.2%} | Graduate: {proba[1]:.2%}")
