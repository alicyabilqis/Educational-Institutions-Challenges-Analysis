# Import dataset
import streamlit as st
import joblib
import pandas as pd

# Load the trained model and scaler
model_rf = joblib.load('model_rf.joblib')
scaler = joblib.load('scaler.joblib')

# Define the feature names used during training (ensure they match the order)
# Get feature names from the original dataframe X before scaling
feature_names = ['Marital_status', 'Application_mode', 'Application_order', 'Course',
                 'Daytime_evening_attendance', 'Previous_qualification',
                 'Previous_qualification_grade', 'Nacionality', 'Mothers_qualification',
                 'Fathers_qualification', 'Mothers_occupation', 'Fathers_occupation',
                 'Admission_grade', 'Displaced', 'Educational_special_needs', 'Debtor',
                 'Tuition_fees_up_to_date', 'Gender', 'Scholarship_holder',
                 'Age_at_enrollment', 'International', 'Curricular_units_1st_sem_credited',
                 'Curricular_units_1st_sem_enrolled', 'Curricular_units_1st_sem_evaluations',
                 'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade',
                 'Curricular_units_1st_sem_without_evaluations', 'Curricular_units_2nd_sem_credited',
                 'Curricular_units_2nd_sem_enrolled', 'Curricular_units_2nd_sem_evaluations',
                 'Curricular_units_2nd_sem_approved', 'Curricular_units_2nd_sem_grade',
                 'Curricular_units_2nd_sem_without_evaluations', 'Unemployment_rate',
                 'Inflation_rate', 'GDP']

st.title('Prediksi Status Kelulusan Mahasiswa')

st.write('Masukkan informasi mahasiswa untuk memprediksi status kelulusan (Dropout/Graduate).')

# Create input fields for the features
# You'll need to create input widgets for each feature in `feature_names`
# Based on your data understanding and EDA, determine the appropriate input type (number input, selectbox, etc.)
# Example for a few features:

marital_status = st.selectbox('Status Perkawinan', [1, 2, 3, 4, 5, 6])
application_order = st.number_input('Urutan Aplikasi', min_value=0, max_value=9, step=1)
daytime_evening_attendance = st.selectbox('Kehadiran (1: Siang, 0: Malam)', [1, 0])
previous_qualification_grade = st.number_input('Nilai Kualifikasi Sebelumnya', min_value=0.0, max_value=200.0, step=0.1)
admission_grade = st.number_input('Nilai Penerimaan', min_value=0.0, max_value=200.0, step=0.1)
age_at_enrollment = st.number_input('Usia Saat Pendaftaran', min_value=0, step=1)
tuition_fees_up_to_date = st.selectbox('Pembayaran Uang Kuliah Lancar (1: Ya, 0: Tidak)', [1, 0])
gender = st.selectbox('Jenis Kelamin (1: Laki-laki, 0: Perempuan)', [1, 0])
scholarship_holder = st.selectbox('Penerima Beasiswa (1: Ya, 0: Tidak)', [1, 0])
international = st.selectbox('Mahasiswa Internasional (1: Ya, 0: Tidak)', [1, 0])
curricular_units_1st_sem_approved = st.number_input('Unit Kurikuler Semester 1 yang Disetujui', min_value=0, step=1)
curricular_units_2nd_sem_approved = st.number_input('Unit Kurikuler Semester 2 yang Disetujui', min_value=0, step=1)
unemployment_rate = st.number_input('Tingkat Pengangguran', min_value=0.0, step=0.1)
inflation_rate = st.number_input('Tingkat Inflasi', min_value=-10.0, step=0.1) # Adjust min/max based on data
gdp = st.number_input('PDB', min_value=-10.0, step=0.1) # Adjust min/max based on data

# You need to add inputs for ALL features in `feature_names`
# To make it easier, you can dynamically create inputs based on the data types in X.columns
# However, providing a fixed set of inputs based on the top important features and other relevant ones is often sufficient for a demo.
# For simplicity in this example, I'll add a few more based on the feature importance shown in the notebook.
# You should add inputs for all features in `feature_names` for a complete application.

# Placeholder for all feature inputs - you need to add more based on `feature_names`
# For now, create a dictionary with the collected inputs.
input_data_dict = {
    'Marital_status': marital_status,
    'Application_order': application_order,
    'Daytime_evening_attendance': daytime_evening_attendance,
    'Previous_qualification_grade': previous_qualification_grade,
    'Admission_grade': admission_grade,
    'Age_at_enrollment': age_at_enrollment,
    'Tuition_fees_up_to_date': tuition_fees_up_to_date,
    'Gender': gender,
    'Scholarship_holder': scholarship_holder,
    'International': international,
    'Curricular_units_1st_sem_approved': curricular_units_1st_sem_approved,
    'Curricular_units_2nd_sem_approved': curricular_units_2nd_sem_approved,
    'Unemployment_rate': unemployment_rate,
    'Inflation_rate': inflation_rate,
    'GDP': gdp,
    # Add all other features from X.columns here with their corresponding st.____ input widgets
    'Application_mode': st.number_input('Mode Aplikasi', min_value=0, step=1),
    'Course': st.number_input('Kode Kursus', min_value=0, step=1),
    'Previous_qualification': st.number_input('Kualifikasi Sebelumnya', min_value=0, step=1),
    'Nacionality': st.number_input('Kebangsaan', min_value=0, step=1),
    'Mothers_qualification': st.number_input('Kualifikasi Ibu', min_value=0, step=1),
    'Fathers_qualification': st.number_input('Kualifikasi Ayah', min_value=0, step=1),
    'Mothers_occupation': st.number_input('Pekerjaan Ibu', min_value=0, step=1),
    'Fathers_occupation': st.number_input('Pekerjaan Ayah', min_value=0, step=1),
    'Displaced': st.selectbox('Mengungsi (1: Ya, 0: Tidak)', [1, 0]),
    'Educational_special_needs': st.selectbox('Kebutuhan Pendidikan Khusus (1: Ya, 0: Tidak)', [1, 0]),
    'Debtor': st.selectbox('Debitur (1: Ya, 0: Tidak)', [1, 0]),
    'Curricular_units_1st_sem_credited': st.number_input('Unit Kurikuler Semester 1 yang Dikreditkan', min_value=0, step=1),
    'Curricular_units_1st_sem_enrolled': st.number_input('Unit Kurikuler Semester 1 yang Diambil', min_value=0, step=1),
    'Curricular_units_1st_sem_evaluations': st.number_input('Evaluasi Unit Kurikuler Semester 1', min_value=0, step=1),
    'Curricular_units_1st_sem_grade': st.number_input('Nilai Rata-rata Unit Kurikuler Semester 1', min_value=0.0, step=0.1),
    'Curricular_units_1st_sem_without_evaluations': st.number_input('Unit Kurikuler Semester 1 Tanpa Evaluasi', min_value=0, step=1),
    'Curricular_units_2nd_sem_credited': st.number_input('Unit Kurikuler Semester 2 yang Dikreditkan', min_value=0, step=1),
    'Curricular_units_2nd_sem_enrolled': st.number_input('Unit Kurikuler Semester 2 yang Diambil', min_value=0, step=1),
    'Curricular_units_2nd_sem_evaluations': st.number_input('Evaluasi Unit Kurikuler Semester 2', min_value=0, step=1),
    'Curricular_units_2nd_sem_grade': st.number_input('Nilai Rata-rata Unit Kurikuler Semester 2', min_value=0.0, step=0.1),
    'Curricular_units_2nd_sem_without_evaluations': st.number_input('Unit Kurikuler Semester 2 Tanpa Evaluasi', min_value=0, step=1),
}


if st.button('Prediksi Status'):
    # Create a DataFrame from the input data
    input_df = pd.DataFrame([input_data_dict])

    # Ensure the column order matches the training data
    input_df = input_df[feature_names]

    # Scale the input data using the loaded scaler
    input_scaled = scaler.transform(input_df)

    # Make a prediction
    prediction = model_rf.predict(input_scaled)

    # Map the prediction back to the original status labels
    status_labels = {1: 'Graduate', 0: 'Dropout'}
    predicted_status = status_labels[prediction[0]]

    st.subheader('Hasil Prediksi:')
    if predicted_status == 'Graduate':
        st.success(f'Status Mahasiswa Diprediksi: **{predicted_status}**')
    else:
        st.error(f'Status Mahasiswa Diprediksi: **{predicted_status}**')

    # Optional: Display prediction probabilities
    try:
        prediction_proba = model_rf.predict_proba(input_scaled)
        st.write('Probabilitas Prediksi:')
        proba_df = pd.DataFrame(prediction_proba, columns=['Dropout Probability', 'Graduate Probability'])
        st.dataframe(proba_df)
    except AttributeError:
        st.write("Model does not support predict_proba.")

st.write("Pastikan semua kolom input terisi.")
