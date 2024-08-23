import streamlit as st
import joblib
import numpy as np
from PIL import Image

# Memuat model dan scaler
knn_model = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')

# Fungsi untuk prediksi
def predict_hypertension(features):
    features_scaled = scaler.transform([features])
    prediction = knn_model.predict(features_scaled)
    return prediction[0]

# Judul aplikasi dan gambar header
st.title('💓 Prediksi Penyakit Jantung 💓')
st.markdown('### Menggunakan KNN untuk Memprediksi Risiko Penyakit Jantung')

# Menambahkan gambar header
# image = Image.open('heart_disease.jpg')
# st.image(image, use_column_width=True)

# Penjelasan singkat
st.markdown("""
Aplikasi ini menggunakan algoritma **K-Nearest Neighbors (KNN)** untuk memprediksi apakah seseorang berisiko menderita penyakit jantung berdasarkan data input.
Masukkan data kesehatan Anda di bawah ini untuk mendapatkan prediksi.
""")

# Penjelasan singkatan
with st.expander("Penjelasan Singkatan"):
    st.markdown("""
    - **age**: Usia pasien (dalam tahun)
    - **sex**: Jenis kelamin pasien (1 = laki-laki, 0 = perempuan)
    - **cp**: Tipe nyeri dada (0 = tidak ada nyeri dada, 1 = angina tidak stabil, 2 = angina stabil, 3 = nyeri tipe bukan angina)
    - **trestbps**: Tekanan darah sistolik dalam kondisi istirahat (dalam mm Hg)
    - **chol**: Kadar kolesterol dalam serum darah (dalam mg/dL)
    - **fbs**: Kadar gula darah puasa (1 = jika > 120 mg/dL, 0 = jika ≤ 120 mg/dL)
    - **restecg**: Hasil elektrokardiografi saat istirahat (0 = normal, 1 = kelainan ST-T, 2 = hipertrofi ventrikel kiri)
    - **thalach**: Denyut jantung maksimal yang dicapai selama tes latihan (dalam bpm)
    - **exang**: Angina yang diinduksi oleh latihan (1 = ya, 0 = tidak)
    - **oldpeak**: Depresi ST yang diinduksi oleh latihan relatif terhadap istirahat (dalam mm)
    - **slope**: Kemiringan segmen ST puncak saat latihan (0 = menurun, 1 = datar, 2 = meningkat)
    - **ca**: Jumlah pembuluh darah utama yang diwarnai oleh fluoroskopi (0-3)
    - **thal**: Kondisi thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)
    """)

# Input pengguna dalam dua kolom untuk tata letak yang lebih baik
col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Usia', min_value=0, max_value=120, value=25)
    sex = st.selectbox('Jenis Kelamin', options=[0, 1], format_func=lambda x: 'Perempuan' if x == 0 else 'Laki-laki')
    cp = st.selectbox('Tipe Nyeri Dada', options=[0, 1, 2, 3])
    trestbps = st.number_input('Tekanan Darah Istirahat (mm Hg)', min_value=80, max_value=200, value=120)
    chol = st.number_input('Kolesterol Serum (mg/dL)', min_value=100, max_value=600, value=200)
    fbs = st.selectbox('Gula Darah Puasa > 120 mg/dL', options=[0, 1])

with col2:
    restecg = st.selectbox('Hasil Elektrokardiografi', options=[0, 1, 2])
    thalach = st.number_input('Denyut Jantung Maksimum (bpm)', min_value=60, max_value=220, value=150)
    exang = st.selectbox('Angina Induksi Latihan', options=[0, 1])
    oldpeak = st.number_input('Depresi ST (mm)', min_value=0.0, max_value=10.0, value=1.0)
    slope = st.selectbox('Kemiringan Segmen ST Puncak', options=[0, 1, 2])
    ca = st.selectbox('Jumlah Pembuluh Darah Utama', options=[0, 1, 2, 3])
    thal = st.selectbox('Kondisi Thalassemia', options=[1, 2, 3])

# Tombol prediksi
if st.button('Prediksi'):
    features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    result = predict_hypertension(features)
    if result == 1:
        st.markdown('### Hasil: **Pasien berpotensi menderita penyakit jantung.**')
        # st.image('alert.png', width=100)
    else:
        st.markdown('### Hasil: **Pasien tidak menderita penyakit jantung.**')
        # st.image('check.png', width=100)
