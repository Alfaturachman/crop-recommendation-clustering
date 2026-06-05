import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Sistem Rekomendasi Tanaman", page_icon="🌱", layout="centered")

# CSS Kustom untuk mempercantik UI
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #2E7D32; font-weight: bold; text-align: center; margin-bottom: 20px;}
    .sub-header {font-size: 1.2rem; color: #558B2F; text-align: center; margin-bottom: 30px;}
    .recommendation-box {background-color: #E8F5E9; padding: 20px; border-radius: 10px; border-left: 5px solid #2E7D32; margin-top: 20px;}
    .stButton>button {background-color: #2E7D32; color: white; font-weight: bold; border-radius: 8px;}
    .stButton>button:hover {background-color: #1B5E20; color: white;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🌱 Sistem Rekomendasi Tanaman Cerdas</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Masukkan kondisi agronomi tanah Anda, dan AI kami akan merekomendasikan tanaman yang paling sesuai berdasarkan klaster lahan.</div>', unsafe_allow_html=True)

# 1. Memuat Dataset dan Melatih Model
# Menggunakan @st.cache_resource agar pelatihan K-Means hanya berjalan sekali (saat aplikasi pertama kali dibuka)
@st.cache_resource
def load_data_and_train_model():
    # Load data
    df = pd.read_csv('crop_recommendation.csv')
    
    # Pemilihan Fitur Numerik
    features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    X = df[features]
    
    # Standardisasi Data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Feature Engineering: PCA dengan 5 komponen
    from sklearn.decomposition import PCA
    pca_5 = PCA(n_components=5)
    X_pca_5 = pca_5.fit_transform(X_scaled)
    
    # Latih model K-Means (K=22 berdasarkan analisis EDA pada data PCA 5D)
    kmeans = KMeans(n_clusters=22, init='k-means++', random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_pca_5)
    
    return df, scaler, pca_5, kmeans, features

try:
    df, scaler, pca_5, kmeans_model, features = load_data_and_train_model()
except Exception as e:
    st.error(f"Error saat memuat dataset. Pastikan file 'crop_recommendation.csv' ada di folder yang sama. Detail error: {e}")
    st.stop()

# 2. Form Input Parameter Lahan
st.write("### 📝 Masukkan Parameter Tanah dan Cuaca")

col1, col2, col3 = st.columns(3)

with col1:
    N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=300.0, value=90.0, step=1.0)
    temperature = st.number_input("Suhu Ruangan (°C)", min_value=0.0, max_value=60.0, value=25.0, step=0.1)
    rainfall = st.number_input("Curah Hujan (mm)", min_value=0.0, max_value=500.0, value=150.0, step=1.0)

with col2:
    P = st.number_input("Fosfor (P)", min_value=0.0, max_value=300.0, value=42.0, step=1.0)
    humidity = st.number_input("Kelembaban (%)", min_value=0.0, max_value=100.0, value=80.0, step=0.1)

with col3:
    K = st.number_input("Kalium (K)", min_value=0.0, max_value=300.0, value=43.0, step=1.0)
    ph = st.number_input("Tingkat Keasaman (pH)", min_value=0.0, max_value=14.0, value=6.5, step=0.1)

# 3. Tombol Eksekusi Prediksi
if st.button("Rekomendasikan Tanaman 🔍", use_container_width=True):
    with st.spinner("AI sedang menganalisis kondisi tanah Anda..."):
        # Menyusun input pengguna menjadi DataFrame
        input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=features)
        
        # Melakukan scaling input dengan scaler yang sama saat pelatihan
        input_scaled = scaler.transform(input_data)
        
        # Melakukan transformasi PCA dengan 5 komponen
        input_pca = pca_5.transform(input_scaled)
        
        # Melakukan prediksi klaster untuk data lahan yang baru menggunakan model K-Means
        predicted_cluster = kmeans_model.predict(input_pca)[0]
        
        # Mengekstrak daftar tanaman unik dari dataset yang berada pada klaster yang sama
        recommended_crops = df[df['Cluster'] == predicted_cluster]['label'].unique()
        
        # Menampilkan output
        st.markdown(f"""
        <div class="recommendation-box">
            <h4>✅ Analisis Selesai: Karakteristik lahan masuk ke dalam <span style="color:#1B5E20">Klaster {predicted_cluster}</span></h4>
            <p>Berdasarkan kemiripan kondisi unsur hara dan cuaca, varietas tanaman yang paling direkomendasikan untuk Anda tanam adalah:</p>
            <h3 style="color:#2E7D32;">🌾 {', '.join([crop.capitalize() for crop in recommended_crops])}</h3>
        </div>
        """, unsafe_allow_html=True)
