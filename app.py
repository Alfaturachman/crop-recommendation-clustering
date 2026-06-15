import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from sklearn.mixture import GaussianMixture
import skfuzzy as fuzz
from sklearn.neighbors import KNeighborsClassifier

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Sistem Rekomendasi Tanaman Cerdas", page_icon="🌱", layout="wide")

# CSS Kustom untuk UI Premium (Modern Sleek Green Theme)
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem; 
        color: #1B5E20; 
        font-weight: 800; 
        text-align: center; 
        margin-bottom: 5px;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    .sub-header {
        font-size: 1.15rem; 
        color: #4E6E58; 
        text-align: center; 
        margin-bottom: 35px;
        font-family: 'Inter', sans-serif;
    }
    .recommendation-box {
        background: rgba(232, 245, 233, 0.85); 
        padding: 25px; 
        border-radius: 12px; 
        border-left: 6px solid #2E7D32; 
        margin-top: 25px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    .metric-card {
        background: #f8faf9;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e1e8e4;
        text-align: center;
    }
    .stButton>button {
        background-color: #2E7D32; 
        color: white; 
        font-weight: bold; 
        border-radius: 8px;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1B5E20; 
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🌱 Sistem Rekomendasi Tanaman Cerdas</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Masukkan kondisi agronomi tanah Anda, dan AI kami akan merekomendasikan tanaman yang paling sesuai berdasarkan klaster lahan.</div>', unsafe_allow_html=True)

# 1. Memuat Dataset dan Melatih Model
# Menggunakan @st.cache_resource agar pelatihan model hanya berjalan sekali
@st.cache_resource
def load_data_and_train_models():
    # Load data
    df = pd.read_csv('crop_recommendation.csv')
    
    # Pemilihan Fitur Numerik
    features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    X = df[features]
    
    # Standardisasi Data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Feature Engineering: PCA dengan 5 komponen
    pca_5 = PCA(n_components=5)
    X_pca_5 = pca_5.fit_transform(X_scaled)
    
    # Latih kelima model klasterisasi (K=22)
    # 1. K-Means
    kmeans = KMeans(n_clusters=22, init='k-means++', random_state=42, n_init=10)
    df['K-Means'] = kmeans.fit_predict(X_pca_5)
    
    # 2. K-Medoids (PAM)
    kmedoids = KMedoids(n_clusters=22, random_state=42)
    df['K-Medoids'] = kmedoids.fit_predict(X_pca_5)
    
    # 3. Gaussian Mixture Model (GMM)
    gmm = GaussianMixture(n_components=22, random_state=42)
    df['GMM'] = gmm.fit_predict(X_pca_5)
    
    # 4. Hierarchical (Agglomerative)
    agg = AgglomerativeClustering(n_clusters=22)
    df['Hierarchical'] = agg.fit_predict(X_pca_5)
    
    # 5. Fuzzy C-Means (FCM)
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        X_pca_5.T, c=22, m=2.0, error=0.005, maxiter=1000, seed=42
    )
    df['Fuzzy C-Means'] = np.argmax(u, axis=0)
    
    # Bungkus Hierarchical dan FCM dengan 1-NN Classifier agar mendukung .predict() untuk data baru
    agg_clf = KNeighborsClassifier(n_neighbors=1)
    agg_clf.fit(X_pca_5, df['Hierarchical'])
    
    fcm_clf = KNeighborsClassifier(n_neighbors=1)
    fcm_clf.fit(X_pca_5, df['Fuzzy C-Means'])
    
    models = {
        'K-Means': kmeans,
        'K-Medoids': kmedoids,
        'GMM': gmm,
        'Hierarchical': agg_clf,
        'Fuzzy C-Means': fcm_clf
    }
    
    return df, scaler, pca_5, models, features

try:
    df, scaler, pca_5, models, features = load_data_and_train_models()
except Exception as e:
    st.error(f"Error saat memuat dataset. Pastikan file 'crop_recommendation.csv' ada di folder yang sama. Detail error: {e}")
    st.stop()

# 2. Sidebar - Pengaturan Model & Info Performa
st.sidebar.header("⚙️ Pengaturan Model")
selected_model_name = st.sidebar.selectbox(
    "Pilih Model Klasterisasi",
    options=['GMM', 'Hierarchical', 'K-Means', 'Fuzzy C-Means', 'K-Medoids'],
    index=0,
    help="GMM (Gaussian Mixture Model) direkomendasikan karena memiliki Adjusted Rand Index (ARI) tertinggi sebesar ~0.82."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Performa Model (PCA 5D)")
# Data hasil evaluasi dari notebook
perf_data = pd.DataFrame({
    'Model': ['GMM', 'Hierarchical', 'K-Means', 'Fuzzy C-Means', 'K-Medoids'],
    'ARI': [0.816, 0.609, 0.519, 0.511, 0.448],
    'DBI': [1.716, 1.098, 1.081, 1.379, 1.221],
    'Silhouette': [0.230, 0.288, 0.307, 0.243, 0.285]
})
st.sidebar.dataframe(perf_data.style.background_gradient(cmap='Greens', subset=['ARI']))
st.sidebar.caption("Keterangan: ARI (Adjusted Rand Index) mengukur kecocokan dengan label tanaman asli (makin tinggi makin baik).")

# 3. Form Input Parameter Lahan
col_form, col_space = st.columns([3, 1])

with col_form:
    st.write("### 📝 Masukkan Parameter Tanah dan Cuaca")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=300.0, value=90.0, step=1.0, help="Kadar Nitrogen dalam tanah")
        temperature = st.number_input("Suhu Ruangan (°C)", min_value=0.0, max_value=60.0, value=25.0, step=0.1, help="Suhu lingkungan tanaman")
        rainfall = st.number_input("Curah Hujan (mm)", min_value=0.0, max_value=500.0, value=150.0, step=1.0, help="Intensitas curah hujan tahunan")
    
    with c2:
        P = st.number_input("Fosfor (P)", min_value=0.0, max_value=300.0, value=42.0, step=1.0, help="Kadar Fosfor dalam tanah")
        humidity = st.number_input("Kelembaban (%)", min_value=0.0, max_value=100.0, value=80.0, step=0.1, help="Kelembaban udara sekitar")
    
    with c3:
        K = st.number_input("Kalium (K)", min_value=0.0, max_value=300.0, value=43.0, step=1.0, help="Kadar Kalium dalam tanah")
        ph = st.number_input("Tingkat Keasaman (pH)", min_value=0.0, max_value=14.0, value=6.5, step=0.1, help="Nilai pH keasaman tanah (0-14)")

    st.markdown("---")
    
    # 4. Tombol Eksekusi Prediksi
    if st.button("Rekomendasikan Tanaman 🔍", use_container_width=True):
        with st.spinner("AI sedang menganalisis kondisi tanah Anda..."):
            # Menyusun input pengguna menjadi DataFrame
            input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=features)
            
            # Melakukan scaling input dengan scaler yang sama saat pelatihan
            input_scaled = scaler.transform(input_data)
            
            # Melakukan transformasi PCA dengan 5 komponen
            input_pca = pca_5.transform(input_scaled)
            
            # Melakukan prediksi klaster untuk data lahan yang baru menggunakan model terpilih
            model_obj = models[selected_model_name]
            predicted_cluster = model_obj.predict(input_pca)[0]
            
            # Mengekstrak daftar tanaman unik dari dataset yang berada pada klaster yang sama
            recommended_crops = df[df[selected_model_name] == predicted_cluster]['label'].unique()
            
            # Menampilkan output
            st.markdown(f"""
            <div class="recommendation-box">
                <h4>✅ Analisis Selesai: Karakteristik lahan masuk ke dalam <span style="color:#1B5E20">Klaster {predicted_cluster}</span> (Model: {selected_model_name})</h4>
                <p>Berdasarkan kemiripan kondisi unsur hara dan cuaca, varietas tanaman yang paling direkomendasikan untuk Anda tanam adalah:</p>
                <h3 style="color:#2E7D32;">🌾 {', '.join([crop.capitalize() for crop in recommended_crops])}</h3>
            </div>
            """, unsafe_allow_html=True)
