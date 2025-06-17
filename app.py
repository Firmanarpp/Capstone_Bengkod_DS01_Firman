import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import warnings
import os

# Menonaktifkan peringatan yang tidak krusial untuk tampilan demo yang bersih
warnings.filterwarnings("ignore")

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="AI Health Analyzer - Prediksi Obesitas",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS Styling yang diperbaiki
st.markdown(
    """
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Font utama dan padding container */
    .main .block-container { 
        padding-top: 2rem; 
        padding-bottom: 2rem; 
        padding-left: 3rem; 
        padding-right: 3rem; 
        font-family: 'Poppins', sans-serif;
    }
    
    /* Judul Utama dengan animasi */
    .title-text { 
        font-size: 3.5rem; 
        font-weight: 700; 
        background: linear-gradient(45deg, #2E86AB, #A23B72);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center; 
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Sub-judul */
    .subtitle-text { 
        font-size: 1.2rem; 
        color: #5D6D7E; 
        text-align: center; 
        margin-bottom: 3rem;
        font-weight: 300;
    }
    
    /* Kartu Hasil Utama dengan efek hover */
    .result-highlight-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; 
        padding: 30px; 
        border-radius: 20px; 
        text-align: center;
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    .result-highlight-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
    }
    .result-highlight-card .label { 
        font-size: 1.2rem; 
        font-weight: 300; 
        opacity: 0.9;
    }
    .result-highlight-card .result { 
        font-size: 2.5rem; 
        font-weight: 700; 
        margin: 10px 0;
    }
    
    /* Info cards styling */
    .info-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    /* Feature cards */
    .feature-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        border: 1px solid #e1e8ed;
        transition: all 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    
    /* Metric styling */
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 5px;
    }
    
    /* Menargetkan tombol di dalam Streamlit */
    .stButton > button {
        white-space: normal;
        word-wrap: break-word;
        min-height: 80px; 
        width: 100%;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        padding: 15px 20px;
        font-size: 16px;
        font-weight: 600;
        border-radius: 30px;
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        cursor: pointer;
    }

    /* Efek saat kursor diarahkan ke tombol */
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 30px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #5a72e0 0%, #6b4196 100%);
        color: white !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 20px; /* Disesuaikan paddingnya */
        background-color: #FFFFFF; /* Latar belakang putih */
        border-radius: 10px;
        font-weight: 600;
        color: #808080; /* <<< PERBAIKAN: Warna teks abu-abu untuk tab tidak aktif */
        border: 1px solid #E0E0E0; /* Menambahkan border tipis */
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #F5F5F5;
        color: #FF4B4B;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        /* <<< BARU: Gaya untuk tab yang sedang aktif */
        background-color: #FFFFFF;
        color: #FF4B4B; /* Warna teks merah */
        border-bottom: 3px solid #FF4B4B; /* Garis bawah merah */
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2E86AB;
    }
    
    /* Progress bars */
    .health-score {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        height: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


# --- Fungsi-Fungsi Pembantu ---
@st.cache_resource
def load_model_components():
    """
    Fungsi untuk memuat komponen model dari file:
    - Model terlatih
    - Scaler (untuk normalisasi)
    - Label Encoder (untuk decoding label kelas)
    - Feature names (urutan fitur)
    - Metadata (informasi performa model)
    """
    try:
        base_dir = "deployment_files"
        
        # Cek apakah folder ada
        if not os.path.exists(base_dir):
            st.error(f"⚠️ Folder '{base_dir}' tidak ditemukan!")
            st.info("📁 folder 'deployment_files' dengan file-file berikut:")
            st.code("""
deployment_files/
├── final_model.pkl
├── scaler.pkl
├── label_encoder.pkl
├── feature_names.pkl
└── model_metadata.pkl
            """)
            return None, None, None, None, None
            
        # Load semua komponen
        model = joblib.load(os.path.join(base_dir, "final_model.pkl"))
        scaler = joblib.load(os.path.join(base_dir, "scaler.pkl"))
        label_encoder = joblib.load(os.path.join(base_dir, "label_encoder.pkl"))
        feature_names = joblib.load(os.path.join(base_dir, "feature_names.pkl"))
        metadata = joblib.load(os.path.join(base_dir, "model_metadata.pkl"))
            
        return model, scaler, label_encoder, feature_names, metadata
        
    except FileNotFoundError as e:
        st.error(f"⚠️ File tidak ditemukan: {str(e)}")
        st.info("Pastikan semua file model tersedia di folder 'deployment_files'")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        return None, None, None, None, None


def calculate_bmi(weight, height):
    """Menghitung BMI berdasarkan berat dan tinggi badan"""
    return weight / (height**2) if height > 0 else 0


def categorize_age(age):
    """Kategorisasi umur sesuai kelompok"""
    if age < 18:
        return "Remaja"
    elif age < 25:
        return "Dewasa Muda"
    elif age < 35:
        return "Dewasa"
    elif age < 50:
        return "Paruh Baya"
    else:
        return "Senior"


def get_health_score(input_data, bmi):
    """
    Menghitung skor kesehatan berdasarkan BMI dan gaya hidup
    - Nilai dikurangi jika memiliki kebiasaan tidak sehat
    - Rentang skor: 0–100
    """
    score = 100
    
    # BMI impact
    if bmi < 18.5 or bmi > 30:
        score -= 30
    elif bmi > 25:
        score -= 15
    
    # Lifestyle factors
    if input_data['FAF'] < 1:
        score -= 15
    if input_data['FCVC'] < 2:
        score -= 10
    if input_data['CH2O'] < 2:
        score -= 10
    if input_data['FAVC'] == 'yes':
        score -= 10
    if input_data['SMOKE'] == 'yes':
        score -= 15
    if input_data['TUE'] > 1.5:
        score -= 5
        
    return max(0, score)


def preprocess_input(input_data, scaler, feature_names):
    """
    Preprocessing input user:
    - Hitung BMI
    - Encode binary dan kategori
    - One-hot encoding
    - Normalisasi fitur numerik
    """
    processed_data = pd.DataFrame([input_data])
    
    # Hitung BMI
    processed_data["BMI"] = calculate_bmi(
        processed_data["Weight"].iloc[0], 
        processed_data["Height"].iloc[0]
    )
    
    # Kategorisasi umur
    processed_data["Age_Group"] = categorize_age(processed_data["Age"].iloc[0])
    
    # Encoding binary columns
    binary_columns = ["FAVC", "SCC", "SMOKE", "family_history_with_overweight"]
    for col in binary_columns:
        if col in processed_data.columns:
            processed_data[f"{col}_encoded"] = (processed_data[col] == "yes").astype(int)
    
    # Gender encoding
    processed_data["Gender_encoded"] = (processed_data["Gender"] == "Male").astype(int)
    
    # One-hot encoding untuk categorical features
    categorical_to_encode = ["CALC", "CAEC", "MTRANS", "Age_Group"]
    for col in categorical_to_encode:
        if col in processed_data.columns:
            dummies = pd.get_dummies(processed_data[col], prefix=col, drop_first=True)
            processed_data = pd.concat([processed_data, dummies], axis=1)

    # Pastikan semua features ada
    for feature in feature_names:
        if feature not in processed_data.columns:
            processed_data[feature] = 0
            
    # Select hanya features yang diperlukan
    processed_data = processed_data[feature_names]

    # Scaling numerical features
    numerical_features = [
        "Age", "Height", "Weight", "BMI", "FCVC", "NCP", "CH2O", "FAF", "TUE"
    ]
    numerical_features_exist = [
        col for col in numerical_features if col in processed_data.columns
    ]
    
    if numerical_features_exist:
        processed_data[numerical_features_exist] = scaler.transform(
            processed_data[numerical_features_exist]
        )
        
    return processed_data


def get_obesity_info(obesity_class):
    """
    Memberikan deskripsi lengkap suatu kategori obesitas:
    - Deskripsi
    - Rentang BMI
    - Risiko kesehatan
    - Rekomendasi personal
    """
    # Mapping nama kelas sesuai dataset
    class_mapping = {
        "Insufficient_Weight": "Insufficient Weight",
        "Normal_Weight": "Normal Weight", 
        "Overweight_Level_I": "Overweight Level I",
        "Overweight_Level_II": "Overweight Level II",
        "Obesity_Type_I": "Obesity Type I",
        "Obesity_Type_II": "Obesity Type II",
        "Obesity_Type_III": "Obesity Type III"
    }
    
    obesity_info = {
        "Insufficient Weight": {
            "description": "Berat badan kurang dari normal - Kondisi di mana berat badan berada di bawah rentang sehat",
            "bmi_range": "BMI < 18.5",
            "health_risks": [
                "Sistem imun lemah",
                "Kekurangan nutrisi",
                "Osteoporosis",
                "Masalah kesuburan"
            ],
            "recommendations": [
                "🥗 Konsultasi dengan ahli gizi untuk rencana peningkatan berat badan yang sehat",
                "🍎 Tingkatkan asupan kalori dengan makanan bergizi tinggi",
                "💪 Lakukan latihan kekuatan untuk membangun massa otot",
                "📊 Monitor progress berat badan secara berkala"
            ],
            "color": "#3498db",
            "icon": "📉"
        },
        "Normal Weight": {
            "description": "Berat badan ideal - Kondisi optimal untuk kesehatan jangka panjang",
            "bmi_range": "BMI 18.5 - 24.9",
            "health_risks": [
                "Risiko minimal untuk penyakit terkait berat badan"
            ],
            "recommendations": [
                "✅ Pertahankan pola makan seimbang dengan variasi nutrisi",
                "🏃‍♂️ Rutin berolahraga minimal 150 menit per minggu",
                "💧 Konsumsi air putih 8 gelas per hari",
                "😴 Tidur berkualitas 7-8 jam per hari"
            ],
            "color": "#27ae60",
            "icon": "✅"
        },
        "Overweight Level I": {
            "description": "Kelebihan berat badan ringan - Tahap awal peningkatan risiko kesehatan",
            "bmi_range": "BMI 25.0 - 27.4",
            "health_risks": [
                "Risiko diabetes tipe 2 meningkat",
                "Tekanan darah tinggi",
                "Kolesterol tinggi",
                "Sleep apnea ringan"
            ],
            "recommendations": [
                "⚖️ Target penurunan 5-10% berat badan dalam 6 bulan",
                "🥬 Kurangi porsi makan dan pilih makanan rendah kalori",
                "🚴‍♂️ Tingkatkan aktivitas fisik menjadi 250 menit per minggu",
                "📝 Catat asupan makanan harian untuk kontrol kalori"
            ],
            "color": "#f39c12",
            "icon": "⚠️"
        },
        "Overweight Level II": {
            "description": "Kelebihan berat badan sedang - Risiko kesehatan mulai signifikan",
            "bmi_range": "BMI 27.5 - 29.9",
            "health_risks": [
                "Risiko tinggi diabetes tipe 2",
                "Penyakit jantung koroner",
                "Stroke",
                "Osteoarthritis"
            ],
            "recommendations": [
                "🎯 Program penurunan berat badan terstruktur dengan target 10-15%",
                "🏋️‍♂️ Kombinasi kardio dan latihan kekuatan 5x per minggu",
                "🍽️ Diet rendah kalori dengan defisit 500-750 kalori per hari",
                "👨‍⚕️ Konsultasi rutin dengan dokter untuk monitoring kesehatan"
            ],
            "color": "#e67e22",
            "icon": "⚠️"
        },
        "Obesity Type I": {
            "description": "Obesitas kelas I - Kondisi medis serius yang memerlukan intervensi",
            "bmi_range": "BMI 30.0 - 34.9",
            "health_risks": [
                "Diabetes tipe 2",
                "Penyakit jantung",
                "Hipertensi",
                "Sleep apnea",
                "Beberapa jenis kanker"
            ],
            "recommendations": [
                "🏥 Konsultasi medis segera untuk evaluasi kesehatan menyeluruh",
                "💊 Pertimbangkan terapi medis jika diperlukan",
                "🥗 Program diet ketat dengan supervisi profesional",
                "🏃‍♂️ Aktivitas fisik bertahap dengan panduan ahli"
            ],
            "color": "#e74c3c",
            "icon": "🚨"
        },
        "Obesity Type II": {
            "description": "Obesitas kelas II - Risiko kesehatan sangat tinggi",
            "bmi_range": "BMI 35.0 - 39.9",
            "health_risks": [
                "Risiko sangat tinggi untuk semua penyakit metabolik",
                "Gagal jantung",
                "Penyakit hati berlemak",
                "Masalah persendian serius",
                "Depresi"
            ],
            "recommendations": [
                "🚑 Intervensi medis intensif dengan tim multidisiplin",
                "💉 Evaluasi untuk terapi farmakologi obesitas",
                "🏥 Program rehabilitasi komprehensif",
                "🧠 Dukungan psikologis untuk perubahan perilaku"
            ],
            "color": "#c0392b",
            "icon": "🚨"
        },
        "Obesity Type III": {
            "description": "Obesitas kelas III (Morbid) - Kondisi kritis yang mengancam jiwa",
            "bmi_range": "BMI ≥ 40",
            "health_risks": [
                "Harapan hidup berkurang signifikan",
                "Gagal organ multipel",
                "Mobilitas sangat terbatas",
                "Komplikasi bedah tinggi",
                "Kualitas hidup sangat rendah"
            ],
            "recommendations": [
                "🏥 Perawatan medis darurat jika ada komplikasi akut",
                "🔪 Evaluasi untuk bedah bariatrik",
                "👨‍⚕️ Pengawasan ketat tim medis spesialis",
                "🏡 Program rehabilitasi jangka panjang"
            ],
            "color": "#8e44ad",
            "icon": "🆘"
        },
    }
    
    # Handle both formats (with underscore and with space)
    if obesity_class in obesity_info:
        return obesity_info[obesity_class]
    else:
        # Try with mapping
        mapped_class = class_mapping.get(obesity_class, obesity_class)
        return obesity_info.get(mapped_class, {
            "description": "Kategori tidak dikenal",
            "bmi_range": "N/A",
            "health_risks": ["Informasi tidak tersedia"],
            "recommendations": ["Konsultasikan dengan dokter"],
            "color": "#95a5a6",
            "icon": "❓"
        })


def create_bmi_gauge(bmi_value):
    """
    Membuat visualisasi gauge (meteran) BMI dengan Plotly:
    - Menunjukkan posisi BMI dalam rentang sehat atau tidak
    """
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=bmi_value,
            title={"text": "Indeks Massa Tubuh (BMI)", "font": {"size": 20, "family": "Poppins"}},
            delta={'reference': 22.5, 'increasing': {'color': "red"}},
            gauge={
                "axis": {"range": [10, 50], "tickwidth": 1},
                "bar": {"color": "#2C3E50", "thickness": 0.8},
                "steps": [
                    {"range": [10, 18.5], "color": "#5DADE2"},
                    {"range": [18.5, 24.9], "color": "#58D68D"},
                    {"range": [25, 29.9], "color": "#F4D03F"},
                    {"range": [30, 34.9], "color": "#F5B041"},
                    {"range": [35, 50], "color": "#E74C3C"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": bmi_value
                }
            },
        )
    )
    fig.update_layout(
        height=300, 
        margin={"t": 50, "b": 20, "l": 20, "r": 20},
        font={"family": "Poppins"}
    )
    return fig


def create_probability_chart(probabilities, class_names):
    """Membuat chart distribusi probabilitas prediksi"""
    prob_df = pd.DataFrame(
        {"Class": class_names, "Probability": probabilities}
    ).sort_values("Probability", ascending=True)
    
    # Color mapping untuk setiap kelas
    color_map = {
        "Insufficient Weight": "#3498db",
        "Normal Weight": "#27ae60",
        "Overweight Level I": "#f39c12",
        "Overweight Level II": "#e67e22",
        "Obesity Type I": "#e74c3c",
        "Obesity Type II": "#c0392b",
        "Obesity Type III": "#8e44ad"
    }
    
    fig = px.bar(
        prob_df,
        x="Probability",
        y="Class",
        orientation="h",
        title="Distribusi Probabilitas Prediksi",
        color="Class",
        color_discrete_map=color_map
    )
    fig.update_layout(
        height=450,
        showlegend=False,
        title_x=0.5,
        xaxis_title="Probabilitas (%)",
        yaxis_title="",
        font={"family": "Poppins"},
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            tickformat='.0%',
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)'
        )
    )
    fig.update_traces(
        texttemplate='%{x:.1%}',
        textposition='outside'
    )
    return fig


def create_health_radar(input_data):
    """Create radar chart for health metrics"""
    categories = ['Aktivitas Fisik', 'Konsumsi Sayur', 'Konsumsi Air', 
                  'Kontrol Kalori', 'Kebiasaan Sehat']
    
    values = [
        input_data['FAF'] / 3 * 100,  
        input_data['FCVC'] / 3 * 100,
        input_data['CH2O'] / 3 * 100,
        100 if input_data['SCC'] == 'yes' else 0,
        100 if input_data['SMOKE'] == 'no' else 0
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Profil Kesehatan',
        fillcolor='rgba(102, 126, 234, 0.3)',
        line=dict(color='rgba(102, 126, 234, 1)', width=2)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        title="Profil Kesehatan Anda",
        height=400,
        font={"family": "Poppins"}
    )
    return fig


# --- Aplikasi Utama ---
def main():
    # Load model components
    result = load_model_components()
    if result[0] is None:  # Jika ada error loading model
        st.error("❌ Aplikasi tidak dapat dijalankan karena model tidak ditemukan.")
        st.info("📚 Silakan ikuti dokumentasi untuk setup model files.")
        return
        
    model, scaler, label_encoder, feature_names, metadata = result

    # Header dengan animasi
    st.markdown(
        '<h1 class="title-text">🏃‍♂️Sistem Prediksi Tingkat Obesitas</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="subtitle-text">Sistem Cerdas untuk Analisis Kesehatan & Prediksi Tingkat Obesitas dengan Teknologi Machine Learning</p>',
        unsafe_allow_html=True,
    )

    # Quick Stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-container">
            <h3>97.56%</h3>
            <p>Akurasi Model</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-container">
            <h3>2,111</h3>
            <p>Data Training</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-container">
            <h3>7</h3>
            <p>Kategori Obesitas</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-container">
            <h3>17</h3>
            <p>Parameter Analisis</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar dengan informasi detail
    with st.sidebar:
        st.markdown("### 🤖 Model AI Information")
        
        with st.expander("📊 Performa Model", expanded=True):
            st.write(f"**Algorithm**: {metadata.get('nama_model', 'Random Forest')}")
            st.write(f"**Accuracy**: {metadata.get('akurasi'):.2%}")
            st.write(f"**F1-Score**: {metadata.get('f1_score'):.2%}")
            st.write(f"**Precision**: 97.59%")
            st.write(f"**Recall**: 97.56%")

        with st.expander("🎯 Kategori Obesitas (NObeyesdad)"):
            classes_info = {
                "1️⃣ Insufficient Weight": "BMI < 18.5",
                "2️⃣ Normal Weight": "BMI 18.5-24.9",
                "3️⃣ Overweight Level I": "BMI 25.0-27.4",
                "4️⃣ Overweight Level II": "BMI 27.5-29.9",
                "5️⃣ Obesity Type I": "BMI 30.0-34.9",
                "6️⃣ Obesity Type II": "BMI 35.0-39.9",
                "7️⃣ Obesity Type III": "BMI ≥ 40"
                    }
            for class_name, bmi_range in classes_info.items():
                st.write(f"**{class_name}**")
                st.write(f"└─ {bmi_range}")

        with st.expander("📚 Tentang Dataset"):
            st.write("**Sumber Data Asli:**")
            st.write("Data untuk dataset ini dikumpulkan dari individu di tiga negara:")
            st.write("🇲🇽 Meksiko")
            st.write("🇵🇪 Peru")
            st.write("🇨🇴 Kolombia")

            st.write("")
            st.write("**Karakteristik:**")
            st.write("• **Total Sampel:** 2,111")
            st.write("• **Total Kolom:** 17 (16 fitur prediktor dan 1 variabel target 'NObeyesdad')")
            st.write("• **Rentang Usia:** 14 hingga 61 tahun")
            st.write("• **Distribusi Gender:**")
            st.write("  - Laki-laki: 1,068")
            st.write("  - Perempuan: 1,043")
            st.write("  (Distribusi gender dapat dianggap seimbang)")
            st.write("")
            st.write("**Komposisi Data:**")
            st.write(
                "• **77% data merupakan data sintetis** yang dihasilkan menggunakan metode SMOTE (Synthetic Minority Over-sampling Technique) untuk menyeimbangkan kelas."
            )
            st.write(
                "• **23% data merupakan data riil** yang dikumpulkan langsung melalui platform web."
            )
            
        with st.expander("👨‍💻 Developer Info"):
            st.write("**Firman Naufal Aryaputra**")
            st.write("NIM: A11.2022.14181")
            st.write("Teknik Informatika UDINUS")
            st.write("")
            st.write("**Capstone Project**")
            st.write("Bengkel Koding Data Science")
            st.write("Semester Genap 2024/2025")
            st.write("")
            st.write("🔗 [GitHub Repository](https://github.com/Firmanarpp/Capstone_Bengkod_DS01_Firman.git)")

    # Tabs dengan ikon yang lebih menarik
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🎯 **Prediksi & Analisis**", "📈 **Dashboard Kesehatan**", "📚 **Panduan Kesehatan**", "ℹ️ **Tentang Proyek**"]
    )

    # --- Tab 1: Prediksi ---
    with tab1:
        st.markdown("### 🔬 Analisis Kesehatan Personal")
        
        with st.expander("📝 **Form Input Data** - Isi dengan data yang akurat untuk hasil optimal", expanded=True):
            with st.form("prediction_form_main"):
                st.markdown("#### 👤 Informasi Personal")
                col1, col2, col3 = st.columns(3)
                with col1:
                    age = st.number_input("Usia (tahun)", 10, 100, 25, 1,
                                         help="Usia mempengaruhi metabolisme dan risiko obesitas")
                    gender = st.selectbox("Jenis Kelamin", ["Male", "Female"],
                                         help="Gender mempengaruhi distribusi lemak tubuh")
                with col2:
                    height = st.number_input("Tinggi Badan (meter)", 1.0, 2.5, 1.70, 0.01,
                                           help="Tinggi badan untuk kalkulasi BMI")
                    weight = st.number_input("Berat Badan (kg)", 30.0, 200.0, 70.0, 0.5,
                                           help="Berat badan saat ini")
                with col3:
                    family_history = st.selectbox("Riwayat obesitas keluarga?", ["yes", "no"],
                                                help="Faktor genetik mempengaruhi 40-70% risiko obesitas")
                    
                st.markdown("#### 🍽️ Pola Makan")
                col4, col5, col6, col7 = st.columns(4)
                with col4:
                    favc = st.selectbox("Sering konsumsi makanan tinggi kalori?", ["yes", "no"],
                                       help="Makanan tinggi kalori: fast food, gorengan, dll")
                    caec = st.selectbox("Frekuensi ngemil", 
                                       ["no", "Sometimes", "Frequently", "Always"],
                                       help="Kebiasaan makan di antara waktu makan utama")
                with col5:
                    fcvc = st.slider("Konsumsi sayuran (1-3)", 1.0, 3.0, 2.0, 0.1,
                                    help="1=Jarang, 2=Kadang, 3=Sering")
                    ncp = st.slider("Jumlah makan utama/hari", 1.0, 4.0, 3.0, 0.1,
                                   help="Idealnya 3x sehari dengan porsi seimbang")
                with col6:
                    ch2o = st.slider("Konsumsi air (liter/hari)", 1.0, 3.0, 2.0, 0.1,
                                    help="Rekomendasi: 2-3 liter per hari")
                    scc = st.selectbox("Monitor asupan kalori?", ["no", "yes"],
                                      help="Kesadaran kalori membantu kontrol berat badan")
                with col7:
                    calc = st.selectbox("Konsumsi alkohol", 
                                       ["no", "Sometimes", "Frequently", "Always"],
                                       help="Alkohol mengandung kalori tinggi")
                    smoke = st.selectbox("Merokok?", ["no", "yes"],
                                        help="Merokok mempengaruhi metabolisme")
                    
                st.markdown("#### 🏃‍♂️ Aktivitas & Gaya Hidup")
                col8, col9 = st.columns(2)
                with col8:
                    faf = st.slider("Frekuensi olahraga/minggu", 0.0, 3.0, 1.0, 0.1,
                                   help="0=Tidak pernah, 3=Sangat sering (>5x/minggu)")
                    tue = st.slider("Screen time (jam/hari)", 0.0, 2.0, 1.0, 0.1,
                                   help="Waktu di depan layar (HP, TV, komputer)")
                with col9:
                    mtrans = st.selectbox("Transportasi utama",
                                         ["Public_Transportation", "Automobile", 
                                          "Walking", "Motorbike", "Bike"],
                                         help="Mode transportasi mempengaruhi aktivitas fisik harian")

                # Submit button dengan text yang jelas
                predict_button = st.form_submit_button(
                    label="🚀 Analisis Kesehatan Saya",
                    type="primary",
                    use_container_width=True,
                )

        # Inisialisasi session state
        if "last_prediction" not in st.session_state:
            st.session_state.last_prediction = None

        if predict_button:
            input_data = {
                "Age": age, 
                "Gender": gender, 
                "Height": height, 
                "Weight": weight,
                "CALC": calc, 
                "FAVC": favc, 
                "FCVC": fcvc, 
                "NCP": ncp,
                "SCC": scc, 
                "SMOKE": smoke, 
                "CH2O": ch2o,
                "family_history_with_overweight": family_history,
                "FAF": faf, 
                "TUE": tue, 
                "CAEC": caec, 
                "MTRANS": mtrans,
            }
            
            with st.spinner("🔄 Menganalisis data kesehatan Anda..."):
                try:
                    # Preprocessing
                    processed_input = preprocess_input(input_data, scaler, feature_names)
                    
                    # Prediksi
                    prediction = model.predict(processed_input)[0]
                    probabilities = model.predict_proba(processed_input)[0]
                    predicted_class = label_encoder.inverse_transform([prediction])[0]
                    
                    # Hitung BMI dan health score
                    bmi = calculate_bmi(weight, height)
                    health_score = get_health_score(input_data, bmi)

                    # Simpan ke session state
                    st.session_state.last_prediction = {
                        "input_data": input_data,
                        "prediction": prediction,
                        "probabilities": probabilities,
                        "predicted_class": predicted_class,
                        "bmi": bmi,
                        "health_score": health_score
                    }
                    
                except Exception as e:
                    st.error(f"❌ Error dalam prediksi: {str(e)}")
                    st.info("💡 Pastikan semua input data sudah benar")

        # Tampilkan hasil
        if st.session_state.last_prediction:
            res = st.session_state.last_prediction
            obesity_info = get_obesity_info(res["predicted_class"])
            
            st.divider()
            st.markdown(f"## {obesity_info.get('icon', '📊')} Hasil Analisis Kesehatan")

            # Main results
            col_res1, col_res2 = st.columns([1, 1])
            with col_res1:
                st.markdown(
                    f'<div class="result-highlight-card">'
                    f'<p class="label">Status Kesehatan Anda</p>'
                    f'<p class="result">{res["predicted_class"].replace("_", " ")}</p>'
                    f'<p>Confidence Level: <strong>{res["probabilities"].max():.1%}</strong></p>'
                    f'<p>Health Score: <strong>{res["health_score"]}/100</strong></p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                
                # Health score progress bar
                st.markdown("### 💪 Overall Health Score")
                st.progress(res["health_score"] / 100)
                
                if res["health_score"] >= 80:
                    st.success("Excellent! Kesehatan Anda sangat baik!")
                elif res["health_score"] >= 60:
                    st.warning("Good! Ada ruang untuk perbaikan.")
                else:
                    st.error("Perlu perhatian serius pada kesehatan Anda.")
                    
            with col_res2:
                st.plotly_chart(create_bmi_gauge(res["bmi"]), use_container_width=True)

            # Detailed recommendations
            st.markdown("### 📋 Analisis Detail & Rekomendasi")
            
            col_detail1, col_detail2 = st.columns(2)
            with col_detail1:
                with st.container():
                    st.markdown(f"#### {obesity_info.get('icon', '')} Informasi Kondisi")
                    st.info(obesity_info.get('description', 'N/A'))
                    st.write(f"**Rentang BMI**: {obesity_info.get('bmi_range', 'N/A')}")
                    
                    st.markdown("#### ⚠️ Risiko Kesehatan")
                    for risk in obesity_info.get('health_risks', []):
                        st.write(f"• {risk}")
                        
            with col_detail2:
                with st.container():
                    st.markdown("#### 💡 Rekomendasi Spesifik")
                    for i, rec in enumerate(obesity_info.get('recommendations', [])):
                        st.write(f"{rec}")

            # Probability distribution
            with st.container():
                class_names = [c.replace("_", " ") for c in label_encoder.classes_]
                st.plotly_chart(
                    create_probability_chart(res["probabilities"], class_names),
                    use_container_width=True,
                )

    # --- Tab 2: Dashboard Kesehatan ---
    with tab2:
        st.header("📊 Dashboard Kesehatan Personal", divider="rainbow")
        
        if st.session_state.last_prediction:
            input_data = st.session_state.last_prediction["input_data"]
            bmi_val = st.session_state.last_prediction["bmi"]
            health_score = st.session_state.last_prediction["health_score"]

            # Summary cards
            col1, col2, col3 = st.columns(3)
            with col1:
                with st.container():
                    st.markdown("### 👤 Profil Kesehatan")
                    st.metric("Usia", f"{input_data['Age']} tahun ({categorize_age(input_data['Age'])})")
                    st.metric("BMI", f"{bmi_val:.1f}")
                    st.metric("Health Score", f"{health_score}/100")
                    
            with col2:
                with st.container():
                    st.markdown("### 🎯 Target Kesehatan")
                    ideal_weight = 22.5 * (input_data['Height'] ** 2)
                    weight_diff = input_data['Weight'] - ideal_weight
                    st.metric("Berat Ideal", f"{ideal_weight:.1f} kg")
                    st.metric("Selisih Berat", f"{weight_diff:+.1f} kg")
                    if weight_diff > 0:
                        st.write("📉 Perlu menurunkan berat badan")
                    elif weight_diff < -5:
                        st.write("📈 Perlu menaikkan berat badan")
                    else:
                        st.write("✅ Berat badan ideal!")
                        
            with col3:
                with st.container():
                    st.markdown("### 📈 Statistik Gaya Hidup")
                    active_days = input_data['FAF'] * 7 / 3
                    st.metric("Hari Aktif/Minggu", f"{active_days:.0f} hari")
                    st.metric("Konsumsi Sayur", f"{input_data['FCVC']:.1f}/3.0")
                    st.metric("Screen Time", f"{input_data['TUE']:.1f} jam/hari")

            # Health radar chart
            st.plotly_chart(create_health_radar(input_data), use_container_width=True)

            # Lifestyle analysis
            st.markdown("### 🏃‍♂️ Analisis Gaya Hidup Detail")
            
            lifestyle_scores = {
                "Aktivitas Fisik": (input_data['FAF'] / 3 * 100, "🏃‍♂️"),
                "Nutrisi Sayuran": (input_data['FCVC'] / 3 * 100, "🥬"),
                "Hidrasi": (input_data['CH2O'] / 3 * 100, "💧"),
                "Kontrol Kalori": (100 if input_data['SCC'] == 'yes' else 0, "📊"),
                "Bebas Rokok": (100 if input_data['SMOKE'] == 'no' else 0, "🚭")
            }
            
            for aspect, (score, icon) in lifestyle_scores.items():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"{icon} **{aspect}**")
                    st.progress(score / 100)
                with col2:
                    if score >= 80:
                        st.success(f"{score:.0f}%")
                    elif score >= 50:
                        st.warning(f"{score:.0f}%")
                    else:
                        st.error(f"{score:.0f}%")
        else:
            st.info("💡 Lakukan prediksi terlebih dahulu untuk melihat dashboard kesehatan Anda")

    # --- Tab 3: Panduan Kesehatan ---
    with tab3:
        st.header("📚 Panduan Kesehatan Komprehensif", divider="rainbow")
        
        health_guide = st.selectbox(
            "Pilih topik panduan:",
            ["Nutrisi Seimbang", "Program Olahraga", "Manajemen Berat Badan", "Kesehatan Mental"]
        )
        
        if health_guide == "Nutrisi Seimbang":
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                ### 🥗 Prinsip Nutrisi Seimbang
                
                **1. Piramida Makanan Sehat**
                - 🌾 Karbohidrat kompleks (40%)
                - 🥬 Sayur & buah (35%)
                - 🥩 Protein (15%)
                - 🥛 Susu & produk olahan (10%)
                
                **2. Porsi Ideal Sekali Makan**
                - ½ piring: Sayuran & buah
                - ¼ piring: Protein
                - ¼ piring: Karbohidrat
                - Minum: Air putih
                """)
                
            with col2:
                st.markdown("""
                ### 🍎 Tips Makan Sehat
                
                **✅ Yang Dianjurkan:**
                - Makan 3x sehari dengan teratur
                - Konsumsi 5 porsi sayur & buah/hari
                - Pilih karbohidrat kompleks
                - Minum 8 gelas air/hari
                
                **❌ Yang Dihindari:**
                - Skip makan
                - Makanan tinggi gula & garam
                - Gorengan berlebihan
                - Minuman bersoda
                """)
                
        elif health_guide == "Program Olahraga":
            st.markdown("""
            ### 💪 Program Olahraga Terstruktur
            
            #### Pemula (Minggu 1-4)
            - **Senin & Kamis**: Jalan kaki 20 menit
            - **Selasa & Jumat**: Stretching 15 menit
            - **Rabu**: Istirahat aktif (aktivitas ringan)
            - **Weekend**: Aktivitas menyenangkan (berenang, bersepeda santai)
            
            #### Menengah (Minggu 5-8)
            - **Senin & Kamis**: Jogging 30 menit
            - **Selasa & Jumat**: Strength training ringan
            - **Rabu**: Yoga atau pilates
            - **Weekend**: Olahraga tim atau hiking
            
            #### Lanjutan (Minggu 9+)
            - **Senin & Kamis**: HIIT 40 menit
            - **Selasa & Jumat**: Weight training
            - **Rabu**: Cardio steady state
            - **Weekend**: Kombinasi aktivitas
            """)
            
        elif health_guide == "Manajemen Berat Badan":
            tab_a, tab_b = st.tabs(["Menurunkan Berat", "Menaikkan Berat"])
            
            with tab_a:
                st.markdown("""
                ### 📉 Strategi Penurunan Berat Badan Sehat
                
                **Target Realistis**: 0.5-1 kg per minggu
                
                **1. Defisit Kalori**
                - Kurangi 500-750 kalori/hari dari kebutuhan
                - Tracking makanan dengan aplikasi
                - Fokus pada makanan mengenyangkan rendah kalori
                
                **2. Olahraga Efektif**
                - Kombinasi cardio & strength training
                - Minimal 150 menit/minggu intensitas sedang
                - HIIT untuk pembakaran maksimal
                
                **3. Perubahan Gaya Hidup**
                - Tidur 7-8 jam/hari
                - Kelola stress
                - Makan mindful (tidak sambil nonton TV)
                """)
                
            with tab_b:
                st.markdown("""
                ### 📈 Strategi Penambahan Berat Badan Sehat
                
                **Target Realistis**: 0.25-0.5 kg per minggu
                
                **1. Surplus Kalori**
                - Tambah 300-500 kalori/hari
                - Fokus pada kalori berkualitas
                - Makan lebih sering (5-6x/hari)
                
                **2. Latihan Kekuatan**
                - Weight training 3-4x/minggu
                - Progressive overload
                - Istirahat cukup antar sesi
                
                **3. Nutrisi Optimal**
                - Protein 1.5-2g/kg berat badan
                - Karbohidrat kompleks
                - Lemak sehat (alpukat, kacang)
                """)
                
        else:  # Kesehatan Mental
            st.markdown("""
            ### 🧠 Kesehatan Mental & Manajemen Stress
            
            #### Teknik Relaksasi
            1. **Breathing Exercise (4-7-8)**
               - Tarik napas 4 detik
               - Tahan 7 detik
               - Hembuskan 8 detik
               - Ulangi 3-4x
            
            2. **Progressive Muscle Relaxation**
               - Tegangkan otot 5 detik
               - Lepaskan dan rasakan relaksasi
               - Mulai dari kaki hingga kepala
            
            #### Mindfulness & Meditasi
            - Meditasi 10 menit/hari
            - Journaling untuk self-reflection
            - Gratitude practice sebelum tidur
            
            #### Support System
            - Berbagi dengan orang terdekat
            - Konsultasi profesional jika perlu
            - Join komunitas dengan minat sama
            """)

    # --- Tab 4: Tentang Proyek ---
    with tab4:
        st.header("ℹ️ Informasi Proyek", divider="rainbow")
        
        col1, col2 = st.columns(2)
        with col1:
            with st.container():
                st.markdown("""
                ### 🎓 Latar Belakang Proyek
                
                **Capstone Project Data Science**  
                Bengkel Koding - Universitas Dian Nuswantoro
                
                Proyek ini dikembangkan sebagai solusi berbasis AI untuk membantu 
                masyarakat dalam memahami dan mengelola risiko obesitas melalui 
                analisis komprehensif dari berbagai faktor gaya hidup.
                
                **Motivasi:**
                - Meningkatnya prevalensi obesitas global
                - Kebutuhan akan tools prediksi yang akurat
                - Edukasi kesehatan yang personalized
                
                **Dataset:**
                - Sumber: 3 negara Amerika Latin
                - Jumlah data: 2,111 sampel
                - Features: 17 atribut
                - Target: NObeyesdad (7 kategori)
                """)
                
                st.markdown("""
                ### 🔬 Metodologi Penelitian
                
                1. **Data Collection & Preparation**
                   - Dataset dari Mexico, Peru, Colombia
                   - 77% data sintetis (SMOTE)
                   - 23% data riil 
                
                2. **Exploratory Data Analysis**
                   - Distribusi data & outlier detection
                   - Feature correlation analysis
                   - Pattern identification
                
                3. **Feature Engineering**
                   - BMI calculation
                   - Age group categorization
                   - Binary & categorical encoding
                """)
                
        with col2:
            with st.container():
                st.markdown("""
                ### 🤖 Model Development
                
                **Algoritma yang Diuji:**
                - Random Forest ✅ (Terpilih)
                - Gradient Boosting
                - Support Vector Machine
                
                **Optimasi:**
                - SMOTE untuk handle imbalance
                - GridSearchCV untuk hyperparameter tuning
                - Cross-validation 5-fold
                
                **Hasil Akhir:**
                - Accuracy: 97.56%
                - F1-Score: 97.55%
                """)
                
                st.markdown("""
                ### 👨‍💻 Tim Pengembang
                
                **Developer:**  
                Firman Naufal Aryaputra  
                NIM: A11.2022.14181  
                Teknik Informatika UDINUS
                
                **Pembimbing:**  
                Tim Dosen Bengkel Koding Data Science
                
                **Tech Stack:**
                - Python
                - Streamlit for web app
                - Plotly for visualizations
                - Joblib for model deployment
                
                
                🔗 **Repository:** [GitHub](https://github.com/Firmanarpp/Capstone_Bengkod_DS01_Firman.git)
                """)
        
        # Project requirements info
        st.markdown("### 📋 Requirements Project")
        
        req_cols = st.columns(3)
        with req_cols[0]:
            st.info("""
            **EDA & Preprocessing**
            - Missing values handling
            - Outlier detection
            - Feature scaling
            - Class imbalance (SMOTE)
            """)
        with req_cols[1]:
            st.warning("""
            **Modeling**
            - Min. 3 algoritma
            - Model comparison
            - Hyperparameter tuning
            - Cross validation
            """)
        with req_cols[2]:
            st.success("""
            **Deployment**
            - Streamlit app
            - GitHub repository
            - Online deployment
            - Demo presentation
            """)
        
        # Disclaimer
        st.warning("""
        ### ⚠️ Disclaimer Penting
        
        Aplikasi ini dikembangkan untuk tujuan edukasi dan penelitian dalam rangka 
        Capstone Project Bengkel Koding Data Science UDINUS. Hasil prediksi tidak dapat 
        menggantikan konsultasi dengan profesional kesehatan. Selalu konsultasikan 
        kondisi kesehatan Anda dengan dokter atau ahli gizi tersertifikasi untuk 
        mendapatkan diagnosa dan treatment yang tepat.
        """,)


if __name__ == "__main__":
    main()