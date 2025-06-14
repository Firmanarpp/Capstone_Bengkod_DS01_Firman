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
    page_title="Dasbor Prediksi Obesitas",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS Kustom untuk Desain Profesional ---
st.markdown(
    """
<style>
    /* Font utama dan padding container */
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; padding-left: 3rem; padding-right: 3rem; }
    /* Judul Utama */
    .title-text { font-size: 3.2rem; font-weight: 700; color: #1F618D; text-align: center; margin-bottom: 0.5rem; }
    /* Sub-judul */
    .subtitle-text { font-size: 1.1rem; color: #5D6D7E; text-align: center; margin-bottom: 2.5rem; }
    /* Kartu Hasil Utama */
    .result-highlight-card {
        background: linear-gradient(135deg, #3498DB 0%, #2980B9 100%);
        color: white; padding: 25px; border-radius: 10px; text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .result-highlight-card .label { font-size: 1.1rem; font-weight: 300; }
    .result-highlight-card .result { font-size: 2.2rem; font-weight: 700; }
</style>
""",
    unsafe_allow_html=True,
)


# --- Fungsi-Fungsi Pembantu ---
@st.cache_resource
def load_model_components():
    try:
        # Menggunakan nama folder dari kode Anda
        base_dir = "deployment_files"
        model = joblib.load(os.path.join(base_dir, "final_model.pkl"))
        scaler = joblib.load(os.path.join(base_dir, "scaler.pkl"))
        label_encoder = joblib.load(os.path.join(base_dir, "label_encoder.pkl"))
        feature_names = joblib.load(os.path.join(base_dir, "feature_names.pkl"))
        metadata = joblib.load(os.path.join(base_dir, "model_metadata.pkl"))
        return model, scaler, label_encoder, feature_names, metadata
    except FileNotFoundError:
        st.error(
            "⚠️ File Model Tidak Ditemukan! Pastikan folder 'deployment_files' ada di direktori yang sama."
        )
        st.stop()


def calculate_bmi(weight, height):
    return weight / (height**2) if height > 0 else 0


def categorize_age(age):
    if age < 18:
        return "Teen"
    elif age < 25:
        return "Young_Adult"
    elif age < 35:
        return "Adult"
    elif age < 50:
        return "Middle_Age"
    else:
        return "Senior"


def preprocess_input(input_data, scaler, feature_names):
    # Menggunakan fungsi pra-pemrosesan dari kode Anda, ini sudah benar
    processed_data = pd.DataFrame([input_data])
    processed_data["BMI"] = calculate_bmi(
        processed_data["Weight"].iloc[0], processed_data["Height"].iloc[0]
    )
    processed_data["Age_Group"] = categorize_age(processed_data["Age"].iloc[0])
    binary_columns = ["FAVC", "SCC", "SMOKE", "family_history_with_overweight"]
    for col in binary_columns:
        if col in processed_data.columns:
            processed_data[f"{col}_encoded"] = (processed_data[col] == "yes").astype(
                int
            )
    processed_data["Gender_encoded"] = (processed_data["Gender"] == "Male").astype(int)
    categorical_to_encode = ["CALC", "CAEC", "MTRANS", "Age_Group"]
    for col in categorical_to_encode:
        if col in processed_data.columns:
            dummies = pd.get_dummies(processed_data[col], prefix=col, drop_first=True)
            processed_data = pd.concat([processed_data, dummies], axis=1)

    for feature in feature_names:
        if feature not in processed_data.columns:
            processed_data[feature] = 0
    processed_data = processed_data[feature_names]

    numerical_features = [
        "Age",
        "Height",
        "Weight",
        "BMI",
        "FCVC",
        "NCP",
        "CH2O",
        "FAF",
        "TUE",
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
    # Kamus informasi ini tetap sama
    obesity_info = {
        "Insufficient_Weight": {
            "description": "Berat badan kurang dari normal",
            "bmi_range": "BMI < 18.5",
            "recommendations": [
                "Konsultasi dengan ahli gizi untuk meningkatkan berat badan",
                "Perbanyak asupan kalori sehat",
                "Olahraga untuk membangun massa otot",
            ],
            "color": "#3498db",
        },
        "Normal_Weight": {
            "description": "Berat badan dalam rentang normal",
            "bmi_range": "BMI 18.5 - 24.9",
            "recommendations": [
                "Pertahankan pola makan sehat",
                "Rutin berolahraga 150 menit per minggu",
                "Jaga keseimbangan kalori",
            ],
            "color": "#27ae60",
        },
        "Overweight_Level_I": {
            "description": "Kelebihan berat badan tingkat I",
            "bmi_range": "BMI 25.0 - 29.9",
            "recommendations": [
                "Kurangi asupan kalori harian",
                "Tingkatkan aktivitas fisik",
                "Batasi makanan tinggi gula dan lemak",
            ],
            "color": "#f39c12",
        },
        "Overweight_Level_II": {
            "description": "Kelebihan berat badan tingkat II",
            "bmi_range": "BMI 30.0 - 34.9",
            "recommendations": [
                "Program penurunan berat badan terstruktur",
                "Olahraga intensitas sedang hingga tinggi",
                "Diet rendah kalori dengan supervisi",
            ],
            "color": "#e67e22",
        },
        "Obesity_Type_I": {
            "description": "Obesitas tipe I",
            "bmi_range": "BMI 30.0 - 34.9",
            "recommendations": [
                "Konsultasi medis segera",
                "Program penurunan berat badan medis",
                "Evaluasi risiko penyakit komorbid",
            ],
            "color": "#e74c3c",
        },
        "Obesity_Type_II": {
            "description": "Obesitas tipe II",
            "bmi_range": "BMI 35.0 - 39.9",
            "recommendations": [
                "Intervensi medis intensif",
                "Pertimbangkan terapi farmakologi",
                "Program rehabilitasi komprehensif",
            ],
            "color": "#c0392b",
        },
        "Obesity_Type_III": {
            "description": "Obesitas tipe III (morbid)",
            "bmi_range": "BMI ≥ 40",
            "recommendations": [
                "Intervensi medis darurat jika diperlukan",
                "Pertimbangkan bedah bariatrik",
                "Tim medis multidisiplin",
            ],
            "color": "#8e44ad",
        },
    }
    return obesity_info.get(obesity_class, {})


def create_bmi_gauge(bmi_value):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=bmi_value,
            title={"text": "Indeks Massa Tubuh (BMI)", "font": {"size": 18}},
            gauge={
                "axis": {"range": [10, 50]},
                "bar": {"color": "#2C3E50"},
                "steps": [
                    {"range": [10, 18.5], "color": "#5DADE2"},
                    {"range": [18.5, 24.9], "color": "#58D68D"},
                    {"range": [25, 29.9], "color": "#F4D03F"},
                    {"range": [30, 34.9], "color": "#F5B041"},
                    {"range": [35, 50], "color": "#E74C3C"},
                ],
            },
        )
    )
    fig.update_layout(height=250, margin={"t": 50, "b": 20, "l": 20, "r": 20})
    return fig


def create_probability_chart(probabilities, class_names):
    # Fungsi chart probabilitas tetap sama
    prob_df = pd.DataFrame(
        {"Class": class_names, "Probability": probabilities}
    ).sort_values("Probability", ascending=True)
    fig = px.bar(
        prob_df,
        x="Probability",
        y="Class",
        orientation="h",
        title="Probabilitas Prediksi per Kelas",
        color="Probability",
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        height=400,
        showlegend=False,
        title_x=0.5,
        xaxis_title="Probabilitas",
        yaxis_title="",
    )
    return fig


# --- Aplikasi Utama ---
def main():
    model, scaler, label_encoder, feature_names, metadata = load_model_components()

    st.markdown(
        '<h1 class="title-text">⚕️ Prediksi Tingkat Obesitas</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="subtitle-text">Aplikasi AI untuk memprediksi tingkat obesitas berdasarkan kebiasaan makan dan kondisi fisik.</p>',
        unsafe_allow_html=True,
    )

    
    with st.sidebar:
        st.markdown("### 📊 Informasi Model")
        st.write(f"**Model**: {metadata.get('nama_model', 'N/A')}")
        st.write(f"**Akurasi**: {metadata.get('akurasi', 0):.2%}")
        st.write(f"**F1-Score**: {metadata.get('f1_score', 0):.2%}")
        st.write(f"**Jumlah Fitur**: {metadata.get('jumlah_fitur', 0)}")

        st.divider()
        st.markdown("### 🎯 Kelas Target")
        for i, class_name in enumerate(metadata.get("kelas_target", [])):
            st.write(f"{i+1}. {class_name.replace('_', ' ')}")

        st.divider()
        st.markdown("### ℹ️ Tentang Dataset")
        st.write("Dataset dari 3 negara: 🇲🇽 Meksiko, 🇵🇪 Peru, 🇨🇴 Kolombia.")
        st.write(f"Total: 2,111 sampel, 17 atribut.")

    # --- Tiga Tab Utama ---
    tab1, tab2, tab3 = st.tabs(
        ["🔮 **Prediksi**", "📊 **Analisis Input**", "ℹ️ **Informasi Proyek**"]
    )

    # --- Konten Tab 1: Prediksi ---
    with tab1:
        with st.expander("📝 Buka Form untuk Input Data Pasien", expanded=True):
            # Input form
            with st.form("prediction_form_main"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Data Fisik & Pribadi**")
                    age = st.number_input("Usia (tahun)", 10, 100, 25, 1)
                    height = st.number_input(
                        "Tinggi Badan (meter)", 1.0, 2.5, 1.70, 0.01
                    )
                    weight = st.number_input("Berat Badan (kg)", 30.0, 200.0, 70.0, 0.5)
                    gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
                    family_history = st.selectbox(
                        "Riwayat keluarga obesitas?", ["yes", "no"]
                    )

                with col2:
                    st.markdown("**Kebiasaan & Gaya Hidup**")
                    favc = st.selectbox(
                        "Sering konsumsi makanan tinggi kalori?", ["yes", "no"]
                    )
                    fcvc = st.slider("Konsumsi sayuran (skala 1-3)", 1.0, 3.0, 2.0, 0.1)
                    ncp = st.slider(
                        "Frekuensi makan utama (skala 1-4)", 1.0, 4.0, 3.0, 0.1
                    )
                    ch2o = st.slider("Konsumsi air (skala 1-3)", 1.0, 3.0, 2.0, 0.1)
                    faf = st.slider(
                        "Frekuensi Aktivitas Fisik (skala 0-3)", 0.0, 3.0, 1.0, 0.1
                    )

                st.markdown("**Lainnya**")
                col3, col4, col5 = st.columns(3)
                with col3:
                    tue = st.slider(
                        "Penggunaan Teknologi (skala 0-2)", 0.0, 2.0, 1.0, 0.1
                    )
                    smoke = st.selectbox("Apakah merokok?", ["no", "yes"])
                with col4:
                    scc = st.selectbox("Memantau asupan kalori?", ["no", "yes"])
                    caec = st.selectbox(
                        "Konsumsi camilan", ["no", "Sometimes", "Frequently", "Always"]
                    )
                with col5:
                    calc = st.selectbox(
                        "Konsumsi alkohol", ["no", "Sometimes", "Frequently", "Always"]
                    )
                    mtrans = st.selectbox(
                        "Transportasi utama",
                        [
                            "Public_Transportation",
                            "Automobile",
                            "Walking",
                            "Motorbike",
                            "Bike",
                        ],
                    )

                predict_button = st.form_submit_button(
                    "🚀 Prediksi Tingkat Obesitas",
                    type="primary",
                    use_container_width=True,
                )

        # Inisialisasi session state untuk menyimpan hasil
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
            with st.spinner("Memproses prediksi..."):
                processed_input = preprocess_input(input_data, scaler, feature_names)
                prediction = model.predict(processed_input)[0]
                probabilities = model.predict_proba(processed_input)[0]
                predicted_class = label_encoder.inverse_transform([prediction])[0]
                bmi = calculate_bmi(weight, height)

                # Simpan hasil ke session state
                st.session_state.last_prediction = {
                    "input_data": input_data,
                    "prediction": prediction,
                    "probabilities": probabilities,
                    "predicted_class": predicted_class,
                    "bmi": bmi,
                }

        # Tampilkan hasil jika ada di session state
        if st.session_state.last_prediction:
            res = st.session_state.last_prediction
            st.divider()
            st.markdown(f"## 🔬 Dasbor Hasil Analisis")

            col_res1, col_res2 = st.columns([1, 1.5])
            with col_res1:
                st.markdown(
                    f'<div class="result-highlight-card"><p class="label">Prediksi Tingkat Obesitas</p><p class="result">{res["predicted_class"].replace("_", " ")}</p><p>Keyakinan Model: <strong>{res["probabilities"].max():.1%}</strong></p></div>',
                    unsafe_allow_html=True,
                )
                st.plotly_chart(create_bmi_gauge(res["bmi"]), use_container_width=True)
            with col_res2:
                with st.container(border=True, height=450):
                    obesity_info = get_obesity_info(res["predicted_class"])
                    st.markdown(
                        f"<h5 style='color:{obesity_info.get('color', '#2C3E50')};'>💡 Rekomendasi untuk: {res['predicted_class'].replace('_', ' ')}</h5>",
                        unsafe_allow_html=True,
                    )
                    st.write(f"**Deskripsi**: {obesity_info.get('description', 'N/A')}")
                    st.write(
                        f"**Rentang BMI**: {obesity_info.get('bmi_range', 'N/A')}"
                    )
                    for rec in obesity_info.get("recommendations", []):
                        st.write(f"- {rec}")

            with st.container(border=True):
                class_names = [c.replace("_", " ") for c in label_encoder.classes_]
                st.plotly_chart(
                    create_probability_chart(res["probabilities"], class_names),
                    use_container_width=True,
                )
        else:
            st.info(
                "Klik tombol prediksi untuk menampilkan dasbor hasil analisis di sini."
            )

    # --- Konten Tab 2: Analisis Input ---
    with tab2:
        st.header("Analisis Detail dari Data Input", divider="rainbow")
        if st.session_state.last_prediction:
            input_data = st.session_state.last_prediction["input_data"]
            bmi_val = st.session_state.last_prediction["bmi"]

            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    st.markdown("### 📏 Ringkasan Data Fisik")
                    st.metric("Usia", f"{input_data['Age']} tahun")
                    st.metric("Tinggi Badan", f"{input_data['Height']} m")
                    st.metric("Berat Badan", f"{input_data['Weight']} kg")
            with col2:
                with st.container(border=True):
                    st.markdown("### 🎯 Status Kategori")
                    if bmi_val < 18.5:
                        bmi_category, bmi_color = "Underweight", "🔵"
                    elif bmi_val < 25:
                        bmi_category, bmi_color = "Normal", "🟢"
                    elif bmi_val < 30:
                        bmi_category, bmi_color = "Overweight", "🟡"
                    else:
                        bmi_category, bmi_color = "Obesity", "🔴"
                    st.metric("Kategori BMI", f"{bmi_color} {bmi_category}")
                    st.metric("Kelompok Usia", categorize_age(input_data["Age"]))

            with st.container(border=True):
                st.markdown("### 🏃‍♂️ Analisis Gaya Hidup")
                lifestyle_data = {
                    "Aktivitas Fisik": input_data["FAF"],
                    "Konsumsi Sayuran": input_data["FCVC"],
                    "Konsumsi Air": input_data["CH2O"],
                    "Penggunaan Teknologi": input_data["TUE"],
                }
                lifestyle_df = pd.DataFrame(
                    list(lifestyle_data.items()), columns=["Aspek", "Nilai"]
                )
                fig_lifestyle = px.bar(
                    lifestyle_df,
                    x="Aspek",
                    y="Nilai",
                    title="Profil Gaya Hidup Anda",
                    color="Aspek",
                )
                st.plotly_chart(fig_lifestyle, use_container_width=True)
        else:
            st.warning(
                "Silakan lakukan prediksi di tab 'Prediksi' terlebih dahulu untuk melihat analisis input di sini."
            )

    # --- Konten Tab 3: Informasi Proyek ---
    with tab3:
        st.header("Informasi Detail Proyek", divider="rainbow")
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown("### 🎓 Tentang Proyek")
                st.write(
                    """**Capstone Project Bengkel Koding Data Science**\n\nUniversitas Dian Nuswantoro\n\n**Tujuan**: Mengembangkan model machine learning untuk prediksi tingkat obesitas berdasarkan kebiasaan makan dan kondisi fisik individu."""
                )
            with st.container(border=True):
                st.markdown("### 🔬 Alur Kerja (Metodologi)")
                st.write(
                    """1. **Analisis Data Eksploratif (EDA)**\n2. **Pra-pemrosesan & Feature Engineering**\n3. **Penanganan Imbalance (SMOTE)**\n4. **Pelatihan & Evaluasi Model**\n5. **Tuning Hyperparameter**\n6. **Deployment Aplikasi**"""
                )
        with col2:
            with st.container(border=True):
                st.markdown("### ⚠️ Disclaimer")
                st.warning(
                    "Aplikasi ini hanya untuk tujuan edukasi dan skrining awal, bukan pengganti konsultasi medis profesional.",
                    icon="❗",
                )
            with st.container(border=True):
                st.markdown("### 👨‍💻 Pengembang")
                st.write(
                    """**Firman Naufal Aryaputra** (A11.2022.14181)\n\nProgram Studi Teknik Informatika\n\n[GitHub Repository](https://github.com/Firmanarpp/Capstone_Bengkod_DS01_Firman.git)"""
                )


if __name__ == "__main__":
    main()
