"""
app.py
Streamlit web app for Diabetes Risk Classification.
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import joblib
import os
import json
import sys

sys.path.insert(0, "src")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main { background-color: #0f172a; }

h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; }

.metric-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.metric-label {
    color: #94a3b8;
    font-size: 13px;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.metric-value {
    color: #f8fafc;
    font-size: 28px;
    font-weight: 700;
    margin-top: 4px;
}
.metric-sub {
    color: #64748b;
    font-size: 11px;
    margin-top: 2px;
}

.result-healthy {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid #10b981;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
}
.result-diabetic {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border: 1px solid #ef4444;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
}
.result-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 26px;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 8px;
}
.result-subtitle {
    color: #cbd5e1;
    font-size: 14px;
}

.prob-bar-bg {
    background: #1e293b;
    border-radius: 8px;
    height: 12px;
    width: 100%;
    overflow: hidden;
    margin: 6px 0;
}
.prob-bar-healthy {
    height: 100%;
    background: linear-gradient(90deg, #10b981, #34d399);
    border-radius: 8px;
    transition: width 0.5s ease;
}
.prob-bar-diabetic {
    height: 100%;
    background: linear-gradient(90deg, #ef4444, #f97316);
    border-radius: 8px;
    transition: width 0.5s ease;
}

.info-box {
    background: #1e293b;
    border-left: 3px solid #38bdf8;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 13px;
    color: #cbd5e1;
}

.section-header {
    color: #38bdf8;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 12px;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 8px;
}

.stSlider > div > div > div > div {
    background: #38bdf8 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Auto-train if model not found ─────────────────────────────────────────────
def auto_train():
    import subprocess
    os.makedirs("data",    exist_ok=True)
    os.makedirs("models",  exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    steps = [
        ("src/generate_data.py", "📦 Generating dataset..."),
        ("src/eda.py",           "📊 Running EDA..."),
        ("src/train.py",         "🤖 Training model (this may take a minute)..."),
    ]

    bar = st.progress(0, text="⏳ Setting up model for the first time...")
    for i, (script, msg) in enumerate(steps):
        bar.progress((i + 1) / len(steps), text=msg)
        result = subprocess.run([sys.executable, script], capture_output=True, text=True)
        if result.returncode != 0:
            st.error(f"Setup error:\n```\n{result.stderr[-1000:]}\n```")
            st.stop()

    bar.progress(1.0, text="✅ Model ready!")
    st.rerun()


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    metrics = {}
    if os.path.exists("reports/metrics.json"):
        with open("reports/metrics.json") as f:
            metrics = json.load(f)
    return model, scaler, metrics


if not os.path.exists("models/best_model.pkl"):
    st.info("🔧 First run detected — training model automatically. Please wait...")
    auto_train()

model, scaler, metrics = load_model()


# ── Feature engineering (mirror of preprocess.py) ────────────────────────────
def engineer_features(d: dict) -> np.ndarray:
    g, b, a = d["Glucose"], d["BMI"], d["Age"]
    d["GlucoseCategory"] = 0 if g <= 100 else (1 if g <= 125 else 2)
    d["BMICategory"]     = 0 if b < 18.5 else (1 if b < 25 else (2 if b < 30 else 3))
    d["AgeGroup"]        = 0 if a <= 30 else (1 if a <= 45 else (2 if a <= 60 else 3))
    features = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
                "Insulin","BMI","DiabetesPedigree","Age",
                "GlucoseCategory","BMICategory","AgeGroup"]
    return np.array([d[f] for f in features]).reshape(1, -1)


def predict(patient: dict):
    X = engineer_features(patient.copy())
    X_scaled = scaler.transform(X)
    pred  = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    return int(pred), float(proba[0]) * 100, float(proba[1]) * 100


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — Model Info
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🩺 About")
    st.markdown("""
    <div class='info-box'>
    Aplikasi ini menggunakan <b>Machine Learning</b> untuk memprediksi risiko diabetes 
    berdasarkan data klinis pasien.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🤖 Model Info")

    model_name = metrics.get("model", "SVM")
    st.markdown(f"**Algorithm:** `{model_name}`")

    if metrics:
        cols = st.columns(2)
        cols[0].metric("Accuracy",  f"{metrics.get('accuracy', 0)*100:.1f}%")
        cols[1].metric("F1-Score",  f"{metrics.get('f1', 0)*100:.1f}%")
        cols2 = st.columns(2)
        cols2[0].metric("Precision", f"{metrics.get('precision', 0)*100:.1f}%")
        cols2[1].metric("ROC-AUC",  f"{metrics.get('roc_auc', 0)*100:.1f}%")

    st.markdown("---")
    st.markdown("### 📊 Reference Ranges")
    st.markdown("""
    | Feature | Normal |
    |---------|--------|
    | Glucose | 70–100 mg/dL |
    | Blood Pressure | 60–80 mmHg |
    | BMI | 18.5–24.9 |
    | Insulin | 16–166 μU/mL |
    """)

    st.markdown("---")
    st.caption("⚠️ Untuk tujuan edukasi saja. Bukan pengganti diagnosis medis.")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN — Header
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<h1 style='font-family: Space Grotesk; font-size: 36px; color: #f8fafc; margin-bottom: 4px;'>
    🩺 Diabetes Risk Classifier
</h1>
<p style='color: #64748b; font-size: 15px; margin-bottom: 32px;'>
    Masukkan data klinis pasien untuk mendapatkan prediksi risiko diabetes berbasis Machine Learning
</p>
""", unsafe_allow_html=True)

# ── Quick fill buttons ────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>🚀 Quick Fill Demo</div>", unsafe_allow_html=True)
col_b1, col_b2, col_b3, col_b4 = st.columns([1, 1, 1, 3])

demo_patients = {
    "low":  dict(Pregnancies=1,  Glucose=89,  BloodPressure=66, SkinThickness=23, Insulin=94,  BMI=28.1, DiabetesPedigree=0.167, Age=21),
    "high": dict(Pregnancies=8,  Glucose=183, BloodPressure=64, SkinThickness=0,  Insulin=0,   BMI=23.3, DiabetesPedigree=0.672, Age=32),
    "mid":  dict(Pregnancies=3,  Glucose=120, BloodPressure=70, SkinThickness=28, Insulin=140, BMI=31.5, DiabetesPedigree=0.45,  Age=37),
}

if "patient" not in st.session_state:
    st.session_state.patient = {}

with col_b1:
    if st.button("✅ Low Risk", use_container_width=True):
        st.session_state.patient = demo_patients["low"]
with col_b2:
    if st.button("⚠️ High Risk", use_container_width=True):
        st.session_state.patient = demo_patients["high"]
with col_b3:
    if st.button("🟡 Borderline", use_container_width=True):
        st.session_state.patient = demo_patients["mid"]

p = st.session_state.patient

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  INPUT FORM
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>📋 Data Klinis Pasien</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    pregnancies = st.slider(
        "🤰 Jumlah Kehamilan",
        min_value=0, max_value=20,
        value=p.get("Pregnancies", 2),
        help="Jumlah total kehamilan pasien"
    )
    glucose = st.slider(
        "🍬 Glukosa Plasma (mg/dL)",
        min_value=60, max_value=210,
        value=p.get("Glucose", 100),
        help="Konsentrasi glukosa 2 jam setelah tes toleransi oral. Normal: 70–100"
    )
    blood_pressure = st.slider(
        "💓 Tekanan Darah Diastolik (mmHg)",
        min_value=40, max_value=130,
        value=p.get("BloodPressure", 70),
        help="Tekanan darah diastolik. Normal: 60–80"
    )
    skin_thickness = st.slider(
        "📏 Ketebalan Kulit Trisep (mm)",
        min_value=0, max_value=100,
        value=p.get("SkinThickness", 20),
        help="Ketebalan lipatan kulit trisep. Normal: 10–40"
    )

with col2:
    insulin = st.slider(
        "💉 Insulin Serum (μU/mL)",
        min_value=0, max_value=400,
        value=p.get("Insulin", 80),
        help="Kadar insulin serum 2 jam. Normal: 16–166. 0 = tidak diketahui"
    )
    bmi = st.slider(
        "⚖️ BMI (kg/m²)",
        min_value=10.0, max_value=70.0,
        value=float(p.get("BMI", 27.0)),
        step=0.1,
        help="Body Mass Index. Normal: 18.5–24.9"
    )
    pedigree = st.slider(
        "🧬 Diabetes Pedigree Function",
        min_value=0.05, max_value=2.50,
        value=float(p.get("DiabetesPedigree", 0.4)),
        step=0.01,
        help="Fungsi yang menghitung kemungkinan diabetes berdasarkan riwayat keluarga"
    )
    age = st.slider(
        "🎂 Usia (tahun)",
        min_value=21, max_value=90,
        value=p.get("Age", 30),
        help="Usia pasien dalam tahun"
    )

# ── BMI & Glucose info pills ──────────────────────────────────────────────────
col_i1, col_i2, col_i3 = st.columns(3)
bmi_cat   = ["Underweight (<18.5)", "Normal (18.5–25)", "Overweight (25–30)", "Obese (>30)"]
bmi_idx   = 0 if bmi < 18.5 else (1 if bmi < 25 else (2 if bmi < 30 else 3))
gluc_cat  = ["Normal (≤100)", "Pre-diabetic (101–125)", "Diabetic range (>125)"]
gluc_idx  = 0 if glucose <= 100 else (1 if glucose <= 125 else 2)
bp_status = "Normal" if 60 <= blood_pressure <= 80 else ("Low" if blood_pressure < 60 else "High")

col_i1.info(f"BMI: **{bmi_cat[bmi_idx]}**")
col_i2.info(f"Glucose: **{gluc_cat[gluc_idx]}**")
col_i3.info(f"Blood Pressure: **{bp_status}**")

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PREDICT BUTTON
# ══════════════════════════════════════════════════════════════════════════════
_, btn_col, _ = st.columns([2, 1, 2])
with btn_col:
    predict_btn = st.button("🔍 Analisis Risiko", use_container_width=True, type="primary")

# ══════════════════════════════════════════════════════════════════════════════
#  RESULT
# ══════════════════════════════════════════════════════════════════════════════
if predict_btn:
    patient_data = dict(
        Pregnancies=pregnancies, Glucose=glucose, BloodPressure=blood_pressure,
        SkinThickness=skin_thickness, Insulin=insulin, BMI=bmi,
        DiabetesPedigree=pedigree, Age=age,
    )

    with st.spinner("Menganalisis data klinis..."):
        pred, prob_healthy, prob_diabetic = predict(patient_data)

    st.markdown("---")
    st.markdown("<div class='section-header'>📊 Hasil Prediksi</div>", unsafe_allow_html=True)

    res_col1, res_col2 = st.columns([1.2, 1], gap="large")

    with res_col1:
        if pred == 0:
            st.markdown(f"""
            <div class='result-healthy'>
                <div style='font-size:52px'>✅</div>
                <div class='result-title'>Risiko Rendah</div>
                <div class='result-subtitle'>Pasien diprediksi <b>tidak diabetes</b> berdasarkan data klinis yang diberikan.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-diabetic'>
                <div style='font-size:52px'>⚠️</div>
                <div class='result-title'>Risiko Tinggi</div>
                <div class='result-subtitle'>Pasien diprediksi <b>berisiko diabetes</b>. Disarankan konsultasi dengan dokter.</div>
            </div>
            """, unsafe_allow_html=True)

    with res_col2:
        st.markdown("#### Probabilitas")

        st.markdown(f"""
        <div style='margin-bottom:16px'>
            <div style='display:flex; justify-content:space-between; color:#94a3b8; font-size:13px'>
                <span>✅ Healthy</span><span><b style='color:#10b981'>{prob_healthy:.1f}%</b></span>
            </div>
            <div class='prob-bar-bg'>
                <div class='prob-bar-healthy' style='width:{prob_healthy}%'></div>
            </div>
        </div>
        <div>
            <div style='display:flex; justify-content:space-between; color:#94a3b8; font-size:13px'>
                <span>⚠️ Diabetic</span><span><b style='color:#ef4444'>{prob_diabetic:.1f}%</b></span>
            </div>
            <div class='prob-bar-bg'>
                <div class='prob-bar-diabetic' style='width:{prob_diabetic}%'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Ringkasan Input")
        summary = {
            "Glucose": f"{glucose} mg/dL",
            "BMI": f"{bmi:.1f} kg/m²",
            "Age": f"{age} tahun",
            "Blood Pressure": f"{blood_pressure} mmHg",
            "Pedigree": f"{pedigree:.3f}",
        }
        for k, v in summary.items():
            st.markdown(f"<div style='display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #1e293b;font-size:13px'><span style='color:#94a3b8'>{k}</span><span style='color:#f8fafc;font-weight:500'>{v}</span></div>", unsafe_allow_html=True)

    # ── Risk factors breakdown ─────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🔎 Faktor Risiko Teridentifikasi")
    risks = []
    if glucose > 125:    risks.append(("🍬 Glukosa tinggi",        f"{glucose} mg/dL — berada di range diabetik", "high"))
    elif glucose > 100:  risks.append(("🍬 Glukosa pre-diabetik",  f"{glucose} mg/dL — perlu dipantau",           "mid"))
    if bmi >= 30:        risks.append(("⚖️ Obesitas",              f"BMI {bmi:.1f} — berisiko lebih tinggi",       "high"))
    elif bmi >= 25:      risks.append(("⚖️ Overweight",            f"BMI {bmi:.1f} — sedikit di atas normal",     "mid"))
    if age >= 45:        risks.append(("🎂 Usia berisiko",          f"{age} tahun — risiko meningkat setelah 45",  "mid"))
    if pedigree >= 0.8:  risks.append(("🧬 Riwayat keluarga kuat", f"Pedigree {pedigree:.3f} — faktor genetik",   "high"))
    if blood_pressure > 90: risks.append(("💓 Tekanan darah tinggi", f"{blood_pressure} mmHg",                    "high"))

    if risks:
        r_cols = st.columns(min(len(risks), 3))
        colors = {"high": "#ef4444", "mid": "#f97316"}
        for i, (title, desc, level) in enumerate(risks):
            with r_cols[i % 3]:
                st.markdown(f"""
                <div style='background:#1e293b;border-left:3px solid {colors[level]};border-radius:0 8px 8px 0;padding:12px;margin-bottom:8px'>
                    <div style='color:#f8fafc;font-weight:600;font-size:13px'>{title}</div>
                    <div style='color:#94a3b8;font-size:12px;margin-top:4px'>{desc}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.success("✅ Tidak ada faktor risiko signifikan teridentifikasi dari data yang diberikan.")

    st.markdown("---")
    st.caption("⚠️ **Disclaimer:** Prediksi ini dihasilkan oleh model Machine Learning untuk tujuan edukasi dan portofolio. Bukan pengganti diagnosis medis profesional.")
