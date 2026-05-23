# 🩺 Diabetes Risk Classification

> **Machine Learning project** untuk memprediksi risiko diabetes berdasarkan data klinis pasien,  
> dilengkapi dengan **web app interaktif berbasis Streamlit**.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange?style=flat-square)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?style=flat-square&logo=streamlit)](https://streamlit.io)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)]()
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?logo=streamlit&logoColor=white)](https://health-classification.streamlit.app/)

---

## 🎯 Demo App

Jalankan web app interaktif:
```bash
python -m streamlit run app.py
```
Buka browser → `http://localhost:8501`



---

## 📋 Daftar Isi
- [Gambaran Umum](#gambaran-umum)
- [Fitur Aplikasi](#fitur-aplikasi)
- [Dataset](#dataset)
- [Metodologi](#metodologi)
- [Hasil Model](#hasil-model)
- [Struktur Proyek](#struktur-proyek)
- [Cara Menjalankan](#cara-menjalankan)
- [Teknologi](#teknologi)

---

## Gambaran Umum

Proyek ini membangun pipeline **end-to-end Machine Learning** untuk mengklasifikasikan apakah seorang pasien berisiko terkena diabetes berdasarkan 8 fitur klinis seperti kadar glukosa, BMI, tekanan darah, dan lainnya.

Hasil akhir berupa **web app Streamlit** yang memungkinkan pengguna memasukkan data klinis pasien dan mendapatkan prediksi risiko secara real-time.

**Tujuan utama:**
- Memahami distribusi dan hubungan antar fitur melalui EDA
- Membandingkan 5 algoritma klasifikasi secara objektif
- Memilih dan menyimpan model terbaik secara otomatis
- Menyajikan prediksi melalui antarmuka web yang interaktif

---

## ✨ Fitur Aplikasi (Streamlit)

| Fitur | Deskripsi |
|-------|-----------|
| 🚀 Quick Fill Demo | Tombol isi otomatis data Low Risk / High Risk / Borderline |
| 🎚️ Slider Input | Input semua fitur klinis dengan hint nilai normal |
| 📊 Info Pills | Status real-time BMI, Glukosa, Tekanan Darah |
| 🎯 Hasil Prediksi | Tampilan hasil dengan warna + probability bar |
| 🔎 Risk Factor Breakdown | Highlight otomatis faktor risiko yang terdeteksi |
| 📈 Sidebar Model Info | Akurasi model dan tabel reference range klinis |

---

## Dataset

Dataset berisi **768 rekam medis pasien** yang di-generate secara sintetis berdasarkan distribusi klinis realistis.

| Kelas     | Jumlah | Persentase |
|-----------|--------|------------|
| Healthy   | 460    | ~60%       |
| Diabetic  | 308    | ~40%       |

### Fitur Input

| Fitur              | Deskripsi                          | Satuan  | Normal Range   |
|--------------------|------------------------------------|---------|----------------|
| `Pregnancies`      | Jumlah kehamilan                   | -       | -              |
| `Glucose`          | Konsentrasi glukosa plasma         | mg/dL   | 70–100         |
| `BloodPressure`    | Tekanan darah diastolik            | mmHg    | 60–80          |
| `SkinThickness`    | Ketebalan lipatan kulit trisep     | mm      | 10–40          |
| `Insulin`          | Insulin serum 2 jam                | μU/mL   | 16–166         |
| `BMI`              | Body Mass Index                    | kg/m²   | 18.5–24.9      |
| `DiabetesPedigree` | Fungsi silsilah diabetes           | -       | < 0.5          |
| `Age`              | Usia pasien                        | tahun   | -              |

---

## Metodologi

```
Raw Data
   │
   ▼
Data Cleaning        ← Imputasi nilai nol/missing dengan median
   │
   ▼
Feature Engineering  ← GlucoseCategory, BMICategory, AgeGroup
   │
   ▼
Train/Test Split     ← 80% train, 20% test, stratified
   │
   ▼
StandardScaler       ← Normalisasi fitur
   │
   ▼
Cross-Validation     ← 5-fold CV pada 5 model
   │
   ▼
GridSearchCV         ← Hyperparameter tuning model terbaik
   │
   ▼
Evaluation           ← Accuracy, Precision, Recall, F1, ROC-AUC
   │
   ▼
Streamlit App        ← Web interface untuk prediksi interaktif
```

---

## Hasil Model

### Perbandingan 5 Model (5-Fold Cross-Validation F1)

| Model                | CV F1 (mean ± std)    |
|----------------------|-----------------------|
| Logistic Regression  | 0.9795 ± 0.0112       |
| Random Forest        | 0.9817 ± 0.0151       |
| Gradient Boosting    | 0.9797 ± 0.0129       |
| **SVM ✅ Best**      | **0.9876 ± 0.0077**   |
| K-Nearest Neighbors  | 0.9724 ± 0.0204       |

### Evaluasi Model Terbaik: SVM (kernel=rbf, C=1)

| Metrik    | Score  |
|-----------|--------|
| Accuracy  | 98.7%  |
| Precision | 100.0% |
| Recall    | 96.8%  |
| F1-Score  | 98.4%  |
| ROC-AUC   | 100.0% |

---

## Struktur Proyek

```
health-classification/
│
├── app.py                       # 🌐 Streamlit web app
├── main.py                      # 🚀 Full pipeline runner
├── requirements.txt
├── README.md
│
├── src/
│   ├── generate_data.py         # Synthetic dataset generator
│   ├── preprocess.py            # Cleaning & feature engineering
│   ├── eda.py                   # Exploratory Data Analysis
│   ├── train.py                 # Model training & evaluation
│   └── predict.py               # Inference helper
│
├── models/                      # ⚙️  Auto-generated after training
│   ├── best_model.pkl
│   └── scaler.pkl
│
├── data/                        # 📂 Auto-generated after training
│   └── diabetes.csv
│
└── reports/                     # 📊 Auto-generated after training
    ├── eda_report.png
    ├── model_evaluation.png
    └── metrics.json
```

> 📌 Folder `models/`, `data/`, dan `reports/` di-generate otomatis saat menjalankan `python main.py`. Tidak perlu di-push ke GitHub.

---

## Cara Menjalankan

### 1. Clone & Install

```bash
git clone https://github.com/devin-novansyah16/Health-classification.git
cd Health-classification
pip install -r requirements.txt
```

### 2. Train Model (wajib sekali saja)

```bash
python main.py
```

Proses ini akan:
- Generate dataset (`data/diabetes.csv`)
- Membuat visualisasi EDA (`reports/eda_report.png`)
- Melatih & membandingkan 5 model
- Menyimpan model terbaik (`models/best_model.pkl`)

### 3. Jalankan Web App Streamlit

```bash
python -m streamlit run app.py
```

Buka browser → `http://localhost:8501`

### 4. Prediksi via Python (opsional)

```python
from src.predict import predict

patient = {
    "Pregnancies": 2, "Glucose": 130, "BloodPressure": 72,
    "SkinThickness": 30, "Insulin": 120, "BMI": 33.0,
    "DiabetesPedigree": 0.52, "Age": 35,
}

result = predict(patient)
print(result)
# {'prediction': 1, 'label': '⚠️  Diabetic (High Risk)',
#  'probability': {'Healthy': 18.3, 'Diabetic': 81.7}}
```

---

## Teknologi

| Library      | Versi    | Kegunaan                             |
|--------------|----------|--------------------------------------|
| pandas       | ≥ 1.5.0  | Manipulasi & analisis data           |
| numpy        | ≥ 1.23.0 | Operasi numerik                      |
| scikit-learn | ≥ 1.2.0  | ML algorithms, preprocessing, tuning |
| matplotlib   | ≥ 3.6.0  | Visualisasi data                     |
| seaborn      | ≥ 0.12.0 | Statistical visualizations           |
| joblib       | ≥ 1.2.0  | Model serialization                  |
| streamlit    | ≥ 1.28.0 | Web app interaktif                   |

---

## Konsep yang Dipraktikkan

- ✅ Data Cleaning & Imputation
- ✅ Feature Engineering (domain knowledge)
- ✅ Exploratory Data Analysis (EDA)
- ✅ Model Comparison (5 algorithms)
- ✅ Cross-Validation & Hyperparameter Tuning
- ✅ Multi-metric Model Evaluation
- ✅ Model Serialization
- ✅ Interactive Web App (Streamlit)
- ✅ Clean project structure & dokumentasi

---

## Author

**Devin Novansyah**  
Mahasiswa Teknik Informatika | Data Science & Machine Learning Enthusiast

[![GitHub](https://img.shields.io/badge/GitHub-devin--novansyah16-black?style=flat-square&logo=github)](https://github.com/devin-novansyah16)

---

> ⚠️ *Disclaimer: Prediksi ini untuk tujuan edukasi dan portofolio. Bukan pengganti diagnosis medis profesional.*
