# 🩺 Diabetes Risk Classification

> **Machine Learning project** untuk memprediksi risiko diabetes berdasarkan data klinis pasien.  
> Membandingkan 5 algoritma klasifikasi dan memilih model terbaik secara otomatis.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange?style=flat-square)](https://scikit-learn.org)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)]()

---

## 📋 Daftar Isi
- [Gambaran Umum](#gambaran-umum)
- [Dataset](#dataset)
- [Metodologi](#metodologi)
- [Hasil](#hasil)
- [Struktur Proyek](#struktur-proyek)
- [Cara Menjalankan](#cara-menjalankan)
- [Teknologi](#teknologi)

---

## Gambaran Umum

Proyek ini membangun pipeline **end-to-end Machine Learning** untuk mengklasifikasikan apakah seorang pasien berisiko terkena diabetes berdasarkan 8 fitur klinis seperti kadar glukosa, BMI, tekanan darah, dan lainnya.

**Tujuan utama:**
- Memahami distribusi dan hubungan antar fitur (EDA)
- Membandingkan beberapa algoritma klasifikasi secara objektif
- Memilih dan menyimpan model terbaik untuk prediksi
- Menyediakan antarmuka prediksi yang mudah digunakan

---

## Dataset

Dataset berisi **768 rekam medis pasien** dengan distribusi:

| Kelas     | Jumlah | Persentase |
|-----------|--------|------------|
| Healthy   | 460    | ~60%       |
| Diabetic  | 308    | ~40%       |

### Fitur Input

| Fitur              | Deskripsi                          | Satuan     |
|--------------------|------------------------------------|------------|
| `Pregnancies`      | Jumlah kehamilan                   | -          |
| `Glucose`          | Konsentrasi glukosa plasma         | mg/dL      |
| `BloodPressure`    | Tekanan darah diastolik            | mmHg       |
| `SkinThickness`    | Ketebalan lipatan kulit trisep     | mm         |
| `Insulin`          | Insulin serum 2 jam                | μU/mL      |
| `BMI`              | Body Mass Index                    | kg/m²      |
| `DiabetesPedigree` | Fungsi silsilah diabetes           | -          |
| `Age`              | Usia pasien                        | tahun      |

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
Save Model           ← models/best_model.pkl
```

---

## Hasil

### Perbandingan Model (5-Fold Cross-Validation F1)

| Model                | CV F1 (mean ± std) |
|----------------------|-------------------|
| Logistic Regression  | 0.9795 ± 0.0112   |
| **SVM**              | **0.9876 ± 0.0077** |
| Gradient Boosting    | 0.9797 ± 0.0129   |
| Random Forest        | 0.9817 ± 0.0151   |
| K-Nearest Neighbors  | 0.9724 ± 0.0204   |

### Evaluasi Model Terbaik: SVM (rbf, C=1)

| Metrik    | Score  |
|-----------|--------|
| Accuracy  | 98.7%  |
| Precision | 100.0% |
| Recall    | 96.8%  |
| F1-Score  | 98.4%  |
| ROC-AUC   | 100.0% |

### Visualisasi

| EDA Report | Model Evaluation |
|---|---|
| `reports/eda_report.png` | `reports/model_evaluation.png` |

---

## Struktur Proyek

```
health-classification/
│
├── data/
│   └── diabetes.csv             # Dataset (generated)
│
├── models/
│   ├── best_model.pkl           # Trained model (SVM)
│   └── scaler.pkl               # StandardScaler
│
├── reports/
│   ├── eda_report.png           # EDA visualizations
│   ├── model_evaluation.png     # Model comparison plots
│   └── metrics.json             # Final metrics
│
├── src/
│   ├── generate_data.py         # Synthetic dataset generator
│   ├── preprocess.py            # Cleaning & feature engineering
│   ├── eda.py                   # Exploratory Data Analysis
│   ├── train.py                 # Model training & evaluation
│   └── predict.py               # Inference / prediction
│
├── main.py                      # 🚀 Full pipeline runner
├── requirements.txt
└── README.md
```

---

## Cara Menjalankan

### 1. Clone & Install

```bash
git clone https://github.com/username/health-classification.git
cd health-classification
pip install -r requirements.txt
```

### 2. Jalankan Pipeline Lengkap

```bash
python main.py
```

Akan menjalankan secara berurutan:
1. Generate dataset
2. EDA & visualisasi
3. Training & evaluasi model
4. Demo prediksi

### 3. Prediksi Interaktif

```bash
python src/predict.py --interactive
```

Masukkan data klinis pasien dan dapatkan prediksi risiko diabetes secara langsung.

### 4. Prediksi dari Kode Python

```python
from src.predict import predict

patient = {
    "Pregnancies":      2,
    "Glucose":          130,
    "BloodPressure":    72,
    "SkinThickness":    30,
    "Insulin":          120,
    "BMI":              33.0,
    "DiabetesPedigree": 0.52,
    "Age":              35,
}

result = predict(patient)
print(result)
# {'prediction': 1, 'label': '⚠️  Diabetic (High Risk)',
#  'probability': {'Healthy': 18.3, 'Diabetic': 81.7}}
```

---

## Teknologi

| Library      | Kegunaan                            |
|--------------|-------------------------------------|
| pandas       | Manipulasi & analisis data          |
| numpy        | Operasi numerik                     |
| scikit-learn | ML algorithms, preprocessing, tuning|
| matplotlib   | Visualisasi data                    |
| seaborn      | Statistical visualizations          |
| joblib       | Model serialization                 |

---

## Konsep yang Dipraktikkan

- ✅ Data Cleaning & Imputation
- ✅ Feature Engineering (domain knowledge)
- ✅ Exploratory Data Analysis (EDA)
- ✅ Model Comparison (5 algorithms)
- ✅ Cross-Validation
- ✅ Hyperparameter Tuning (GridSearchCV)
- ✅ Model Evaluation (multi-metric)
- ✅ Model Serialization & Deployment-ready inference
- ✅ Clean project structure & documentation

---

## Author

**[Nama Kamu]**  
Mahasiswa Teknik Informatika | Data Science & Machine Learning Enthusiast  

[![GitHub](https://img.shields.io/badge/GitHub-username-black?style=flat-square&logo=github)](https://github.com/username)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?style=flat-square&logo=linkedin)](https://linkedin.com)

---

> *"Data is the new oil, but only if refined."*
