import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.utils import resample
 
# 1. Load & Train Model
# ===============================
@st.cache(allow_output_mutation=True)
def load_model():
    df = pd.read_csv("data1.csv", sep=";")
    df = df.drop(columns=["NO", "STASUS BMI"])
    df = df.rename(columns=lambda x: x.strip().lower().replace(" ", "_").replace("-", "_"))

    # Mapping kategori → angka
    df['jenis_kelamin'] = df['jenis_kelamin'].map({'L':0,'P':1})
    df['konsumsi_obat_ht'] = df['konsumsi_obat_ht'].map({'TIDAK':0,'YA':1})
    df['beresiko_hipertensi'] = df['beresiko_hipertensi'].map({'TIDAK':0,'YA':1})

    # Imputasi missing values
    num_cols = ['umur','bb','tb','bmi','td_sistolik','td_diastolik','gula_darah']
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    cat_cols = ['jenis_kelamin','konsumsi_obat_ht','beresiko_hipertensi']
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0]).astype(int)

    # Balance dataset
    class_0 = df[df['beresiko_hipertensi'] == 0]
    class_1 = df[df['beresiko_hipertensi'] == 1]
    class_1_over = resample(class_1, replace=True, n_samples=len(class_0), random_state=42)
    df_balanced = pd.concat([class_0, class_1_over])

    X = df_balanced.drop("beresiko_hipertensi", axis=1)
    y = df_balanced["beresiko_hipertensi"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    dt = DecisionTreeClassifier(random_state=42)
    bagging = BaggingClassifier(estimator=dt, n_estimators=100, random_state=42, bootstrap=True)
    bagging.fit(X_scaled, y)

    return bagging, scaler, X.columns.tolist()

model, scaler, feature_names = load_model()

# ===============================
# 2. Streamlit UI
# ===============================
st.title("🫀 Prediksi Risiko Hipertensi")
st.write("Masukkan data pasien untuk memprediksi apakah berisiko hipertensi.")

# Input user
jenis_kelamin = st.radio("Jenis Kelamin", ["Laki-laki", "Perempuan"])
umur = st.number_input("Umur", 1, 120, 30)
bb = st.number_input("Berat Badan (kg)", 30, 200, 60)
tb = st.number_input("Tinggi Badan (cm)", 100, 220, 165)
bmi = round(bb / ((tb/100)**2), 1)
td_sistolik = st.number_input("Tekanan Darah Sistolik", 80, 250, 120)
td_diastolik = st.number_input("Tekanan Darah Diastolik", 50, 150, 80)
konsumsi_obat_ht = st.radio("Konsumsi Obat HT", ["Ya", "Tidak"])
gula_darah = st.number_input("Gula Darah", 50, 400, 100)

# Mapping input
input_dict = {
    "jenis_kelamin": 0 if jenis_kelamin=="Laki-laki" else 1,
    "umur": umur,
    "bb": bb,
    "tb": tb,
    "bmi": bmi,
    "td_sistolik": td_sistolik,
    "td_diastolik": td_diastolik,
    "konsumsi_obat_ht": 1 if konsumsi_obat_ht=="Ya" else 0,
    "gula_darah": gula_darah
}

input_df = pd.DataFrame([input_dict])[feature_names]
input_scaled = scaler.transform(input_df)

if st.button("🔍 Prediksi"):
    pred = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]
    if pred == 1:
        st.error(f"⚠️ Pasien **BERISIKO** hipertensi (Probabilitas: {proba[1]:.2f})")
    else:
        st.success(f"✅ Pasien **TIDAK BERISIKO** hipertensi (Probabilitas: {proba[0]:.2f})")

    
