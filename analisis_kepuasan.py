# ==========================================
# ANALISIS DATA PELANGGAN E-COMMERCE
# Menggunakan Random Forest Regressor
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# MEMBACA DATASET
# ==========================================

df = pd.read_csv("ecommerce_customer_satisfaction.csv")

print("===== 5 DATA PERTAMA =====")
print(df.head())

print("\n===== INFORMASI DATASET =====")
print(df.info())

# ==========================================
# MEMILIH FITUR DAN TARGET
# ==========================================

target = "customer_zip_code_prefix"

features = [
    "customer_city",
    "customer_state"
]

X = df[features].copy()
y = df[target]

# ==========================================
# ENCODING DATA KATEGORIK
# ==========================================

le_city = LabelEncoder()
le_state = LabelEncoder()

X["customer_city"] = le_city.fit_transform(
    X["customer_city"]
)

X["customer_state"] = le_state.fit_transform(
    X["customer_state"]
)

# ==========================================
# SPLIT DATA TRAINING DAN TESTING
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# MEMBANGUN MODEL RANDOM FOREST
# ==========================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================
# PREDIKSI
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# EVALUASI MODEL
# ==========================================

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n===== HASIL EVALUASI =====")
print("Mean Absolute Error :", round(mae, 3))
print("R2 Score :", round(r2, 3))

# ==========================================
# ANALISIS PENGARUH FAKTOR
# ==========================================

importance = model.feature_importances_

hasil = pd.DataFrame({
    "Faktor": features,
    "Pengaruh": importance
})

hasil = hasil.sort_values(
    by="Pengaruh",
    ascending=False
)

print("\n===== TINGKAT PENGARUH FAKTOR =====")
print(hasil)

# ==========================================
# VISUALISASI
# ==========================================

plt.figure(figsize=(8,5))

plt.bar(
    hasil["Faktor"],
    hasil["Pengaruh"]
)

plt.title(
    "Pengaruh Faktor Terhadap Customer Zip Code Prefix"
)

plt.xlabel("Faktor")
plt.ylabel("Nilai Pengaruh")

plt.tight_layout()
plt.show()