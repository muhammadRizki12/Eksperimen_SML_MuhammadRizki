import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder # Encoder
from sklearn.preprocessing import StandardScaler # Standarisasi

# Memuat dataset
df = pd.read_csv('../obesity_data_raw.csv')

num_features = df.select_dtypes(include=[np.number]).columns
cat_features = df.select_dtypes(include=['object']).columns

# Data Preprocessing

# IQR untuk penanganan data outlier
# Hitung Q1, Q3, dan IQR hanya untuk kolom numerikal
Q1 = df[num_features].quantile(0.25)
Q3 = df[num_features].quantile(0.75)
IQR = Q3 - Q1
# Buat filter untuk menghapus baris yang mengandung outlier di kolom numerikal
filter_outliers = ~((df[num_features] < (Q1 - 1.5 * IQR)) |
                    (df[num_features] > (Q3 + 1.5 * IQR))).any(axis=1)
# Terapkan filter ke dataset asli (termasuk kolom non-numerikal)
df = df[filter_outliers]
# Cek ukuran dataset setelah outlier dihapus
df.shape

# Standarisasi
scaler = StandardScaler()
df[num_features] = scaler.fit_transform(df[num_features])

# Encoding
label_encoder = LabelEncoder()

# Encode kolom kategorikal
for column in cat_features:
    df[column] = label_encoder.fit_transform(df[column])

# Simpan CSV ke folder tersebut
df.to_csv('obesity_data_preprocessing.csv', index=False)