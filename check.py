import pandas as pd
import matplotlib.pyplot as plt

# Load the generated file
fico = pd.read_csv("./output/fico_style_features.csv")

# --- Quick Overview ---
print(fico.info())
print(fico.describe().T)

# --- Check target balance ---
print(fico['TARGET'].value_counts(normalize=True))

# --- Check missing values ---
print(fico.isna().mean().sort_values(ascending=False).head(10))

# --- Plot a few key distributions ---
fico[['dpd_mean','on_time_ratio','total_utilization','aaoa_m']].hist(bins=30, figsize=(10,6))
plt.suptitle("Distributions of FICO-style Features", fontsize=14)
plt.show()
