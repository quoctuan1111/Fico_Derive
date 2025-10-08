import pandas as pd
import numpy as np

# Required datasets
app = pd.read_csv("./data/application_train.csv")
bureau = pd.read_csv("./data/bureau.csv")
install = pd.read_csv("./data/installments_payments.csv")

# Optional: include for credit card utilization
try:
    credit = pd.read_csv("./data/credit_card_balance.csv")
except FileNotFoundError:
    credit = None
    print("credit_card_balance.csv not found — skipping credit card utilization features.")

def months_between(a, b):
    """
    Compute number of months between two time points (in days).
    HomeCredit uses negative day counts to represent dates before application.
    """
    return abs(a - b) / 30.44  # convert from days to months

#1. Repayment Simulation (Payment History)
# Calculate DPD (Days Past Due) and On-time Ratio
install["DAYS_ENTRY_PAYMENT"] = install["DAYS_ENTRY_PAYMENT"].fillna(0)
install["dpd"] = (install["DAYS_ENTRY_PAYMENT"] - install["DAYS_INSTALMENT"]).clip(lower=0)
install["on_time"] = (install["dpd"] == 0).astype(int)

repay_feat = install.groupby("SK_ID_CURR").agg({
    "dpd": ["mean", "max"],
    "on_time": "mean",
    "SK_ID_PREV": "count"
})
repay_feat.columns = ["dpd_mean", "dpd_max", "on_time_ratio", "num_payments"]
repay_feat.reset_index(inplace=True)

# 2. Amounts Owed (Indebtedness / Utilization) 
# Calculate total credit and total debt per customer
bureau["CREDIT_ACTIVE_FLAG"] = (bureau["CREDIT_ACTIVE"] == "Active").astype(int)

owed_feat = bureau.groupby("SK_ID_CURR").agg({
    "AMT_CREDIT_SUM": "sum",           # Total credit amount
    "AMT_CREDIT_SUM_DEBT": "sum",      # Total outstanding debt
    "CREDIT_ACTIVE_FLAG": "sum"        # Number of active accounts
}).reset_index()

# Utilization ratio (clip to avoid extreme outliers)
owed_feat["total_utilization"] = (
    owed_feat["AMT_CREDIT_SUM_DEBT"] /
    owed_feat["AMT_CREDIT_SUM"].replace(0, np.nan)
).clip(upper=1.5)

# Optional: Add credit card utilization
if credit is not None:
    card_util = credit.groupby("SK_ID_CURR").agg({
        "AMT_BALANCE": "mean",
        "AMT_CREDIT_LIMIT_ACTUAL": "mean"
    }).reset_index()
    card_util["credit_card_utilization"] = (
        card_util["AMT_BALANCE"] / card_util["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
    ).clip(upper=1.5)
    owed_feat = owed_feat.merge(card_util[["SK_ID_CURR", "credit_card_utilization"]],
                                on="SK_ID_CURR", how="left")
else:
    owed_feat["credit_card_utilization"] = np.nan
    
#3. Credit History Length (Maturity
# Days in HomeCredit are negative → older = more negative
bureau["CREDIT_AGE_MONTHS"] = abs(bureau["DAYS_CREDIT"]) / 30.44

hist_feat = bureau.groupby("SK_ID_CURR").agg({
    "CREDIT_AGE_MONTHS": ["min", "max", "mean"]
}).reset_index()

hist_feat.columns = ["SK_ID_CURR", "oldest_account_m", "newest_account_m", "aaoa_m"]

#4. Merge All Features
fico_features = app[["SK_ID_CURR", "TARGET"]]
fico_features = fico_features.merge(repay_feat, on="SK_ID_CURR", how="left")
fico_features = fico_features.merge(owed_feat, on="SK_ID_CURR", how="left")
fico_features = fico_features.merge(hist_feat, on="SK_ID_CURR", how="left")

#5. Data Cleaning and Flags
fico_features.fillna({
    "dpd_mean": 0,
    "dpd_max": 0,
    "on_time_ratio": 0,
    "AMT_CREDIT_SUM": 0,
    "AMT_CREDIT_SUM_DEBT": 0,
    "total_utilization": 0,
    "credit_card_utilization": 0,
    "aaoa_m": 0
}, inplace=True)

# Flag customers with no credit history (thin-file)
fico_features["thin_file_flag"] = (fico_features["AMT_CREDIT_SUM"] == 0).astype(int)

fico_features.to_csv("./output/fico_style_features.csv", index=False)
print("FICO-style feature table saved to fico_style_features.csv")
print(fico_features.head())