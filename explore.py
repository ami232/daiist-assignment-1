import pandas as pd

df = pd.read_csv("data/telco.csv")
df["Churn"] = (df["Churn"] == "Yes").astype(int)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

for col in ["Contract", "InternetService", "PaymentMethod", "TechSupport", "SeniorCitizen"]:
     print(df.groupby(col)["Churn"].agg(["mean", "count"]).round(3), "\n")

df["tenure_bin"] = pd.cut(df["tenure"], [-1, 6, 12, 24, 48, 72])
print(df.groupby("tenure_bin")["Churn"].agg(["mean", "count"]).round(3), "\n")
print(df[["tenure", "MonthlyCharges", "TotalCharges"]].corr().round(2))
