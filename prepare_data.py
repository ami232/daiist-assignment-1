import pandas as pd

RAW = "raw/accepted_2007_to_2018Q4.csv.gz"
OUT = "data/loans.csv"

cols = [
    "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
    "emp_length", "home_ownership", "annual_inc", "verification_status",
    "issue_d", "purpose", "addr_state", "dti", "delinq_2yrs",
    "earliest_cr_line", "fico_range_low", "fico_range_high",
    "inq_last_6mths", "mths_since_last_delinq", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "loan_status",
]

df = pd.read_csv(RAW, usecols=cols)

df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])]
df = df[df["term"].str.strip() == "36 months"]
df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
df = df[(df["issue_d"] >= "2012-01-01") & (df["issue_d"] < "2015-01-01")]

df["default"] = (df["loan_status"] == "Charged Off").astype(int)
df = df.drop(columns=["loan_status"])

df = df.sample(n=18000, random_state=42).sort_values("issue_d")
df.to_csv(OUT, index=False)
print(df.shape, df["default"].mean())