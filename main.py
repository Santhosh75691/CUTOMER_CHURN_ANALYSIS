# ============================================================
# CUSTOMER CHURN ANALYSIS
# Beginner Data Analytics Project
# ============================================================

# Install these libraries once in Terminal / Command Prompt:
# pip install pandas numpy matplotlib seaborn

# STEP 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# STEP 2: Load the dataset
# Keep customer_churn_analysis.csv in the same folder as this Python file.
file_path = "customer_churn_analysis.csv"

df = pd.read_csv(file_path)

print("\n================ DATA LOADED ================")
print(df.head())

# STEP 3: Understand the dataset
print("\n================ SHAPE ================")
print(df.shape)

print("\n================ COLUMNS ================")
print(df.columns.tolist())

print("\n================ DATA TYPES ================")
print(df.dtypes)

print("\n================ MISSING VALUES ================")
print(df.isnull().sum())

print("\n================ DUPLICATE ROWS ================")
print(df.duplicated().sum())

print("\n================ STATISTICAL SUMMARY ================")
print(df.describe(include="all"))

# STEP 4: Data cleaning
# Remove duplicate records
df = df.drop_duplicates()

# Convert numeric columns to numeric format
numeric_columns = [
    "Age", "TenureMonths", "MonthlyCharges",
    "TotalCharges", "SupportTickets", "SatisfactionScore"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill missing categorical values with the mode
categorical_columns = ["OnlineSecurity", "TechSupport"]

for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Fill missing satisfaction scores with the median
df["SatisfactionScore"] = df["SatisfactionScore"].fillna(
    df["SatisfactionScore"].median()
)

print("\n================ AFTER CLEANING ================")
print(df.isnull().sum())

# STEP 5: Create useful analysis columns

# Convert Churn into 1/0 for calculations
df["ChurnFlag"] = df["Churn"].map({"Yes": 1, "No": 0})

# Monthly revenue at risk from churned customers
df["ChurnedMonthlyRevenue"] = np.where(
    df["Churn"] == "Yes",
    df["MonthlyCharges"],
    0
)

# Customer tenure group
df["TenureGroup"] = pd.cut(
    df["TenureMonths"],
    bins=[0, 12, 24, 48, 72],
    labels=["0-12 Months", "13-24 Months", "25-48 Months", "49-72 Months"]
)

# Age group
df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[17, 30, 45, 60, 100],
    labels=["18-30", "31-45", "46-60", "61+"]
)

# STEP 6: Basic KPI calculations
total_customers = len(df)
churned_customers = df["ChurnFlag"].sum()
retained_customers = total_customers - churned_customers
churn_rate = df["ChurnFlag"].mean() * 100
average_monthly_charge = df["MonthlyCharges"].mean()
total_monthly_revenue = df["MonthlyCharges"].sum()
churned_monthly_revenue = df["ChurnedMonthlyRevenue"].sum()

print("\n================ KEY PERFORMANCE INDICATORS ================")
print("Total Customers:", total_customers)
print("Churned Customers:", churned_customers)
print("Retained Customers:", retained_customers)
print("Churn Rate:", round(churn_rate, 2), "%")
print("Average Monthly Charge:", round(average_monthly_charge, 2))
print("Total Monthly Revenue:", round(total_monthly_revenue, 2))
print("Monthly Revenue from Churned Customers:", round(churned_monthly_revenue, 2))

# STEP 7: Churn by contract type
contract_churn = (
    df.groupby("ContractType")["ChurnFlag"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

print("\n================ CHURN RATE BY CONTRACT ================")
print(contract_churn)

# STEP 8: Churn by internet service
internet_churn = (
    df.groupby("InternetService")["ChurnFlag"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

print("\n================ CHURN RATE BY INTERNET SERVICE ================")
print(internet_churn)

# STEP 9: Churn by payment method
payment_churn = (
    df.groupby("PaymentMethod")["ChurnFlag"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

print("\n================ CHURN RATE BY PAYMENT METHOD ================")
print(payment_churn)

# STEP 10: Churn by tenure group
tenure_churn = (
    df.groupby("TenureGroup", observed=False)["ChurnFlag"]
    .mean()
    .mul(100)
)

print("\n================ CHURN RATE BY TENURE ================")
print(tenure_churn)

# STEP 11: Average charges for churned vs retained customers
charge_analysis = (
    df.groupby("Churn")[["MonthlyCharges", "TotalCharges", "SatisfactionScore"]]
    .mean()
    .round(2)
)

print("\n================ CHURN VS RETENTION ================")
print(charge_analysis)

# STEP 12: Correlation analysis
correlation_columns = [
    "Age", "TenureMonths", "MonthlyCharges",
    "TotalCharges", "SupportTickets",
    "SatisfactionScore", "ChurnFlag"
]

correlation = df[correlation_columns].corr()

print("\n================ CORRELATION MATRIX ================")
print(correlation.round(2))

# STEP 13: VISUALIZATION 1 - Churn distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# STEP 14: VISUALIZATION 2 - Churn rate by contract
plt.figure(figsize=(8, 5))
contract_churn.plot(kind="bar")
plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# STEP 15: VISUALIZATION 3 - Churn rate by internet service
plt.figure(figsize=(8, 5))
internet_churn.plot(kind="bar")
plt.title("Churn Rate by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# STEP 16: VISUALIZATION 4 - Monthly charges vs churn
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Monthly Charges by Churn Status")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.show()

# STEP 17: VISUALIZATION 5 - Tenure vs churn
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Churn", y="TenureMonths")
plt.title("Tenure by Churn Status")
plt.xlabel("Churn")
plt.ylabel("Tenure (Months)")
plt.tight_layout()
plt.show()

# STEP 18: VISUALIZATION 6 - Satisfaction vs churn
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="SatisfactionScore", hue="Churn")
plt.title("Satisfaction Score vs Churn")
plt.xlabel("Satisfaction Score")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# STEP 19: VISUALIZATION 7 - Support tickets vs churn
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Churn", y="SupportTickets")
plt.title("Support Tickets by Churn Status")
plt.xlabel("Churn")
plt.ylabel("Support Tickets")
plt.tight_layout()
plt.show()

# STEP 20: VISUALIZATION 8 - Correlation heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# STEP 21: Create a customer-risk table
df["RiskLevel"] = np.select(
    [
        (df["Churn"] == "Yes"),
        (df["Churn"] == "No") & (df["SatisfactionScore"] <= 2),
        (df["Churn"] == "No") & (df["SupportTickets"] >= 5)
    ],
    [
        "High Risk",
        "Medium Risk",
        "Medium Risk"
    ],
    default="Low Risk"
)

risk_summary = df["RiskLevel"].value_counts()

print("\n================ CUSTOMER RISK SUMMARY ================")
print(risk_summary)

# STEP 22: Save cleaned dataset for Power BI
output_file = "customer_churn_analysis_cleaned.csv"
df.to_csv(output_file, index=False)

print("\n========================================================")
print("PROJECT COMPLETED SUCCESSFULLY")
print("Cleaned dataset saved as:", output_file)
print("========================================================")
