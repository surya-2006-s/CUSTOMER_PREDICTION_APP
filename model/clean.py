import pandas as pd
from pathlib import Path

# ---------- Load raw data ----------
data_path = Path(__file__).parent / 'churn.csv'
df = pd.read_csv(data_path)

print("=" * 60)
print("BEFORE CLEANING")
print("=" * 60)
print("Shape:", df.shape)
print("TotalCharges dtype:", df['TotalCharges'].dtype)

# ---------- Fix TotalCharges ----------
# Convert to numeric; blank spaces ' ' become NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

missing_total = df['TotalCharges'].isnull().sum()
print(f"\nRows with invalid TotalCharges: {missing_total}")

# Drop those rows
df = df.dropna(subset=['TotalCharges'])

# ---------- Drop customerID ----------
# It's just an identifier, not useful for prediction
df = df.drop(columns=['customerID'])

print("\n" + "=" * 60)
print("AFTER CLEANING")
print("=" * 60)
print("Shape:", df.shape)
print("TotalCharges dtype:", df['TotalCharges'].dtype)
print("Columns:", list(df.columns))

# ---------- Save cleaned data ----------
out_path = Path(__file__).parent / 'churn_clean.csv'
df.to_csv(out_path, index=False)
print(f"\n✅ Saved cleaned data to: {out_path}")
