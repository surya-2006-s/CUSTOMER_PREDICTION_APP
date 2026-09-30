import pandas as pd
from pathlib import Path

# Load the dataset — path relative to THIS script's folder
df = pd.read_csv(Path(__file__).parent / 'churn.csv')

# ---------- Basic info ----------
print("=" * 60)
print("SHAPE:", df.shape)
print("=" * 60)

print("\nCOLUMN TYPES:")
print(df.dtypes)

print("\n" + "=" * 60)
print("MISSING VALUES PER COLUMN:")
print("=" * 60)
print(df.isnull().sum())

print("\n" + "=" * 60)
print("TARGET (Churn) DISTRIBUTION:")
print("=" * 60)
print(df['Churn'].value_counts())
print(f"\nChurn rate: {(df['Churn'] == 'Yes').mean() * 100:.2f}%")

print("\n" + "=" * 60)
print("BASIC STATS FOR NUMERIC COLUMNS:")
print("=" * 60)
print(df.describe())