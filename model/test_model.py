import joblib
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent

# Load the trained model
model = joblib.load(BASE / 'churn_model.pkl')
print("Model loaded:", type(model).__name__)

# Load a few sample customers from the cleaned data
df = pd.read_csv(BASE / 'churn_clean.csv')

# Take the first 3 customers and predict
samples = df.drop(columns=['Churn']).head(3)
actual = df['Churn'].head(3).tolist()

predictions = model.predict(samples)
probabilities = model.predict_proba(samples)[:, 1]  # probability of Churn

print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

for i in range(3):
    pred_label = "Churn" if predictions[i] == 1 else "No Churn"
    print(f"\nCustomer {i+1}:")
    print(f"  Actual:      {actual[i]}")
    print(f"  Predicted:   {pred_label}")
    print(f"  Churn prob:  {probabilities[i]:.2%}")
    