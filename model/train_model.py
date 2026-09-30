import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------- Paths ----------
BASE = Path(__file__).parent
DATA_PATH = BASE / 'churn_clean.csv'
MODEL_PATH = BASE / 'churn_model.pkl'

# ---------- Load cleaned data ----------
df = pd.read_csv(DATA_PATH)
print("Data shape:", df.shape)

# ---------- Split features and target ----------
X = df.drop(columns=['Churn'])
y = df['Churn'].map({'Yes': 1, 'No': 0})   # convert target to 0/1

# ---------- Identify column types ----------
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(exclude=['object']).columns.tolist()

print("\nCategorical columns:", len(categorical_cols))
print("Numeric columns:", len(numeric_cols), numeric_cols)

# ---------- Build preprocessing pipeline ----------
# OneHotEncoder converts each category into a 0/1 column
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numeric_cols)
    ]
)

# ---------- Full pipeline: preprocess + classifier ----------
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# ---------- Split train/test ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nTrain size:", X_train.shape, "| Test size:", X_test.shape)

# ---------- Train ----------
print("\nTraining...")
model.fit(X_train, y_train)

# ---------- Evaluate ----------
y_pred = model.predict(X_test)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# ---------- Save model ----------
joblib.dump(model, MODEL_PATH)
print(f"\n✅ Model saved to: {MODEL_PATH}")

