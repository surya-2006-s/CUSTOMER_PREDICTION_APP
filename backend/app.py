from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from pathlib import Path

# ---------- Setup ----------
BASE = Path(__file__).resolve().parent.parent   # project root
MODEL_PATH = BASE / 'model' / 'churn_model.pkl'

app = Flask(__name__)
CORS(app)   # allow the frontend (different origin) to call this API

# Load the model ONCE at startup (not on every request)
print("Loading model...")
model = joblib.load(MODEL_PATH)
print("✅ Model loaded.")


# ---------- Routes ----------
@app.route('/')
def home():
    return jsonify({
        "status": "ok",
        "message": "Churn Prediction API is running",
        "endpoint": "POST /predict"
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body received"}), 400

        # Convert incoming JSON to a single-row DataFrame
        df = pd.DataFrame([data])

        # Make prediction
        pred = int(model.predict(df)[0])
        prob = float(model.predict_proba(df)[0][1])

        return jsonify({
            "prediction": "Churn" if pred == 1 else "No Churn",
            "churn_probability": round(prob, 4),
            "confidence": f"{prob * 100:.2f}%"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
    