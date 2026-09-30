const API_URL = "http://127.0.0.1:5000/predict";

const form = document.getElementById("churn-form");
const resultBox = document.getElementById("result");
const submitBtn = document.getElementById("predict-btn");

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  // Build JSON from form fields
  const formData = new FormData(form);
  const payload = {};
  formData.forEach((value, key) => {
    // Convert numeric fields to numbers
    if (key === "tenure" || key === "SeniorCitizen") {
      payload[key] = parseInt(value, 10);
    } else if (key === "MonthlyCharges" || key === "TotalCharges") {
      payload[key] = parseFloat(value);
    } else {
      payload[key] = value;
    }
  });

  // UI feedback
  submitBtn.disabled = true;
  submitBtn.textContent = "Predicting...";
  resultBox.className = "result hidden";

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`Server returned ${response.status}`);
    }

    const data = await response.json();

    // Show result
    const isChurn = data.prediction === "Churn";
    resultBox.className = `result ${isChurn ? "churn" : "no-churn"}`;
    resultBox.innerHTML = `
      <div class="label">Prediction</div>
      <div class="prediction">${data.prediction}</div>
      <div class="prob">Churn probability: ${data.confidence}</div>
    `;
  } catch (err) {
    resultBox.className = "result error";
    resultBox.innerHTML = `
      <div class="label">Error</div>
      <div class="prediction">Could not get prediction</div>
      <div class="prob">${err.message}</div>
    `;
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Predict Churn";
  }
});
