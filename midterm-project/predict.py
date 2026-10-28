#!/usr/bin/env python3
"""
Bank Term Deposit Prediction - Serving Script
ML Zoomcamp Midterm Project
"""
import os
import pickle
from flask import Flask, request, jsonify

MODEL_FILE = os.path.join(os.path.dirname(__file__), "model.bin")

with open(MODEL_FILE, "rb") as f_in:
    dv, model = pickle.load(f_in)

app = Flask("bank-marketing-predict")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "bank-term-deposit-predictor"})

@app.route("/predict", methods=["POST"])
def predict():
    customer = request.get_json()
    if not customer:
        return jsonify({"error": "No JSON payload provided"}), 400

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    will_subscribe = bool(y_pred >= 0.5)

    result = {
        "subscription_probability": round(float(y_pred), 4),
        "will_subscribe": will_subscribe
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
