#!/usr/bin/env python3
"""
Bank Term Deposit Prediction - Test Request Script
ML Zoomcamp Midterm Project
"""
import requests

URL = "http://localhost:9696/predict"

# Sample 1: High propensity customer (previous success, high balance, long duration)
customer_positive = {
    "age": 35,
    "job": "management",
    "marital": "single",
    "education": "tertiary",
    "default": "no",
    "balance": 5200,
    "housing": "no",
    "loan": "no",
    "contact": "cellular",
    "day": 15,
    "month": "aug",
    "duration": 580,
    "campaign": 1,
    "pdays": 90,
    "previous": 2,
    "poutcome": "success"
}

# Sample 2: Low propensity customer (defaulted, housing & personal loan, previous failure)
customer_negative = {
    "age": 28,
    "job": "blue-collar",
    "marital": "married",
    "education": "secondary",
    "default": "yes",
    "balance": -150,
    "housing": "yes",
    "loan": "yes",
    "contact": "unknown",
    "day": 5,
    "month": "may",
    "duration": 45,
    "campaign": 6,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown"
}

def test_customer(name: str, payload: dict):
    print(f"\n--- Testing: {name} ---")
    try:
        response = requests.post(URL, json=payload, timeout=5)
        response.raise_for_status()
        data = response.json()
        print(f"Response: {data}")
        prob = data["subscription_probability"]
        sub = data["will_subscribe"]
        print(f"Probability: {prob*100:.2f}% | Decision: {'SUBSCRIBE' if sub else 'NO SUBSCRIBE'}")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    print("[*] Sending test requests to prediction service...")
    test_customer("Customer A (High Propensity)", customer_positive)
    test_customer("Customer B (Low Propensity)", customer_negative)
