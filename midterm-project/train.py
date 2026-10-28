#!/usr/bin/env python3
"""
Bank Term Deposit Prediction - Model Training Script
ML Zoomcamp Midterm Project
"""
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "bank.csv")
MODEL_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "model.bin")

FEATURE_COLUMNS = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "poutcome", "age", "balance", "day",
    "duration", "campaign", "pdays", "previous"
]

def load_and_preprocess_data(data_path: str):
    print(f"[*] Loading data from: {data_path}")
    df = pd.read_csv(data_path, sep=";")
    df["y"] = (df["y"] == "yes").astype(int)
    return df

def train_model(df: pd.DataFrame):
    print("[*] Splitting dataset into train (80%) and test (20%)...")
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["y"])
    
    y_train = df_train["y"].values
    y_test = df_test["y"].values
    
    train_dicts = df_train[FEATURE_COLUMNS].to_dict(orient="records")
    test_dicts = df_test[FEATURE_COLUMNS].to_dict(orient="records")
    
    print("[*] Vectorizing features with DictVectorizer...")
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dicts)
    X_test = dv.transform(test_dicts)
    
    print("[*] Training RandomForestClassifier (n_estimators=150, max_depth=10, min_samples_leaf=3)...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    print("[*] Evaluating model performance on test set...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, (y_pred_proba >= 0.5).astype(int))
    
    print(f"[+] Final Test ROC-AUC: {auc:.4f}")
    print(f"[+] Final Test Accuracy: {acc:.4f}")
    
    return dv, model

def save_model(dv, model, output_path: str):
    print(f"[*] Serializing pipeline and model to: {output_path}")
    with open(output_path, "wb") as f_out:
        pickle.dump((dv, model), f_out)
    print("[+] Model saved successfully!")

def main():
    df = load_and_preprocess_data(DATA_PATH)
    dv, model = train_model(df)
    save_model(dv, model, MODEL_OUTPUT_PATH)

if __name__ == "__main__":
    main()
