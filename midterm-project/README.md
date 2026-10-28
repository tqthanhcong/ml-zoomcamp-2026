# Bank Term Deposit Subscription Predictor

Machine Learning Zoomcamp 2026 — Midterm Project  
**Author:** Cong Tran (`tqthanhcong`)

---

## 1. Problem Description

Direct marketing campaigns (phone calls) remain one of the most effective ways for retail banking institutions to acquire term deposit customers. However, mass cold-calling is expensive, operationally inefficient, and causes customer annoyance.

**Project Objective:**
Build, optimize, and deploy an end-to-end Machine Learning pipeline that predicts whether a banking client will subscribe to a term deposit (`y` = `yes` / `no`). 

By accurately identifying high-propensity leads:
- The marketing team can prioritize targeted outreach, reducing operational campaign costs.
- The bank avoids annoying low-propensity customers with irrelevant calls.
- Overall conversion rates and return on marketing investment (ROMI) increase significantly.

---

## 2. Dataset Information

- **Dataset Source:** [UCI Machine Learning Repository — Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **Samples:** 4,521 customer records across 16 input features + 1 target variable (`y`).
- **Feature Set:**
  - **Demographics:** `age`, `job`, `marital`, `education`
  - **Financial Status:** `default` (credit in default), `balance` (yearly balance in EUR), `housing` (housing loan), `loan` (personal loan)
  - **Campaign Outreach:** `contact` (communication type), `day`, `month`, `duration` (last contact duration in seconds), `campaign` (number of contacts during current campaign)
  - **Historical Outcome:** `pdays` (days since previous campaign contact), `previous` (number of contacts before this campaign), `poutcome` (outcome of previous marketing campaign: `unknown`, `failure`, `other`, `success`)
  - **Target:** `y` (`yes` = subscribed, `no` = did not subscribe).

---

## 3. Project Structure

```text
midterm-project/
├── data/
│   ├── bank.csv             # Cleaned dataset (4,521 rows)
│   └── bank-full.csv        # Full dataset (45,211 rows)
├── Dockerfile               # Container definition for model serving
├── model.bin                # Serialized model and DictVectorizer
├── notebook.ipynb           # EDA, feature analysis, model tuning & evaluation
├── predict.py               # Flask REST API web service for real-time predictions
├── predict-test.py          # Test script sending client requests to predict.py
├── README.md                # Project documentation and reproduction guide
├── requirements.txt         # Project dependencies
└── train.py                 # Final model training and serialization script
```

---

## 4. Modeling & Performance Summary

We explored three baseline architectures using 60/20/20 train/validation/test splits with stratified sampling:
1. **Logistic Regression:** ROC-AUC ~ 0.8925
2. **Decision Tree Classifier:** ROC-AUC ~ 0.8412
3. **Random Forest Classifier (Tuned):** **ROC-AUC: 0.9021 | Accuracy: 89.61%**

**Key Findings:**
- Contact duration (`duration`), previous campaign outcome (`poutcome=success`), and account balance (`balance`) are the most influential predictive features.
- The tuned Random Forest model (`n_estimators=150`, `max_depth=10`, `min_samples_leaf=3`) delivers the highest discriminative power without overfitting.

---

## 5. How to Run Locally

### Step 1: Clone the repository and navigate to the project directory

```bash
git clone https://github.com/tqthanhcong/ml-zoomcamp-2026.git
cd ml-zoomcamp-2026/midterm-project
```

### Step 2: Set up virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 3: Run the training script

```bash
python3 train.py
```
*Output: trains the model and exports `model.bin`.*

### Step 4: Run the web service

```bash
python3 predict.py
```
*The service will start at `http://0.0.0.0:9696`.*

### Step 5: Test the prediction endpoint in a separate terminal

```bash
python3 predict-test.py
```

Sample output:
```text
[*] Sending test requests to prediction service...

--- Testing: Customer A (High Propensity) ---
Response: {'subscription_probability': 0.7181, 'will_subscribe': True}
Probability: 71.81% | Decision: SUBSCRIBE

--- Testing: Customer B (Low Propensity) ---
Response: {'subscription_probability': 0.0134, 'will_subscribe': False}
Probability: 1.34% | Decision: NO SUBSCRIBE
```

---

## 6. Running with Docker

### Step 1: Build the Docker image

```bash
docker build -t bank-marketing-predict .
```

### Step 2: Run the container

```bash
docker run -it --rm -p 9696:9696 bank-marketing-predict
```

### Step 3: Send test requests to the containerized service

```bash
python3 predict-test.py
```
