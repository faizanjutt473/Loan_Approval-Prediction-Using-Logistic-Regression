from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

# 1. INITIALIZE FASTAPI APP
app = FastAPI(
    title="Logistic Regression API",
    description="Professional API for Loan Approval Prediction using Logistic Regression",
    version="1.0.0"
)

# 2. DEFINE INPUT DATA MODEL
class CustomerData(BaseModel):
    Age: float
    Income: float
    Credit_Score: float
    Loan_Amount: float
    Years_Experience: float

# Global variables for model and scaler
model = None
scaler = None
feature_names = ['Age', 'Income', 'Credit_Score', 'Loan_Amount', 'Years_Experience']

# 3. FUNCTION TO TRAIN AND SAVE MODEL
def train_model():
    global model, scaler
    print("Training model...")

    # Generate Synthetic Data
    X, y = make_classification(
        n_samples=1000,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        weights=[0.7, 0.3],
        random_state=42
    )

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Trained. Accuracy: {acc:.4f}")

    # Save model and scaler to files
    joblib.dump(model, "logistic_model.joblib")
    joblib.dump(scaler, "scaler.joblib")

# 4. LOAD MODEL ON STARTUP
@app.on_event("startup")
def load_model():
    global model, scaler
    if os.path.exists("logistic_model.joblib"):
        model = joblib.load("logistic_model.joblib")
        scaler = joblib.load("scaler.joblib")
        print("Model loaded successfully.")
    else:
        train_model()

# 5. ROOT ENDPOINT
@app.get("/")
def read_root():
    return {"message": "Welcome to Logistic Regression API. Go to /docs for API documentation."}

# 6. PREDICTION ENDPOINT
@app.post("/predict")
def predict_loan_approval(data: CustomerData):
    # Convert input to numpy array
    input_data = np.array([[
        data.Age,
        data.Income,
        data.Credit_Score,
        data.Loan_Amount,
        data.Years_Experience
    ]])

    # Scale the input
    input_scaled = scaler.transform(input_data)

    # Make Prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Return Result
    result = "Approved" if prediction == 1 else "Rejected"

    return {
        "prediction": result,
        "approval_probability": round(float(probability), 4),
        "input_data": data.dict()
    }

# 7. RETRAIN ENDPOINT
@app.post("/retrain")
def retrain_model():
    train_model()
    return {"message": "Model retrained successfully."}