import pickle
import pandas as pd
import json
from pathlib import Path

# Load the retrained model
with open('pickle files/randomf.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the scaler
with open('models/minmax_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the threshold
threshold = json.loads(open('models/threshold.json').read())['threshold']

print(f"Model loaded: {type(model)}")
print(f"Features expected: {model.n_features_in_}")
print(f"Threshold: {threshold}")

# Test with low (normal) values
low_risk_input = {
    'sysBP': 110,      # Normal
    'glucose': 85,     # Normal
    'age': 35,         # Young
    'totChol': 180,    # Normal
    'diaBP': 70,       # Normal
    'prevalentHyp': 0, # No hypertension
    'diabetes': 0,     # No diabetes
    'male': 1,         # Male
    'BPMeds': 0,       # No BP meds
    'BMI': 22          # Normal BMI
}

# Create feature vector in correct order
FEATURE_ORDER = ['sysBP', 'glucose', 'age', 'totChol', 'diaBP', 'prevalentHyp', 'diabetes', 'male', 'BPMeds', 'BMI']
x = pd.DataFrame([[low_risk_input[col] for col in FEATURE_ORDER]], columns=FEATURE_ORDER)

print(f"\nInput features: {x.values[0]}")
print(f"Feature order: {FEATURE_ORDER}")

# Scale the input
x_scaled = pd.DataFrame(scaler.transform(x.values), columns=FEATURE_ORDER)
print(f"Scaled features: {x_scaled.values[0]}")

# Get prediction
proba = model.predict_proba(x_scaled)[0][1]
prediction = 1 if proba >= threshold else 0

print(f"\nResults:")
print(f"Probability of heart disease: {proba:.3f}")
print(f"Threshold: {threshold:.3f}")
print(f"Prediction: {'Risk' if prediction == 1 else 'No Risk'}")
print(f"Expected: No Risk (since all values are normal)")

# Test with high risk values
high_risk_input = {
    'sysBP': 170,      # High
    'glucose': 160,    # High
    'age': 65,         # Elderly
    'totChol': 260,    # High
    'diaBP': 100,      # High
    'prevalentHyp': 1, # Has hypertension
    'diabetes': 1,     # Has diabetes
    'male': 1,         # Male
    'BPMeds': 1,       # On BP meds
    'BMI': 32          # High BMI
}

x_high = pd.DataFrame([[high_risk_input[col] for col in FEATURE_ORDER]], columns=FEATURE_ORDER)
x_high_scaled = pd.DataFrame(scaler.transform(x_high.values), columns=FEATURE_ORDER)

proba_high = model.predict_proba(x_high_scaled)[0][1]
prediction_high = 1 if proba_high >= threshold else 0

print(f"\nHigh Risk Test:")
print(f"Probability of heart disease: {proba_high:.3f}")
print(f"Threshold: {threshold:.3f}")
print(f"Prediction: {'Risk' if prediction_high == 1 else 'No Risk'}")
print(f"Expected: Risk (since all values are high)")
