import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import joblib
import json
import os

print("Training anomaly detection model...")

# Load NASA data
column_names = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3',
                's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10',
                's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19',
                's20', 's21']

df = pd.read_csv(r'C:\Users\tyler\data\train_FD001.txt',
                 sep=r'\s+', header=None,
                 names=column_names, engine='python')

# Selected sensors
selected_sensors = ['s9', 's14', 's4', 's3', 's17', 's7', 's12', 's11', 's2']

# Fit scaler
scaler = MinMaxScaler()
X = scaler.fit_transform(df[selected_sensors])

# Train model
model = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
model.fit(X)

# Save model and scaler
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/isolation_forest.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

# Save model metadata
metadata = {
    'model_name': 'Turbofan Engine Anomaly Detector',
    'version': '1.0.0',
    'algorithm': 'Isolation Forest',
    'features': selected_sensors,
    'contamination': 0.05,
    'n_estimators': 100,
    'training_samples': len(df),
    'description': 'Detects anomalous sensor readings in turbofan jet engines'
}

with open('model/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Model trained on {len(df)} samples")
print(f"Features: {selected_sensors}")
print("Saved: model/isolation_forest.pkl")
print("Saved: model/scaler.pkl")
print("Saved: model/metadata.json")
print("Training complete")