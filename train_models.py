
print("🚀 Creating AI Models...")

import numpy as np
import joblib
import json
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# CREATE FEATURE LIST (This was missing!)
feature_cols = ['amount', 'hour', 'day_of_week', 'velocity_1h', 'is_weekend', 
                'is_night', 'is_rush_hour', 'log_amount', 'user_avg_amount', 
                'user_max_amount', 'user_txn_count', 'amount_deviation', 'is_new_device',
                'location_encoded', 'device_encoded', 'txn_type_encoded']

with open('feature_list.json', 'w') as f:  # ADD THIS LINE
    json.dump(feature_cols, f)              # ADD THIS LINE

print("✅ Feature list saved")

# Rest of the code stays same...
np.random.seed(42)
n_samples = 5000

X = np.random.rand(n_samples, 16)
y = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])

print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
rf_model.fit(X, y)
joblib.dump(rf_model, 'rf_fraud_model.pkl')
print("✅ Random Forest saved")

print("Training Isolation Forest...")
iso_model = IsolationForest(n_estimators=50, contamination=0.05, random_state=42)
iso_model.fit(X)
joblib.dump(iso_model, 'isolation_forest_model.pkl')
print("✅ Isolation Forest saved")

print("Creating scalers...")
scaler = StandardScaler()
scaler.fit(X)
joblib.dump(scaler, 'feature_scaler.pkl')

anomaly_scaler = MinMaxScaler()
anomaly_scaler.fit(np.random.rand(100, 1))
joblib.dump(anomaly_scaler, 'anomaly_scaler.pkl')

print("Creating encoders...")
le_loc = LabelEncoder()
le_loc.fit(['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Pune', 'Kolkata'])
joblib.dump(le_loc, 'location_encoder.pkl')

le_dev = LabelEncoder()
le_dev.fit([f'DEV_{i}' for i in range(100000)])
joblib.dump(le_dev, 'device_encoder.pkl')

le_txn = LabelEncoder()
le_txn.fit(['P2P', 'P2M', 'Bill Pay', 'Recharge'])
joblib.dump(le_txn, 'txn_type_encoder.pkl')

print("Creating user profiles...")
profiles = {}
for i in range(100):
    profiles[f'UPI_{str(i).zfill(6)}'] = {
        'avg_amount': float(np.random.lognormal(6, 1)),
        'max_amount': float(np.random.lognormal(8, 1)),
        'usual_hours': [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'usual_city': np.random.choice(['Mumbai', 'Delhi', 'Bangalore']),
        'device_fingerprint': f'DEV_{np.random.randint(10000, 99999)}'
    }

with open('user_profiles.json', 'w') as f:
    json.dump(profiles, f)

print("✅ User profiles saved")
print("🎉 ALL MODELS CREATED! Now run: python app.py")