import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models
import joblib

# Load CSV
data = pd.read_csv("sample_overdose_data.csv")

# Encode categorical columns (fit on full data - mapping only, no leakage)
le_gender = LabelEncoder()
data['Gender'] = le_gender.fit_transform(data['Gender'])

# Encode comorbidities to numeric (Yes=1, No=0)
for col in ['Alzheimers', 'Diabetes', 'Heart_Attack']:
    if data[col].dtype == 'object' or data[col].dtype.name == 'string':
        data[col] = (data[col] == 'Yes').astype(int)

# Encode Overdose Risk (fit on full data - mapping only, no leakage)
le_risk = LabelEncoder()
data['Overdose_Risk'] = le_risk.fit_transform(data['Overdose_Risk'])  # 0 = Low, 1 = High

# Features and target
X = data.drop('Overdose_Risk', axis=1)
y = data['Overdose_Risk']

# ========================================
# FIX #1: SPLIT BEFORE SCALING
# ========================================
# Split data FIRST to prevent scaler seeing test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========================================
# FIX #2: FIT SCALER ONLY ON TRAINING DATA
# ========================================
# Critical: Scaler must learn min/max ONLY from training data
# Then apply same transformation to both train and test
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on train only
X_test_scaled = scaler.transform(X_test)        # Transform test using train statistics

# -------------------
# Step 1: Logistic Regression
# -------------------
lr_model = LogisticRegression()
lr_model.fit(X_train_scaled, y_train)

# Save LR model
joblib.dump(lr_model, "lr_model.pkl")

# Get LR probabilities
lr_train_probs = lr_model.predict_proba(X_train_scaled)[:,1].reshape(-1,1)
lr_test_probs = lr_model.predict_proba(X_test_scaled)[:,1].reshape(-1,1)

# Combine original features + LR probability
X_train_combined = np.hstack((X_train_scaled, lr_train_probs))
X_test_combined = np.hstack((X_test_scaled, lr_test_probs))

# -------------------
# Step 2: Neural Network
# -------------------
nn_model = models.Sequential([
    layers.Dense(16, activation='relu', input_shape=(X_train_combined.shape[1],)),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
nn_model.fit(X_train_combined, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=1)

# Save NN model
nn_model.save("nn_model.h5")

# -------------------
# Evaluate
# -------------------
train_acc = nn_model.evaluate(X_train_combined, y_train, verbose=0)[1]
test_acc = nn_model.evaluate(X_test_combined, y_test, verbose=0)[1]

# Get predictions for detailed metrics
y_test_pred_proba = nn_model.predict(X_test_combined, verbose=0).flatten()
y_test_pred = (y_test_pred_proba >= 0.5).astype(int)

# Calculate precision, recall, F1
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average='binary')
roc_auc = roc_auc_score(y_test, y_test_pred_proba)

# Confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()

print("=" * 60)
print("HYBRID MODEL PERFORMANCE (LR→NN) - FIXED DATA LEAKAGE")
print("=" * 60)
print(f"Training Accuracy:  {train_acc:.4f}")
print(f"Testing Accuracy:   {test_acc:.4f}")
print(f"Precision:          {precision:.4f} (True Positive Rate among predictions)")
print(f"Recall:             {recall:.4f} (Catch rate for actual High Risk cases)")
print(f"F1-Score:           {f1:.4f} (Harmonic mean of Precision & Recall)")
print(f"ROC-AUC:            {roc_auc:.4f} (Probability model ranks positives higher)")
print()
print("Confusion Matrix (Test Set):")
print(f"  True Negatives:   {tn}  | False Positives: {fp}")
print(f"  False Negatives:  {fn}  | True Positives:  {tp}")
print()
print("Classification Report:")
print(classification_report(y_test, y_test_pred, target_names=['Low Risk', 'High Risk']))
print("=" * 60)

# Save scaler
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le_gender, "le_gender.pkl")
joblib.dump(le_risk, "le_risk.pkl")
