"""
FIXED TRAINING SCRIPT - Data Leakage Prevention
This script demonstrates the fix for data leakage in the training pipeline.

KEY FIX:
Before: scaler.fit_transform(X) on ALL data -> train_test_split
After:  train_test_split(X) FIRST -> scaler.fit(X_train) -> transform(X_train, X_test)

Impact: Prevents information leakage from test set into training via scaling parameters.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_fscore_support, 
    roc_auc_score, 
    confusion_matrix, 
    classification_report
)
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("DATA LEAKAGE FIX - TRAINING SCRIPT")
print("=" * 70)

# Load CSV
print("\n[1] Loading data...")
data = pd.read_csv("sample_overdose_data.csv")
print(f"    Loaded {len(data)} samples with {data.shape[1]} columns")

# Show class distribution
print("\n[2] Target Distribution (BEFORE SPLITTING):")
print(data["Overdose_Risk"].value_counts())

# Encode categorical columns
print("\n[3] Encoding categorical features...")
le_gender = LabelEncoder()
data['Gender'] = le_gender.fit_transform(data['Gender'])
print(f"    Gender encoded: {dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_)))}")

# Encode comorbidities to numeric (Yes=1, No=0)
for col in ['Alzheimers', 'Diabetes', 'Heart_Attack']:
    if data[col].dtype == 'object' or data[col].dtype.name == 'string':
        data[col] = (data[col] == 'Yes').astype(int)

le_risk = LabelEncoder()
data['Overdose_Risk'] = le_risk.fit_transform(data['Overdose_Risk'])
print(f"    Risk encoded: {dict(zip(le_risk.classes_, le_risk.transform(le_risk.classes_)))}")

# Features and target
X = data.drop('Overdose_Risk', axis=1)
y = data['Overdose_Risk']

# ============================================================================
# CRITICAL FIX: Split BEFORE fitting scaler
# ============================================================================
print("\n[4] SPLITTING DATA (BEFORE scaling - This is the fix!)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"    Training set: {len(X_train)} samples")
print(f"    Test set:     {len(X_test)} samples")
print(f"    Train distribution: {np.bincount(y_train)}")
print(f"    Test distribution:  {np.bincount(y_test)}")

# ============================================================================
# CRITICAL FIX: Fit scaler ONLY on training data, transform both
# ============================================================================
print("\n[5] SCALING (ONLY on training data - This is the fix!)...")
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"    Scaler fit on training data only")
print(f"    Scaler min values: {scaler.data_min_}")
print(f"    Scaler max values: {scaler.data_max_}")

# Save scaler
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le_gender, "le_gender.pkl")
joblib.dump(le_risk, "le_risk.pkl")
print(f"    ✓ Saved: scaler.pkl, le_gender.pkl, le_risk.pkl")

# -------------------
# Step 1: Logistic Regression
# -------------------
print("\n[6] TRAINING LOGISTIC REGRESSION...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
joblib.dump(lr_model, "lr_model.pkl")
print(f"    ✓ Saved: lr_model.pkl")

# Get LR probabilities
lr_train_probs = lr_model.predict_proba(X_train_scaled)[:,1].reshape(-1,1)
lr_test_probs = lr_model.predict_proba(X_test_scaled)[:,1].reshape(-1,1)

# Evaluate LR standalone
y_train_pred_lr = (lr_train_probs >= 0.5).astype(int).flatten()
y_test_pred_lr = (lr_test_probs >= 0.5).astype(int).flatten()

prec_lr, rec_lr, f1_lr, _ = precision_recall_fscore_support(y_test, y_test_pred_lr, average='binary')
auc_lr = roc_auc_score(y_test, lr_test_probs)

print("\n    LR Performance (Test Set):")
print(f"      Precision: {prec_lr:.4f}")
print(f"      Recall:    {rec_lr:.4f}")
print(f"      F1-Score:  {f1_lr:.4f}")
print(f"      ROC-AUC:   {auc_lr:.4f}")

# Combine original features + LR probability for NN
print("\n[7] PREPARING HYBRID INPUT (scaled features + LR probability)...")
X_train_combined = np.hstack((X_train_scaled, lr_train_probs))
X_test_combined = np.hstack((X_test_scaled, lr_test_probs))
print(f"    Combined input shape: {X_train_combined.shape}")

# -------------------
# Step 2: Neural Network (if tensorflow available)
# -------------------
print("\n[8] CHECKING TENSORFLOW AVAILABILITY...")
try:
    from tensorflow.keras import layers, models
    import tensorflow as tf
    
    print("    ✓ TensorFlow available - Training NN...")
    
    nn_model = models.Sequential([
        layers.Dense(16, activation='relu', input_shape=(X_train_combined.shape[1],)),
        layers.Dropout(0.3),
        layers.Dense(8, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    
    nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = nn_model.fit(
        X_train_combined, y_train, 
        epochs=50, 
        batch_size=16, 
        validation_split=0.2, 
        verbose=0
    )
    
    nn_model.save("nn_model.h5")
    print(f"    ✓ Saved: nn_model.h5")
    
    # Get NN predictions
    y_test_pred_nn_proba = nn_model.predict(X_test_combined, verbose=0).flatten()
    y_test_pred_nn = (y_test_pred_nn_proba >= 0.5).astype(int)
    
    # Evaluate NN
    prec_nn, rec_nn, f1_nn, _ = precision_recall_fscore_support(y_test, y_test_pred_nn, average='binary')
    auc_nn = roc_auc_score(y_test, y_test_pred_nn_proba)
    
    print("\n    NN Performance (Test Set):")
    print(f"      Precision: {prec_nn:.4f}")
    print(f"      Recall:    {rec_nn:.4f}")
    print(f"      F1-Score:  {f1_nn:.4f}")
    print(f"      ROC-AUC:   {auc_nn:.4f}")
    
except ImportError as e:
    print(f"    ✗ TensorFlow not available: {e}")
    print(f"    → To install: pip install tensorflow")
    print(f"    → Using LR-only for now")

# -------------------
# FINAL SUMMARY
# -------------------
print("\n" + "=" * 70)
print("FIX SUMMARY")
print("=" * 70)
print("""
BEFORE (LEAKED):
  1. scaler.fit_transform(X)           ← Fit on ALL 1000 samples
  2. X_scaled = result
  3. train_test_split(X_scaled, y)     ← Test set already in scaler params
  
  Result: Test metrics are optimistic because scaler "saw" test data

AFTER (FIXED):
  1. train_test_split(X, y)            ← Split FIRST
  2. scaler.fit(X_train)               ← Fit ONLY on 800 training samples
  3. X_train_scaled = scaler.transform(X_train)
  4. X_test_scaled = scaler.transform(X_test)   ← Transform test with train stats
  
  Result: Realistic test metrics - scaler doesn't have test info
""")

print("\nFILES SAVED:")
print("  • scaler.pkl         (MinMaxScaler fit on training data only)")
print("  • le_gender.pkl      (Gender label encoder)")
print("  • le_risk.pkl        (Risk label encoder)")
print("  • lr_model.pkl       (Logistic Regression)")
print("  • nn_model.h5        (Neural Network - if TensorFlow available)")

print("\n" + "=" * 70)
print("✓ TRAINING COMPLETE")
print("=" * 70)
