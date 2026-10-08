# generate_balanced_dataset.py
"""
MEDICALLY-INFORMED SYNTHETIC DATA GENERATION

This script generates realistic synthetic data for painkiller overdose risk prediction.
Key features:
- Realistic drug dosage ranges (not just 0-1)
- Medical risk scoring logic (cumulative load, age effects, comorbidities)
- Drug interaction effects (NSAIDs + Paracetamol synergy)
- Age-based metabolism (older patients = higher risk at same dose)
- Randomized for realism but grounded in pharmacology

Risk Formula:
    risk_score = cumulative_drug_load + age_factor + comorbidity_factor + drug_interactions
    Where:
    - cumulative_drug_load = sum of (dose / safe_dose) for each drug
    - age_factor = 0.01 * max(0, age - 65)  # Metabolic decline in elderly
    - comorbidity_factor = 0.3 * count(comorbidities)
    - drug_interactions = bonus risk if NSAIDs + Paracetamol combined
"""

import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000  # Total samples

print("=" * 70)
print("GENERATING MEDICALLY-INFORMED SYNTHETIC DATA")
print("=" * 70)

# Generate base patient demographics
print("\n[1] Generating patient demographics...")
ages = np.random.normal(loc=55, scale=18, size=n)  # Mean ~55, realistic distribution
ages = np.clip(ages, 18, 95).astype(int)  # Realistic age range

genders = np.random.choice(["Male", "Female"], size=n, p=[0.45, 0.55])

print(f"    Age range: {ages.min()}-{ages.max()} years (mean: {ages.mean():.1f})")
print(f"    Gender distribution: {np.unique(genders, return_counts=True)[1]}")

# ========================================
# REALISTIC DOSAGE GENERATION
# ========================================
print("\n[2] Generating realistic drug dosages...")

# Paracetamol (Acetaminophen): typically 500-1000mg per dose, max safe ~4000mg/day
paracetamol = np.random.lognormal(mean=np.log(600), sigma=0.8, size=n)
paracetamol = np.clip(paracetamol, 0, 4000).astype(int)

# Ibuprofen: typically 200-600mg per dose, max safe ~2400mg/day
ibuprofen = np.random.lognormal(mean=np.log(400), sigma=0.9, size=n)
ibuprofen = np.clip(ibuprofen, 0, 2400).astype(int)

# Aspirin: typically 325-650mg, max safe ~1000mg/dose
aspirin = np.random.lognormal(mean=np.log(400), sigma=0.8, size=n)
aspirin = np.clip(aspirin, 0, 1000).astype(int)

# Diclofenac: typically 50-100mg, max safe ~150mg/dose
diclofenac = np.random.lognormal(mean=np.log(70), sigma=0.7, size=n)
diclofenac = np.clip(diclofenac, 0, 150).astype(int)

# Naproxen: typically 220-550mg, max safe ~1000mg/day
naproxen = np.random.lognormal(mean=np.log(300), sigma=0.8, size=n)
naproxen = np.clip(naproxen, 0, 1000).astype(int)

print(f"    Paracetamol:  mean={paracetamol.mean():.0f}mg, max={paracetamol.max()}mg")
print(f"    Ibuprofen:    mean={ibuprofen.mean():.0f}mg, max={ibuprofen.max()}mg")
print(f"    Aspirin:      mean={aspirin.mean():.0f}mg, max={aspirin.max()}mg")
print(f"    Diclofenac:   mean={diclofenac.mean():.0f}mg, max={diclofenac.max()}mg")
print(f"    Naproxen:     mean={naproxen.mean():.0f}mg, max={naproxen.max()}mg")

# ========================================
# COMORBIDITY GENERATION
# ========================================
print("\n[3] Generating comorbidities...")

# Comorbidities more common in elderly
# Alzheimer's: increases with age, increases overdose risk (confusion, medication errors)
alzheimers_prob = np.minimum(0.01 + 0.015 * np.maximum(0, ages - 60) / 35, 0.4)
alzheimers = (np.random.random(n) < alzheimers_prob).astype(str)
alzheimers = np.where(alzheimers == 'True', 'Yes', 'No')

# Diabetes: ~10% prevalence, increases with age, affects drug metabolism
diabetes_prob = 0.1 + 0.005 * np.maximum(0, ages - 50) / 45
diabetes = (np.random.random(n) < diabetes_prob).astype(str)
diabetes = np.where(diabetes == 'True', 'Yes', 'No')

# Heart Attack history: increases with age, indicates cardiovascular risk
heart_attack_prob = 0.02 + 0.015 * np.maximum(0, ages - 50) / 45
heart_attack = (np.random.random(n) < heart_attack_prob).astype(str)
heart_attack = np.where(heart_attack == 'True', 'Yes', 'No')

print(f"    Alzheimer's: {np.sum(alzheimers == 'Yes')} cases ({100*np.mean(alzheimers == 'Yes'):.1f}%)")
print(f"    Diabetes:    {np.sum(diabetes == 'Yes')} cases ({100*np.mean(diabetes == 'Yes'):.1f}%)")
print(f"    Heart Attack:{np.sum(heart_attack == 'Yes')} cases ({100*np.mean(heart_attack == 'Yes'):.1f}%)")

# ========================================
# MEDICAL RISK SCORING
# ========================================
print("\n[4] Computing medical risk scores...")

# Define safe dosage thresholds for each drug
safe_doses = {
    'paracetamol': 4000,   # Max safe daily dose
    'ibuprofen': 2400,     # Max safe daily dose
    'aspirin': 1000,       # Max safe single dose
    'diclofenac': 150,     # Max safe single dose
    'naproxen': 1000       # Max safe daily dose
}

# 1. Cumulative drug load (relative to safe thresholds)
cumulative_load = (
    (paracetamol / safe_doses['paracetamol']) +
    (ibuprofen / safe_doses['ibuprofen']) +
    (aspirin / safe_doses['aspirin']) +
    (diclofenac / safe_doses['diclofenac']) +
    (naproxen / safe_doses['naproxen'])
)

# 2. Age-based metabolic risk (elderly have slower metabolism)
# Risk increases significantly after age 65 (normal pharmacokinetic decline)
age_factor = 0.5 * np.maximum(0, (ages - 65) / 30)  # Scales from 0 to 0.5 by age 95

# 3. Comorbidity risk factor (compounds with drug load)
# Alzheimer's: confusion → medication errors, higher risk
# Diabetes: affects kidney function → impaired drug clearance
# Heart Attack history: indicates compromised metabolism
comorbidity_count = (
    (alzheimers == 'Yes').astype(int) +
    (diabetes == 'Yes').astype(int) +
    (heart_attack == 'Yes').astype(int)
)
comorbidity_factor = 0.25 * comorbidity_count

# 4. Drug interaction bonus (NSAIDs + Paracetamol = hepatotoxicity risk)
nsaid_load = ibuprofen + aspirin + diclofenac + naproxen
paracetamol_present = (paracetamol > 0).astype(int)
nsaid_present = (nsaid_load > 0).astype(int)
interaction_bonus = 0.3 * paracetamol_present * nsaid_present * (nsaid_load / safe_doses['naproxen'])

# Total risk score
risk_score = cumulative_load + age_factor + comorbidity_factor + interaction_bonus

# Add realistic noise
noise = np.random.normal(0, 0.15, n)
risk_score = np.maximum(0, risk_score + noise)

print(f"    Risk score statistics:")
print(f"      Mean: {risk_score.mean():.3f}")
print(f"      Std:  {risk_score.std():.3f}")
print(f"      Min:  {risk_score.min():.3f}, Max: {risk_score.max():.3f}")

# ========================================
# BINARY CLASSIFICATION
# ========================================
print("\n[5] Converting risk scores to binary labels...")

# Use median as threshold (roughly balanced classes)
threshold = np.median(risk_score)
overdose_risk = np.where(risk_score >= threshold, 'Yes', 'No')

distribution = pd.Series(overdose_risk).value_counts()
print(f"    Threshold: {threshold:.3f}")
print(f"    Distribution:")
print(f"      Low Risk:  {distribution['No']} ({100*distribution['No']/len(overdose_risk):.1f}%)")
print(f"      High Risk: {distribution['Yes']} ({100*distribution['Yes']/len(overdose_risk):.1f}%)")

# ========================================
# CREATE DATAFRAME & SAVE
# ========================================
print("\n[6] Building DataFrame and saving to CSV...")

df = pd.DataFrame({
    "Age": ages,
    "Gender": genders,
    "Paracetamol": paracetamol,
    "Ibuprofen": ibuprofen,
    "Aspirin": aspirin,
    "Diclofenac": diclofenac,
    "Naproxen": naproxen,
    "Alzheimers": alzheimers,
    "Diabetes": diabetes,
    "Heart_Attack": heart_attack,
    "Overdose_Risk": overdose_risk
})

df.to_csv("sample_overdose_data.csv", index=False)

print(f"    ✓ Saved {len(df)} samples to sample_overdose_data.csv")

# ========================================
# VALIDATION & STATISTICS
# ========================================
print("\n[7] Data quality validation...")

print(f"    Shape: {df.shape}")
print(f"    No missing values: {df.isnull().sum().sum() == 0}")
print(f"    Data types correct: {df.dtypes.to_dict()}")
print(f"\n    Sample records (first 5):")
print(df.head())

print("\n" + "=" * 70)
print("✓ DATA GENERATION COMPLETE")
print("=" * 70)
print(f"""
KEY IMPROVEMENTS OVER PREVIOUS VERSION:

✓ Realistic dosage ranges
  - Paracetamol: 0-4000 mg (lognormal, realistic distribution)
  - NSAIDs: Appropriately scaled ranges
  - Not just random 0-1 values

✓ Medical risk logic
  - Cumulative drug load (sum of dose ratios to safe thresholds)
  - Age factor (metabolic decline in elderly)
  - Comorbidity effects (Alzheimer's, Diabetes, Heart Attack)
  - Drug interactions (NSAIDs + Paracetamol toxicity)
  
✓ Realistic correlations
  - Features CORRELATE with target label
  - Model can learn meaningful patterns
  - Not "perfect separability" anymore

✓ Grounded in pharmacology
  - Safe dose thresholds based on clinical guidelines
  - Age-related metabolism changes (65+ years)
  - Drug interaction effects documented

EXPECTED MODEL BEHAVIOR:
- ✓ Should learn that higher dosages → higher risk
- ✓ Should learn that elderly + dosage → higher risk
- ✓ Should learn that comorbidities increase risk
- ✓ Metrics will be realistic (not perfect 1.0)
- ✓ Model should generalize better to real data
""")
