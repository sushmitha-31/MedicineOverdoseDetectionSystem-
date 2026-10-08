print("!!!!! THIS IS THE VERSION I EDITED !!!!!")
from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import pandas as pd
import pickle
import joblib
import logging
from tensorflow.keras.models import load_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = "secret_key_for_session"
app.logger = logger

lr_model = joblib.load("lr_model.pkl")
nn_model = load_model("nn_model.h5")
scaler = joblib.load("scaler.pkl")
le_gender = joblib.load("le_gender.pkl")
le_risk = joblib.load("le_risk.pkl")


# Load trained ML model (expects 10 numeric features)
with open("painkiller_overdose_model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------- LOGIN PAGE ---------------- #
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        if username:
            session["user"] = username
            return redirect(url_for("dashboard"))
        else:
            return render_template("login.html", error="Please enter a username")
    return render_template("login.html")


# ---------------- DASHBOARD ---------------- #
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html", user=session["user"])


# ========================================
# Input Validation & Error Handling
# ========================================
VALID_RANGES = {
    'Age': (0, 120),
    'Paracetamol': (0, 4000),      # Max safe single dose ~4000mg
    'Ibuprofen': (0, 2400),         # Max safe daily ~2400mg
    'Aspirin': (0, 1000),           # Max safe single dose ~650-1000mg
    'Diclofenac': (0, 150),         # Max safe single dose ~150mg
    'Naproxen': (0, 1000)           # Max safe single dose ~500-550mg
}

VALID_GENDERS = ['Male', 'Female']
VALID_COMORBIDITIES = ['Yes', 'No']

def validate_prediction_input(form_data):
    """
    Validate all prediction input fields.
    Returns: (is_valid: bool, features: list|None, error_message: str|None)
    """
    errors = []
    
    # Define required numeric and categorical fields
    numeric_fields = ['Age', 'Paracetamol', 'Ibuprofen', 'Aspirin', 'Diclofenac', 'Naproxen']
    categorical_fields = ['Gender', 'Alzheimers', 'Diabetes', 'Heart_Attack']
    all_required = numeric_fields + categorical_fields
    
    # Check for missing fields
    missing_fields = [f for f in all_required if f not in form_data or not form_data[f].strip()]
    if missing_fields:
        return False, None, f"Missing required fields: {', '.join(missing_fields)}"
    
    # Validate numeric fields
    numeric_values = {}
    for field in numeric_fields:
        try:
            value = float(form_data[field].strip())
            numeric_values[field] = value
            
            # Range validation
            min_val, max_val = VALID_RANGES[field]
            if not (min_val <= value <= max_val):
                errors.append(f"{field} must be between {min_val} and {max_val} (got {value})")
        except ValueError:
            errors.append(f"{field} must be a number (got '{form_data[field]}')")
    
    if errors:
        return False, None, "; ".join(errors)
    
    # Validate categorical fields
    gender = form_data['Gender'].strip()
    if gender not in VALID_GENDERS:
        return False, None, f"Gender must be 'Male' or 'Female' (got '{gender}')"
    
    for field in ['Alzheimers', 'Diabetes', 'Heart_Attack']:
        value = form_data[field].strip()
        if value not in VALID_COMORBIDITIES:
            return False, None, f"{field} must be 'Yes' or 'No' (got '{value}')"
    
    # Build feature array (numeric first, then encoded gender)
    features = [
        numeric_values['Age'],
        gender,  # Will be encoded later
        numeric_values['Paracetamol'],
        numeric_values['Ibuprofen'],
        numeric_values['Aspirin'],
        numeric_values['Diclofenac'],
        numeric_values['Naproxen'],
        1 if form_data['Alzheimers'].strip() == 'Yes' else 0,
        1 if form_data['Diabetes'].strip() == 'Yes' else 0,
        1 if form_data['Heart_Attack'].strip() == 'Yes' else 0
    ]
    
    return True, features, None

# ---------------- PREDICTION ---------------- #
# ---------- PREDICTION PAGE ---------- #
@app.route("/predict", methods=['GET', 'POST'])
def predict():
    risk = None
    prob = None
    error = None
    
    if request.method == 'POST':
        # Validate input
        is_valid, features, error_msg = validate_prediction_input(request.form)
        
        if not is_valid:
            error = error_msg
        else:
            try:
                # Encode gender (features[1] is the gender string)
                features[1] = le_gender.transform([features[1]])[0]
                
                # Scale features using the trained scaler
                features_scaled = scaler.transform([features])
                
                # Get LR probability
                lr_prob = lr_model.predict_proba(features_scaled)[:,1].reshape(-1,1)
                
                # Combine original features + LR probability for hybrid NN input
                nn_input = np.hstack((features_scaled, lr_prob))
                
                # Get final risk probability from NN
                nn_prob = nn_model.predict(nn_input, verbose=0)[0][0]
                
                # Determine risk category
                risk = "High" if nn_prob >= 0.5 else "Low"
                prob = float(nn_prob)
                
                # Log prediction for audit trail
                app.logger.info(f"Prediction made - Risk: {risk}, Probability: {prob:.4f}")
                
            except Exception as e:
                error = "An error occurred during prediction. Please contact support."
                app.logger.error(f"Prediction error: {str(e)}", exc_info=True)

    return render_template("prediction_result.html", risk=risk, prob=prob, error=error)







def _prepare_analysis_dataset_for_model(frame):
    """Encode the CSV columns to the same numeric form used in model training."""
    df = frame.copy()

    if 'Gender' in df.columns:
        df['Gender'] = le_gender.transform(df['Gender'].astype(str).str.strip())

    for col in ['Alzheimers', 'Diabetes', 'Heart_Attack']:
        if col in df.columns and (df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col])):
            df[col] = df[col].astype(str).str.strip().str.lower().eq('yes').astype(int)

    if 'Overdose_Risk' in df.columns:
        df['Overdose_Risk'] = le_risk.transform(df['Overdose_Risk'].astype(str).str.strip())

    return df


# ---------------- ANALYSIS ---------------- #
@app.route("/analysis")
def analysis():
    if "user" not in session:
        return redirect(url_for("login"))

    analysis_error = None

    try:
        data = pd.read_csv("sample_overdose_data.csv")
        total_patients = len(data)
        high_risk = len(data[data['Overdose_Risk'] == 'Yes'])
        low_risk = len(data[data['Overdose_Risk'] == 'No'])
        high_risk_pct = round(100 * high_risk / total_patients, 1)

        avg_age_high = round(data[data['Overdose_Risk'] == 'Yes']['Age'].mean(), 1)
        avg_age_low = round(data[data['Overdose_Risk'] == 'No']['Age'].mean(), 1)
        avg_age_by_risk = {"High Risk": avg_age_high, "Low Risk": avg_age_low}

        gender_high_risk = data[data['Overdose_Risk'] == 'Yes']['Gender'].value_counts().to_dict()
        gender_low_risk = data[data['Overdose_Risk'] == 'No']['Gender'].value_counts().to_dict()

        alzheimers_high = len(data[(data['Overdose_Risk'] == 'Yes') & (data['Alzheimers'] == 'Yes')])
        diabetes_high = len(data[(data['Overdose_Risk'] == 'Yes') & (data['Diabetes'] == 'Yes')])
        heart_attack_high = len(data[(data['Overdose_Risk'] == 'Yes') & (data['Heart_Attack'] == 'Yes')])
        comorbidity_stats = {
            'Alzheimers': round(100 * alzheimers_high / high_risk, 1) if high_risk > 0 else 0,
            'Diabetes': round(100 * diabetes_high / high_risk, 1) if high_risk > 0 else 0,
            'Heart_Attack': round(100 * heart_attack_high / high_risk, 1) if high_risk > 0 else 0
        }

        avg_paracetamol_high = round(data[data['Overdose_Risk'] == 'Yes']['Paracetamol'].mean(), 0)
        avg_paracetamol_low = round(data[data['Overdose_Risk'] == 'No']['Paracetamol'].mean(), 0)
        avg_ibuprofen_high = round(data[data['Overdose_Risk'] == 'Yes']['Ibuprofen'].mean(), 0)
        avg_ibuprofen_low = round(data[data['Overdose_Risk'] == 'No']['Ibuprofen'].mean(), 0)
        avg_aspirin_high = round(data[data['Overdose_Risk'] == 'Yes']['Aspirin'].mean(), 0)
        avg_aspirin_low = round(data[data['Overdose_Risk'] == 'No']['Aspirin'].mean(), 0)
        avg_diclofenac_high = round(data[data['Overdose_Risk'] == 'Yes']['Diclofenac'].mean(), 0)
        avg_diclofenac_low = round(data[data['Overdose_Risk'] == 'No']['Diclofenac'].mean(), 0)
        avg_naproxen_high = round(data[data['Overdose_Risk'] == 'Yes']['Naproxen'].mean(), 0)
        avg_naproxen_low = round(data[data['Overdose_Risk'] == 'No']['Naproxen'].mean(), 0)

        from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

        data_copy = _prepare_analysis_dataset_for_model(data)
        X_train = data_copy.drop(columns=['Overdose_Risk'])
        y_train = data_copy['Overdose_Risk']

        X_train_scaled = scaler.transform(X_train)

        lr_train_probs = lr_model.predict_proba(X_train_scaled)[:, 1]
        lr_train_preds = (lr_train_probs >= 0.5).astype(int)

        lr_probs_for_nn = lr_train_probs.reshape(-1, 1)
        X_train_combined = np.hstack((X_train_scaled, lr_probs_for_nn))
        nn_train_probs = nn_model.predict(X_train_combined, verbose=0).flatten()
        nn_train_preds = (nn_train_probs >= 0.5).astype(int)

        lr_prec, lr_rec, lr_f1, _ = precision_recall_fscore_support(y_train, lr_train_preds, average='binary')
        lr_auc = roc_auc_score(y_train, lr_train_probs)
        nn_prec, nn_rec, nn_f1, _ = precision_recall_fscore_support(y_train, nn_train_preds, average='binary')
        nn_auc = roc_auc_score(y_train, nn_train_probs)

        model_performance = {
            'LR': {'precision': round(lr_prec, 4), 'recall': round(lr_rec, 4), 'f1': round(lr_f1, 4), 'auc': round(lr_auc, 4)},
            'NN': {'precision': round(nn_prec, 4), 'recall': round(nn_rec, 4), 'f1': round(nn_f1, 4), 'auc': round(nn_auc, 4)}
        }

        high_risk_data = data[data['Overdose_Risk'] == 'Yes']
        low_risk_data = data[data['Overdose_Risk'] == 'No']
        feature_importance = {
            'Age': round(high_risk_data['Age'].mean() - low_risk_data['Age'].mean(), 1),
            'Paracetamol': round(high_risk_data['Paracetamol'].mean() - low_risk_data['Paracetamol'].mean(), 0),
            'Ibuprofen': round(high_risk_data['Ibuprofen'].mean() - low_risk_data['Ibuprofen'].mean(), 0),
        }

    except Exception as e:
        logger.exception("Analysis processing failed")
        analysis_error = "The dataset could not be processed. Please check the input file or data format."

        total_patients = 0
        high_risk = 0
        low_risk = 0
        high_risk_pct = 0.0
        avg_age_by_risk = {"High Risk": 0, "Low Risk": 0}
        comorbidity_stats = {'Alzheimers': 0.0, 'Diabetes': 0.0, 'Heart_Attack': 0.0}
        model_performance = {'LR': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auc': 0.0}, 'NN': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auc': 0.0}}
        feature_importance = {'Age': 0.0, 'Paracetamol': 0, 'Ibuprofen': 0}
        gender_high_risk = {}
        gender_low_risk = {}
        avg_paracetamol_high = 0
        avg_paracetamol_low = 0
        avg_ibuprofen_high = 0
        avg_ibuprofen_low = 0
        avg_aspirin_high = 0
        avg_aspirin_low = 0
        avg_diclofenac_high = 0
        avg_diclofenac_low = 0
        avg_naproxen_high = 0
        avg_naproxen_low = 0

    return render_template("analysis.html",
                           total_patients=total_patients,
                           high_risk=high_risk,
                           low_risk=low_risk,
                           high_risk_pct=high_risk_pct,
                           avg_age_by_risk=avg_age_by_risk,
                           comorbidity_stats=comorbidity_stats,
                           model_performance=model_performance,
                           feature_importance=feature_importance,
                           gender_high_risk=gender_high_risk,
                           gender_low_risk=gender_low_risk,
                           avg_paracetamol_high=avg_paracetamol_high,
                           avg_paracetamol_low=avg_paracetamol_low,
                           avg_ibuprofen_high=avg_ibuprofen_high,
                           avg_ibuprofen_low=avg_ibuprofen_low,
                           avg_aspirin_high=avg_aspirin_high,
                           avg_aspirin_low=avg_aspirin_low,
                           avg_diclofenac_high=avg_diclofenac_high,
                           avg_diclofenac_low=avg_diclofenac_low,
                           avg_naproxen_high=avg_naproxen_high,
                           avg_naproxen_low=avg_naproxen_low,
                           analysis_error=analysis_error)

# ---------------- PRECAUTIONS ---------------- #
@app.route("/precautions")
def precautions():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("precautions.html")


# ---------------- LOGOUT ---------------- #
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


# ---------------- RUN APP ---------------- #
if __name__ == "__main__":
    app.run(debug=True)


