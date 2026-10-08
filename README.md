# Medicine Overdose Detection System

A Flask-based machine learning application for predicting medicine overdose risk using patient data and a hybrid logistic regression + neural network model.

## Overview

This project analyzes patient records and predicts whether a medication combination indicates high overdose risk. The system includes:

- a Flask web application
- patient risk prediction form
- analysis dashboard
- model training pipeline
- synthetic data generation
- patient safety precautions page

## Features

- Predict overdose risk using patient age, medication dosage, and comorbidities
- Hybrid model combining logistic regression and neural network output
- Dashboard summarizing analytics from the dataset
- Validation and synthetic data generation support
- Responsive HTML/CSS UI

## Project Structure

- `app.py` — Flask application and routes
- `train_model.py` — model training logic
- `generate_dataset.py` — synthetic dataset generation
- `sample_overdose_data.csv` — dataset used by the app
- `templates/` — HTML pages
- `static/` — CSS and JS assets
- `backups/` — backup files and earlier versions
- `requirements.txt` — project dependencies

## Tech Stack

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Jinja2
- Bootstrap

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sushmitha-31/MedicineOverdoseDetectionSystem-.git
   cd MedicineOverdoseDetectionSystem-
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:

   Windows:
   ```bash
   venv\Scripts\activate
   ```

   Linux/macOS:
   ```bash
   source venv/bin/activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Run the Application

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000/
```

## Usage

- Log in with any username on the landing page
- Go to the prediction section
- Enter patient details
- View the predicted overdose risk and probability
- Explore the analytics dashboard for patient-risk trends

## Model Notes

The project uses a hybrid architecture where logistic regression probabilities are combined with raw normalized inputs before being processed by a neural network. This design helps capture both linear and nonlinear patterns in overdose risk prediction.

## Notes

- The app uses a synthetic dataset to simulate medical risk patterns.
- The analysis dashboard reads the dataset and presents summary statistics.
- The project is intended for demonstration and learning purposes.

## License

This project is for educational and demonstration use.
