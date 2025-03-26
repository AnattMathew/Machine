import os
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from model_train import generate_all_datasets, train_all_models

app = Flask(__name__)

# Create data directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

# Generate datasets and train models on startup
generate_all_datasets()
train_all_models()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/slr')
def slr():
    return render_template('slr.html')

@app.route('/mlr')
def mlr():
    return render_template('mlr.html')

@app.route('/polynomial')
def polynomial():
    return render_template('polynomial.html')

@app.route('/knn')
def knn():
    return render_template('knn.html')

@app.route('/logistic')
def logistic():
    return render_template('logistic.html')

@app.route('/predict_slr', methods=['POST'])
def predict_slr():
    try:
        x = float(request.form['x'])  # House size in sq ft
        model = joblib.load('data/slr_model.pkl')
        prediction = model.predict([[x]])[0]
        return jsonify({'prediction': f"{prediction:,.2f}"})  # Format with commas and 2 decimal places
    except Exception as e:
        return jsonify({'error': 'Could not make prediction'}), 400

@app.route('/predict_mlr', methods=['POST'])
def predict_mlr():
    try:
        square_feet = float(request.form['x1'])  # Square feet
        bedrooms = float(request.form['x2'])     # Number of bedrooms
        bathrooms = float(request.form['x3'])    # Number of bathrooms
        model = joblib.load('data/mlr_model.pkl')
        prediction = model.predict([[square_feet, bedrooms, bathrooms]])[0]
        return jsonify({'prediction': f"{prediction:,.2f}"})
    except Exception as e:
        return jsonify({'error': 'Could not make prediction'}), 400

@app.route('/predict_polynomial', methods=['POST'])
def predict_polynomial():
    try:
        temperature = float(request.form['x'])  # Temperature in Celsius
        model = joblib.load('data/poly_model.pkl')
        prediction = model.predict([[temperature]])[0]
        return jsonify({'prediction': f"{prediction:.2f}"})
    except Exception as e:
        return jsonify({'error': 'Could not make prediction'}), 400

@app.route('/predict_knn', methods=['POST'])
def predict_knn():
    try:
        study_hours = float(request.form['x1'])  # Weekly study hours
        prev_score = float(request.form['x2'])   # Previous exam score
        scaler, model = joblib.load('data/knn_model.pkl')
        X = scaler.transform([[study_hours, prev_score]])
        prediction = model.predict(X)[0]
        return jsonify({'prediction': f"{prediction:.2f}"})
    except Exception as e:
        return jsonify({'error': 'Could not make prediction'}), 400

@app.route('/predict_logistic', methods=['POST'])
def predict_logistic():
    try:
        annual_income = float(request.form['x1'])  # Annual income
        credit_score = float(request.form['x2'])   # Credit score
        scaler, model = joblib.load('data/logistic_model.pkl')
        X = scaler.transform([[annual_income, credit_score]])
        probability = model.predict_proba(X)[0][1]  # Probability of approval
        prediction = 'Approved' if probability >= 0.5 else 'Not Approved'
        return jsonify({
            'prediction': prediction,
            'probability': f"{probability:.2%}"
        })
    except Exception as e:
        return jsonify({'error': 'Could not make prediction'}), 400

if __name__ == '__main__':
    # Use environment variables for host and port
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
