import os
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from model_train import generate_all_datasets, train_all_models
import time

app = Flask(__name__)

# Create data directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

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
    # Add version parameter to force template reload
    return render_template('knn.html', version=int(time.time()))

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
        weight = float(request.form['x1'])    # Weight in grams
        texture = float(request.form['x2'])   # Texture (0=Smooth, 1=Bumpy)
        
        if not (130 <= weight <= 190):
            return jsonify({'error': 'Weight must be between 130 and 190 grams'}), 400
            
        if texture not in [0, 1]:
            return jsonify({'error': 'Texture must be either Smooth (0) or Bumpy (1)'}), 400
            
        # Load the model
        model = joblib.load('data/knn_model.joblib')
        
        # Standardize input
        X = np.array([[weight, texture]])
        prediction = model.predict(X)[0]
        
        return jsonify({'prediction': f"{prediction:.2f}"})
    except ValueError as e:
        return jsonify({'error': 'Please enter valid numbers'}), 400
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({'error': 'An error occurred while making the prediction'}), 400

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
    # Create datasets and train models
    generate_all_datasets()
    train_all_models()
    
    # Get port from environment variable or use 5000 as default
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
