import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
import joblib
import os

# Create data directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

def generate_slr_dataset():
    """Generate Simple Linear Regression dataset for house price prediction."""
    np.random.seed(42)
    house_sizes = np.random.uniform(1000, 5000, 100)  # House sizes in sq ft
    house_prices = 150000 + 100 * house_sizes + np.random.normal(0, 25000, 100)  # House prices
    
    df = pd.DataFrame({
        'house_size_sqft': house_sizes,
        'price': house_prices
    })
    df.to_csv('data/slr_dataset.csv', index=False)
    return df

def generate_mlr_dataset():
    """Generate Multiple Linear Regression dataset for advanced house price prediction."""
    np.random.seed(42)
    n_samples = 100
    square_feet = np.random.uniform(1000, 5000, n_samples)
    bedrooms = np.random.uniform(2, 6, n_samples)
    bathrooms = np.random.uniform(1, 4, n_samples)
    
    house_prices = (
        200000 +  # Base price
        150 * square_feet +  # Price per sq ft
        25000 * bedrooms +   # Price per bedroom
        35000 * bathrooms +  # Price per bathroom
        np.random.normal(0, 15000, n_samples)  # Random variation
    )
    
    df = pd.DataFrame({
        'square_feet': square_feet,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'price': house_prices
    })
    df.to_csv('data/mlr_dataset.csv', index=False)
    return df

def generate_polynomial_dataset():
    """Generate Polynomial Regression dataset for energy consumption prediction."""
    np.random.seed(42)
    temperatures = np.linspace(15, 35, 100)  # Temperature in Celsius
    energy = (
        100 +  # Base consumption
        0.5 * temperatures**2 -  # Quadratic relationship
        2 * temperatures +  # Linear relationship
        np.random.normal(0, 50, 100)  # Random variation
    )
    
    df = pd.DataFrame({
        'temperature': temperatures,
        'energy_consumption': energy
    })
    df.to_csv('data/poly_dataset.csv', index=False)
    return df

def generate_knn_dataset():
    """Generate KNN Regression dataset for student performance prediction."""
    np.random.seed(42)
    n_samples = 200
    
    # Generate study hours (between 1 and 8 hours per week)
    study_hours = np.random.uniform(1, 8, n_samples)
    
    # Generate previous scores (between 40 and 95)
    prev_scores = np.random.uniform(40, 95, n_samples)
    
    # Calculate test scores
    test_scores = (
        50 +  # Base score
        (25 * study_hours / 8) +  # Up to 25 points for study hours
        (25 * (prev_scores - 40) / 55) +  # Up to 25 points for previous performance
        np.random.normal(0, 2, n_samples)  # Small random variation
    )
    
    # Ensure scores are within valid range
    test_scores = np.clip(test_scores, 0, 100)
    
    df = pd.DataFrame({
        'study_hours': study_hours,
        'prev_score': prev_scores,
        'test_score': test_scores
    })
    df.to_csv('data/knn_dataset.csv', index=False)
    return df

def generate_logistic_dataset():
    """Generate Logistic Regression dataset for loan approval prediction."""
    np.random.seed(42)
    n_samples = 100
    
    # Generate features
    annual_income = np.random.uniform(30000, 150000, n_samples)
    credit_score = np.random.uniform(300, 850, n_samples)
    
    # Calculate probability of approval based on features
    z = -6 + (annual_income / 15000) + (credit_score / 100)
    prob_approval = 1 / (1 + np.exp(-z))
    
    # Generate binary outcomes
    approved = (prob_approval > 0.5).astype(int)
    
    df = pd.DataFrame({
        'annual_income': annual_income,
        'credit_score': credit_score,
        'approved': approved
    })
    df.to_csv('data/logistic_dataset.csv', index=False)
    return df

def train_slr_model(df):
    """Train Simple Linear Regression model for house price prediction."""
    X = df['house_size_sqft'].values.reshape(-1, 1)
    y = df['price'].values
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, 'data/slr_model.pkl')
    return model

def train_mlr_model(df):
    """Train Multiple Linear Regression model for house price prediction."""
    X = df[['square_feet', 'bedrooms', 'bathrooms']].values
    y = df['price'].values
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, 'data/mlr_model.pkl')
    return model

def train_polynomial_model(df):
    """Train Polynomial Regression model for energy consumption prediction."""
    X = df['temperature'].values.reshape(-1, 1)
    y = df['energy_consumption'].values
    model = make_pipeline(PolynomialFeatures(2), LinearRegression())
    model.fit(X, y)
    joblib.dump(model, 'data/poly_model.pkl')
    return model

def train_knn_model(df):
    """Train KNN Regression model for student performance prediction."""
    X = df[['study_hours', 'prev_score']].values
    y = df['test_score'].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X_scaled, y)
    
    # Save both scaler and model
    joblib.dump((scaler, model), 'data/knn_model.pkl')
    return scaler, model

def train_logistic_model(df):
    """Train Logistic Regression model for loan approval prediction."""
    X = df[['annual_income', 'credit_score']].values
    y = df['approved'].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = LogisticRegression()
    model.fit(X_scaled, y)
    
    # Save both scaler and model
    joblib.dump((scaler, model), 'data/logistic_model.pkl')
    return scaler, model

def generate_all_datasets():
    """Generate all datasets."""
    print("Generating datasets...")
    generate_slr_dataset()
    generate_mlr_dataset()
    generate_polynomial_dataset()
    generate_knn_dataset()
    generate_logistic_dataset()
    print("All datasets generated successfully!")

def train_all_models():
    """Train all models using their respective datasets."""
    print("Training models...")
    
    # Train SLR model
    df_slr = pd.read_csv('data/slr_dataset.csv')
    train_slr_model(df_slr)
    
    # Train MLR model
    df_mlr = pd.read_csv('data/mlr_dataset.csv')
    train_mlr_model(df_mlr)
    
    # Train Polynomial model
    df_poly = pd.read_csv('data/poly_dataset.csv')
    train_polynomial_model(df_poly)
    
    # Train KNN model
    df_knn = pd.read_csv('data/knn_dataset.csv')
    train_knn_model(df_knn)
    
    # Train Logistic model
    df_log = pd.read_csv('data/logistic_dataset.csv')
    train_logistic_model(df_log)
    
    print("All models trained and saved successfully!")

if __name__ == '__main__':
    generate_all_datasets()
    train_all_models()
