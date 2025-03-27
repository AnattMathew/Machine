import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
    """Generate KNN dataset for fruit prediction based on weight and texture."""
    # Create the dataset
    data = {
        'Weight': [150, 170, 140, 130, 180, 190, 160, 175, 155, 165, 145, 185],
        'Texture': [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1],
        'Label': [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1]
    }
    
    df = pd.DataFrame(data)
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
    """Train KNN model for fruit prediction."""
    X = df[['Weight', 'Texture']]
    y = df['Label']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    model = make_pipeline(
        StandardScaler(),
        KNeighborsRegressor(
            n_neighbors=3,  # Using smaller k due to smaller dataset
            weights='distance',
            metric='minkowski',
            p=2,
            n_jobs=-1
        )
    )
    model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(model, 'data/knn_model.joblib')
    
    # Calculate and print metrics
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)
    
    print(f"\nKNN Model Performance Metrics:")
    print(f"Train Score: {train_score:.4f}")
    print(f"Test Score: {test_score:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Add some example predictions
    example_data = pd.DataFrame({
        'Weight': [160, 175, 145, 185],
        'Texture': [0, 1, 0, 1]
    })
    example_predictions = model.predict(example_data)
    print("\nExample Predictions:")
    for i, (weight, texture) in enumerate(zip(example_data['Weight'], example_data['Texture'])):
        pred = 1 if example_predictions[i] > 0.5 else 0
        print(f"Weight: {weight}, Texture: {texture} -> Predicted Label: {pred}")
    
    return model

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
