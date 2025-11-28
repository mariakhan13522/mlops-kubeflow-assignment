"""
MLflow Pipeline for California Housing Price Prediction

This pipeline uses MLflow to track experiments and log models.
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import math
import sys
import os

# Force UTF-8 encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')



warnings.filterwarnings('ignore')

# ===========================================================================
# CONFIGURATION
# ===========================================================================

# Set MLflow experiment name
EXPERIMENT_NAME = "California_Housing_ML_Pipeline"
MODEL_NAME = "RandomForest_Housing_Model"

# Model hyperparameters
N_ESTIMATORS = 100
MAX_DEPTH = 10
RANDOM_STATE = 42
TEST_SIZE = 0.2

print("\n" + "=" * 75)
print("MLFLOW PIPELINE - CALIFORNIA HOUSING PRICE PREDICTION")
print("=" * 75)
print(f"Experiment: {EXPERIMENT_NAME}")
print(f"Model: Random Forest Regressor")
print(f"Parameters: n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}")
print("=" * 75)


# ===========================================================================
# STEP 1: DATA EXTRACTION
# ===========================================================================

def extract_data():
    """
    Extract California Housing dataset
    """
    print("\n" + "=" * 75)
    print("STEP 1: DATA EXTRACTION")
    print("=" * 75)
    
    # Load dataset
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['PRICE'] = housing.target
    
    print(f"✓ Dataset loaded successfully")
    print(f"✓ Shape: {df.shape}")
    print(f"✓ Features: {list(df.columns)}")
    
    # Save raw data
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/raw_data.csv', index=False)
    print(f"✓ Raw data saved to: data/raw_data.csv")
    
    return df


# ===========================================================================
# STEP 2: DATA PREPROCESSING
# ===========================================================================

def preprocess_data(df):
    """
    Preprocess data: clean, scale, and split
    """
    print("\n" + "=" * 75)
    print("STEP 2: DATA PREPROCESSING")
    print("=" * 75)
    
    # Check for missing values
    missing = df.isnull().sum().sum()
    print(f"✓ Missing values: {missing}")
    
    # Separate features and target
    X = df.drop('PRICE', axis=1)
    y = df['PRICE']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"✓ Data split: {len(X_train)} train, {len(X_test)} test")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"✓ Features scaled using StandardScaler")
    
    # Log preprocessing parameters to MLflow
    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_param("train_samples", len(X_train))
    mlflow.log_param("test_samples", len(X_test))
    mlflow.log_param("n_features", X_train.shape[1])
    
    print(f"✓ Preprocessing complete")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ===========================================================================
# STEP 3: MODEL TRAINING
# ===========================================================================

def train_model(X_train, y_train):
    """
    Train Random Forest model
    """
    print("\n" + "=" * 75)
    print("STEP 3: MODEL TRAINING")
    print("=" * 75)
    
    # Initialize model
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )
    
    print(f"✓ Training Random Forest...")
    print(f"  - n_estimators: {N_ESTIMATORS}")
    print(f"  - max_depth: {MAX_DEPTH}")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Log model parameters to MLflow
    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_param("max_depth", MAX_DEPTH)
    mlflow.log_param("min_samples_split", 5)
    mlflow.log_param("min_samples_leaf", 2)
    mlflow.log_param("algorithm", "RandomForestRegressor")
    
    print(f"✓ Model training complete")
    
    return model


# ===========================================================================
# STEP 4: MODEL EVALUATION
# ===========================================================================

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model and log metrics to MLflow
    """
    print("\n" + "=" * 75)
    print("STEP 4: MODEL EVALUATION")
    print("=" * 75)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    accuracy = r2 * 100
    
    # Display results
    print("\n" + "=" * 75)
    print("EVALUATION RESULTS")
    print("=" * 75)
    print(f"✓ R² Score: {r2:.4f}")
    print(f"✓ Accuracy (R² as %): {accuracy:.2f}%")
    print(f"✓ Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"✓ Mean Absolute Error (MAE): {mae:.4f}")
    print(f"✓ Mean Squared Error (MSE): {mse:.4f}")
    print("=" * 75)
    print(f"\nInterpretation:")
    print(f"- Model explains {accuracy:.1f}% of variance in house prices")
    print(f"- Average prediction error: ${mae * 100000:,.2f}")
    print(f"- Typical prediction error: ${rmse * 100000:,.2f}")
    print("=" * 75)
    
    # Log metrics to MLflow
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("accuracy_percentage", accuracy)
    
    # Save metrics to file
    metrics = {
        'r2_score': float(r2),
        'rmse': float(rmse),
        'mae': float(mae),
        'mse': float(mse),
        'accuracy_percentage': float(accuracy)
    }
    
    return metrics


# ===========================================================================
# MAIN PIPELINE EXECUTION
# ===========================================================================

def run_pipeline():
    """
    Execute complete ML pipeline with MLflow tracking
    """
    
    # Set MLflow experiment
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Start MLflow run
    with mlflow.start_run(run_name="california_housing_run_1") as run:
        
        print("\n" + "=" * 75)
        print("STARTING MLFLOW PIPELINE")
        print("=" * 75)
        print(f"Run ID: {run.info.run_id}")
        print(f"Experiment ID: {run.info.experiment_id}")
        print("=" * 75)
        
        # Log pipeline information
        mlflow.set_tag("author", "Maria Khan")
        mlflow.set_tag("assignment", "Cloud MLOps #4")
        mlflow.set_tag("dataset", "California Housing")
        mlflow.set_tag("pipeline_version", "1.0")
        
        # Step 1: Extract data
        df = extract_data()
        
        # Step 2: Preprocess data
        X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
        
        # Step 3: Train model
        model = train_model(X_train, y_train)
        
        # Step 4: Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Log model to MLflow
        print("\n✓ Logging model to MLflow...")
        mlflow.sklearn.log_model(
            model, 
            "random_forest_model",
            registered_model_name=MODEL_NAME
        )
        
        # Log scaler as artifact
        import joblib
        scaler_path = "scaler.pkl"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path)
        os.remove(scaler_path)
        
        print(f"✓ Model logged to MLflow")
        print(f"✓ Scaler saved as artifact")
        
        print("\n" + "=" * 75)
        print("PIPELINE EXECUTION COMPLETE!")
        print("=" * 75)
        print(f"\nRun ID: {run.info.run_id}")
        print(f"\nTo view results, run:")
        print(f"  mlflow ui")
        print(f"\nThen open: http://localhost:5000")
        print("=" * 75)
        
        return run.info.run_id, metrics


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    
    print("\n" + "=" * 75)
    print("CALIFORNIA HOUSING ML PIPELINE WITH MLFLOW")
    print("=" * 75)
    print("=" * 75)
    
    try:
        # Run pipeline
        run_id, metrics = run_pipeline()
        
        print("\n" + "=" * 75)
        print("✓✓✓ SUCCESS! ✓✓✓")
        print("=" * 75)
        print(f"\nPipeline executed successfully!")
        print(f"Run ID: {run_id}")
        print(f"\nKey Metrics:")
        print(f"  - R² Score: {metrics['r2_score']:.4f}")
        print(f"  - RMSE: {metrics['rmse']:.4f}")
        print(f"  - MAE: {metrics['mae']:.4f}")
        print(f"  - Accuracy: {metrics['accuracy_percentage']:.2f}%")
        print("\n" + "=" * 75)
        
    except Exception as e:
        print("\n" + "=" * 75)
        print("✗✗✗ ERROR ✗✗✗")
        print("=" * 75)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()