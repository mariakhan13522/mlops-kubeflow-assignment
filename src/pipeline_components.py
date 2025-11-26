"""
Kubeflow Pipeline Components for California Housing ML Pipeline

This file contains 4 Kubeflow components:
1. data_extraction - Fetches data using DVC
2. data_preprocessing - Cleans, scales, and splits data
3. model_training - Trains Random Forest model
4. model_evaluation - Evaluates model performance
"""

from kfp.dsl import component, Input, Output, Dataset, Model, Metrics
from typing import NamedTuple

# ===========================================================================
# COMPONENT 1: DATA EXTRACTION (Using DVC)
# ===========================================================================

@component(
    base_image="python:3.9",
    packages_to_install=["pandas==2.0.3", "scikit-learn==1.3.0", "numpy==1.24.3", "dvc==3.30.0"]
)
def data_extraction(
    dataset_output: Output[Dataset]
):
    """
    Component 1: Data Extraction
    Fetches the versioned California Housing dataset
    
    This component simulates DVC data fetching by loading the California Housing dataset.
    In a real scenario, it would use 'dvc get' or 'dvc import' to fetch versioned data.
    
    Outputs:
        dataset_output (Output[Dataset]): Raw dataset saved as CSV file
            - Contains 20,640 samples
            - 8 features + 1 target variable (PRICE)
            - Saved to path provided by Kubeflow
    """
    import pandas as pd
    from sklearn.datasets import fetch_california_housing
    import warnings
    warnings.filterwarnings('ignore')
    
    print("=" * 70)
    print("COMPONENT 1: DATA EXTRACTION")
    print("=" * 70)
    print("Fetching California Housing dataset...")
    print("(In production, this would use: dvc get or dvc import)")
    
    # Load California Housing dataset
    # NOTE: In real scenario, you would use:
    # subprocess.run(["dvc", "get", ".", "data/raw_data.csv", "-o", dataset_output.path])
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['PRICE'] = housing.target
    
    # Save extracted data
    df.to_csv(dataset_output.path, index=False)
    
    print(f"\n✓ Data extraction completed!")
    print(f"✓ Dataset shape: {df.shape}")
    print(f"✓ Total samples: {df.shape[0]:,}")
    print(f"✓ Total features: {df.shape[1] - 1}")
    print(f"✓ Features: {list(df.columns)}")
    print(f"✓ Target variable: PRICE (median house value)")
    print(f"✓ Data saved to: {dataset_output.path}")
    print("=" * 70)


# ===========================================================================
# COMPONENT 2: DATA PREPROCESSING
# ===========================================================================

@component(
    base_image="python:3.9",
    packages_to_install=["pandas==2.0.3", "scikit-learn==1.3.0", "numpy==1.24.3"]
)
def data_preprocessing(
    dataset_input: Input[Dataset],
    train_data: Output[Dataset],
    test_data: Output[Dataset]
) -> NamedTuple('PreprocessingOutputs', [
    ('train_size', int), 
    ('test_size', int),
    ('n_features', int)
]):
    """
    Component 2: Data Preprocessing
    Handles cleaning, scaling, and splitting data into train/test sets
    
    Inputs:
        dataset_input (Input[Dataset]): Raw dataset from extraction component
            - CSV file with features and target variable
    
    Outputs:
        train_data (Output[Dataset]): Preprocessed training dataset
            - 80% of total data (~16,512 samples)
            - Features scaled using StandardScaler
            - Saved as CSV file
        
        test_data (Output[Dataset]): Preprocessed test dataset
            - 20% of total data (~4,128 samples)
            - Features scaled using StandardScaler
            - Saved as CSV file
        
        train_size (int): Number of training samples
        test_size (int): Number of test samples
        n_features (int): Number of features
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from collections import namedtuple
    
    print("=" * 70)
    print("COMPONENT 2: DATA PREPROCESSING")
    print("=" * 70)
    
    # Load data
    df = pd.read_csv(dataset_input.path)
    print(f"✓ Loaded dataset: {df.shape}")
    
    # Check for missing values
    missing = df.isnull().sum().sum()
    print(f"✓ Missing values found: {missing}")
    if missing > 0:
        print("  Handling missing values...")
        df = df.dropna()
    
    # Separate features and target
    X = df.drop('PRICE', axis=1)
    y = df['PRICE']
    n_features = X.shape[1]
    
    print(f"✓ Features shape: {X.shape}")
    print(f"✓ Target shape: {y.shape}")
    
    # Split data into train/test (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        shuffle=True
    )
    print(f"\n✓ Data split completed:")
    print(f"  - Training samples: {len(X_train):,} (80%)")
    print(f"  - Test samples: {len(X_test):,} (20%)")
    
    # Scale features using StandardScaler
    print(f"\n✓ Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"  - Mean: ~0, Std: ~1 (standardized)")
    
    # Create DataFrames with scaled data
    train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_df['PRICE'] = y_train.values
    
    test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_df['PRICE'] = y_test.values
    
    # Save processed data
    train_df.to_csv(train_data.path, index=False)
    test_df.to_csv(test_data.path, index=False)
    
    print(f"\n✓ Preprocessed data saved:")
    print(f"  - Training data: {train_data.path}")
    print(f"  - Test data: {test_data.path}")
    print("=" * 70)
    
    # Return output tuple
    outputs = namedtuple('PreprocessingOutputs', ['train_size', 'test_size', 'n_features'])
    return outputs(len(train_df), len(test_df), n_features)


# ===========================================================================
# COMPONENT 3: MODEL TRAINING
# ===========================================================================

@component(
    base_image="python:3.9",
    packages_to_install=["pandas==2.0.3", "scikit-learn==1.3.0", "joblib==1.3.0"]
)
def model_training(
    train_data: Input[Dataset],
    model_output: Output[Model],
    n_estimators: int = 100,
    max_depth: int = 10
) -> NamedTuple('TrainingOutputs', [
    ('model_path', str), 
    ('training_samples', int),
    ('n_trees', int)
]):
    """
    Component 3: Model Training
    Trains a Random Forest classifier on the training data and saves model artifact
    
    Inputs:
        train_data (Input[Dataset]): Preprocessed training dataset
            - CSV file with scaled features and target
            - Contains ~16,512 samples
        
        n_estimators (int): Number of trees in Random Forest (default: 100)
        max_depth (int): Maximum depth of trees (default: 10)
    
    Outputs:
        model_output (Output[Model]): Trained Random Forest model
            - Saved as .joblib file
            - Contains 100 decision trees
            - Ready for evaluation/deployment
        
        model_path (str): File path where model is saved
        training_samples (int): Number of samples used for training
        n_trees (int): Number of trees in the trained model
    
    Algorithm Details:
        - Model: Random Forest Regressor
        - n_estimators: 100 trees
        - max_depth: 10 levels per tree
        - min_samples_split: 5
        - min_samples_leaf: 2
        - random_state: 42 (for reproducibility)
    """
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    import joblib
    from collections import namedtuple
    
    print("=" * 70)
    print("COMPONENT 3: MODEL TRAINING")
    print("=" * 70)
    
    # Load training data
    train_df = pd.read_csv(train_data.path)
    X_train = train_df.drop('PRICE', axis=1)
    y_train = train_df['PRICE']
    
    print(f"✓ Training data loaded:")
    print(f"  - Samples: {X_train.shape[0]:,}")
    print(f"  - Features: {X_train.shape[1]}")
    print(f"  - Feature names: {list(X_train.columns)}")
    
    # Initialize Random Forest model
    print(f"\n✓ Initializing Random Forest Regressor...")
    print(f"  - Number of trees (n_estimators): {n_estimators}")
    print(f"  - Max depth: {max_depth}")
    print(f"  - Min samples split: 5")
    print(f"  - Min samples leaf: 2")
    print(f"  - Random state: 42")
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    # Train the model
    print(f"\n✓ Training model...")
    print(f"  (This may take 1-2 minutes)")
    model.fit(X_train, y_train)
    
    # Save model artifact
    model_file = model_output.path + '.joblib'
    joblib.dump(model, model_file)
    
    print(f"\n✓ Model training completed!")
    print(f"✓ Model saved to: {model_file}")
    print(f"✓ Model ready for evaluation")
    print("=" * 70)
    
    # Return outputs
    outputs = namedtuple('TrainingOutputs', ['model_path', 'training_samples', 'n_trees'])
    return outputs(model_file, len(X_train), n_estimators)


# ===========================================================================
# COMPONENT 4: MODEL EVALUATION
# ===========================================================================

@component(
    base_image="python:3.9",
    packages_to_install=["pandas==2.0.3", "scikit-learn==1.3.0", "joblib==1.3.0"]
)
def model_evaluation(
    model_input: Input[Model],
    test_data: Input[Dataset],
    metrics_output: Output[Metrics]
) -> NamedTuple('EvaluationOutputs', [
    ('r2_score', float), 
    ('rmse', float), 
    ('mae', float),
    ('accuracy_percentage', float)
]):
    """
    Component 4: Model Evaluation
    Loads trained model, evaluates on test set, and saves metrics
    
    Inputs:
        model_input (Input[Model]): Trained model from training component
            - Random Forest model in .joblib format
        
        test_data (Input[Dataset]): Preprocessed test dataset
            - CSV file with scaled features and target
            - Contains ~4,128 samples
    
    Outputs:
        metrics_output (Output[Metrics]): Evaluation metrics file
            - JSON file containing all metrics
            - Includes: R², RMSE, MAE, MSE, accuracy%
        
        r2_score (float): R-squared score (0 to 1)
            - Measures how well model explains variance
            - Higher is better (1.0 = perfect)
        
        rmse (float): Root Mean Squared Error
            - Average prediction error
            - Lower is better
        
        mae (float): Mean Absolute Error
            - Average absolute prediction error
            - Lower is better
        
        accuracy_percentage (float): R² score as percentage
    
    Metrics Saved to File:
        - r2_score: Coefficient of determination
        - rmse: Root Mean Squared Error
        - mae: Mean Absolute Error
        - mse: Mean Squared Error
        - accuracy_percentage: R² as percentage
        - test_samples: Number of test samples
    """
    import pandas as pd
    import joblib
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    import json
    from collections import namedtuple
    import math
    
    print("=" * 70)
    print("COMPONENT 4: MODEL EVALUATION")
    print("=" * 70)
    
    # Load model
    model_file = model_input.path + '.joblib'
    model = joblib.load(model_file)
    print(f"✓ Model loaded from: {model_file}")
    
    # Load test data
    test_df = pd.read_csv(test_data.path)
    X_test = test_df.drop('PRICE', axis=1)
    y_test = test_df['PRICE']
    
    print(f"✓ Test data loaded:")
    print(f"  - Samples: {X_test.shape[0]:,}")
    print(f"  - Features: {X_test.shape[1]}")
    
    # Make predictions
    print(f"\n✓ Generating predictions...")
    y_pred = model.predict(X_test)
    print(f"  - Predictions generated for {len(y_pred):,} samples")
    
    # Calculate evaluation metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    accuracy = r2 * 100
    
    # Display results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"✓ R² Score: {r2:.4f}")
    print(f"✓ Accuracy (R² as %): {accuracy:.2f}%")
    print(f"✓ Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"✓ Mean Absolute Error (MAE): {mae:.4f}")
    print(f"✓ Mean Squared Error (MSE): {mse:.4f}")
    print("=" * 70)
    print("\nInterpretation:")
    print(f"- Model explains {accuracy:.1f}% of variance in house prices")
    print(f"- Average prediction error: ${mae * 100000:,.2f}")
    print(f"- Typical prediction error: ${rmse * 100000:,.2f}")
    print("=" * 70)
    
    # Save metrics to file
    metrics = {
        'r2_score': float(r2),
        'rmse': float(rmse),
        'mae': float(mae),
        'mse': float(mse),
        'accuracy_percentage': float(accuracy),
        'test_samples': len(y_test)
    }
    
    with open(metrics_output.path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Metrics saved to: {metrics_output.path}")
    
    # Return outputs
    outputs = namedtuple('EvaluationOutputs', ['r2_score', 'rmse', 'mae', 'accuracy_percentage'])
    return outputs(r2, rmse, mae, accuracy)


# ===========================================================================
# MAIN - For Testing Components
# ===========================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("KUBEFLOW PIPELINE COMPONENTS")
    print("=" * 70)
    print("Author: Maria Khan")
    print("Assignment: Cloud MLOps #4")
    print("=" * 70)
    print("\n✓ Components defined successfully!")
    print("\nAvailable Components:")
    print("1. data_extraction")
    print("   - Fetches California Housing dataset (simulates DVC)")
    print("   - Output: Raw dataset (20,640 samples)")
    print("\n2. data_preprocessing")
    print("   - Cleans, scales, and splits data")
    print("   - Outputs: Training data (80%), Test data (20%)")
    print("\n3. model_training")
    print("   - Trains Random Forest Regressor (100 trees)")
    print("   - Output: Trained model (.joblib file)")
    print("\n4. model_evaluation")
    print("   - Evaluates model on test data")
    print("   - Outputs: R², RMSE, MAE metrics")
    print("=" * 70)