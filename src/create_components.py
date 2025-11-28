# src/create_components.py
"""
Script to compile Kubeflow pipeline components into YAML files
Compatible with KFP 2.0.1
"""

from kfp import compiler
from kfp.dsl import component
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.dirname(__file__))

# We'll write YAML manually since KFP 2.x uses decorators differently
def create_yaml_components():
    """Create YAML files for components"""
    
    components_dir = os.path.join(os.path.dirname(__file__), '..', 'components')
    os.makedirs(components_dir, exist_ok=True)
    
    print("=" * 60)
    print("Creating Kubeflow Pipeline Component YAML Files")
    print("=" * 60)
    
    # Component 1: Data Extraction
    data_extraction_yaml = """name: Data extraction
description: Extracts the Boston Housing dataset
inputs:
outputs:
- {name: dataset_output, type: Dataset, description: 'Raw dataset in CSV format'}
implementation:
  container:
    image: python:3.9
    command:
    - sh
    - -c
    - |
      pip install pandas==2.0.3 scikit-learn==1.3.0 numpy==1.24.3 && python3 -c "
      import pandas as pd
      from sklearn.datasets import fetch_california_housing
      import warnings
      warnings.filterwarnings('ignore')
      
      print('='*50)
      print('STEP 1: DATA EXTRACTION')
      print('='*50)
      
      housing = fetch_california_housing()
      df = pd.DataFrame(housing.data, columns=housing.feature_names)
      df['PRICE'] = housing.target
      
      df.to_csv('$0', index=False)
      
      print(f'✓ Data extracted successfully!')
      print(f'✓ Dataset shape: {df.shape}')
      print(f'✓ Saved to: $0')
      "
    - {outputPath: dataset_output}
"""
    
    # Component 2: Data Preprocessing
    data_preprocessing_yaml = """name: Data preprocessing
description: Cleans, scales, and splits data into train/test sets
inputs:
- {name: dataset_input, type: Dataset, description: 'Raw dataset from extraction'}
outputs:
- {name: train_data, type: Dataset, description: 'Preprocessed training dataset'}
- {name: test_data, type: Dataset, description: 'Preprocessed test dataset'}
- {name: train_size, type: Integer, description: 'Number of training samples'}
- {name: test_size, type: Integer, description: 'Number of test samples'}
implementation:
  container:
    image: python:3.9
    command:
    - sh
    - -c
    - |
      pip install pandas==2.0.3 scikit-learn==1.3.0 numpy==1.24.3 && python3 -c "
      import pandas as pd
      from sklearn.model_selection import train_test_split
      from sklearn.preprocessing import StandardScaler
      
      print('='*50)
      print('STEP 2: DATA PREPROCESSING')
      print('='*50)
      
      df = pd.read_csv('$0')
      print(f'✓ Loaded dataset with shape: {df.shape}')
      
      X = df.drop('PRICE', axis=1)
      y = df['PRICE']
      
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
      print(f'✓ Data split: {len(X_train)} train, {len(X_test)} test')
      
      scaler = StandardScaler()
      X_train_scaled = scaler.fit_transform(X_train)
      X_test_scaled = scaler.transform(X_test)
      
      train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
      train_df['PRICE'] = y_train.values
      
      test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
      test_df['PRICE'] = y_test.values
      
      train_df.to_csv('$1', index=False)
      test_df.to_csv('$2', index=False)
      
      with open('$3', 'w') as f:
          f.write(str(len(train_df)))
      with open('$4', 'w') as f:
          f.write(str(len(test_df)))
      
      print(f'✓ Training data saved')
      print(f'✓ Test data saved')
      "
    - {inputPath: dataset_input}
    - {outputPath: train_data}
    - {outputPath: test_data}
    - {outputPath: train_size}
    - {outputPath: test_size}
"""
    
    # Component 3: Model Training
    model_training_yaml = """name: Model training
description: Trains a Random Forest Regressor on the training data
inputs:
- {name: train_data, type: Dataset, description: 'Preprocessed training dataset'}
- {name: n_estimators, type: Integer, default: '100', description: 'Number of trees'}
outputs:
- {name: model_output, type: Model, description: 'Trained model'}
- {name: model_path, type: String, description: 'Path where model is saved'}
implementation:
  container:
    image: python:3.9
    command:
    - sh
    - -c
    - |
      pip install pandas==2.0.3 scikit-learn==1.3.0 joblib==1.3.0 && python3 -c "
      import pandas as pd
      from sklearn.ensemble import RandomForestRegressor
      import joblib
      
      print('='*50)
      print('STEP 3: MODEL TRAINING')
      print('='*50)
      
      train_df = pd.read_csv('$0')
      X_train = train_df.drop('PRICE', axis=1)
      y_train = train_df['PRICE']
      
      n_trees = int('$1')
      model = RandomForestRegressor(n_estimators=n_trees, max_depth=10, random_state=42, n_jobs=-1)
      
      print(f'✓ Training Random Forest with {n_trees} trees...')
      model.fit(X_train, y_train)
      
      model_file = '$2'
      joblib.dump(model, model_file)
      
      with open('$3', 'w') as f:
          f.write(model_file)
      
      print(f'✓ Model trained and saved!')
      "
    - {inputPath: train_data}
    - {inputValue: n_estimators}
    - {outputPath: model_output}
    - {outputPath: model_path}
"""
    
    # Component 4: Model Evaluation
    model_evaluation_yaml = """name: Model evaluation
description: Evaluates the trained model on test data
inputs:
- {name: model_input, type: Model, description: 'Trained model'}
- {name: test_data, type: Dataset, description: 'Preprocessed test dataset'}
outputs:
- {name: metrics_output, type: Metrics, description: 'Evaluation metrics'}
- {name: r2_score, type: Float, description: 'R-squared score'}
- {name: rmse, type: Float, description: 'Root Mean Squared Error'}
- {name: mae, type: Float, description: 'Mean Absolute Error'}
implementation:
  container:
    image: python:3.9
    command:
    - sh
    - -c
    - |
      pip install pandas==2.0.3 scikit-learn==1.3.0 joblib==1.3.0 && python3 -c "
      import pandas as pd
      import joblib
      from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
      import json
      import math
      
      print('='*50)
      print('STEP 4: MODEL EVALUATION')
      print('='*50)
      
      model = joblib.load('$0')
      test_df = pd.read_csv('$1')
      
      X_test = test_df.drop('PRICE', axis=1)
      y_test = test_df['PRICE']
      
      y_pred = model.predict(X_test)
      
      r2 = r2_score(y_test, y_pred)
      mse = mean_squared_error(y_test, y_pred)
      rmse = math.sqrt(mse)
      mae = mean_absolute_error(y_test, y_pred)
      
      print(f'✓ R² Score: {r2:.4f}')
      print(f'✓ RMSE: {rmse:.4f}')
      print(f'✓ MAE: {mae:.4f}')
      
      metrics = {
          'r2_score': float(r2),
          'rmse': float(rmse),
          'mae': float(mae),
          'mse': float(mse)
      }
      
      with open('$2', 'w') as f:
          json.dump(metrics, f, indent=2)
      
      with open('$3', 'w') as f:
          f.write(str(r2))
      with open('$4', 'w') as f:
          f.write(str(rmse))
      with open('$5', 'w') as f:
          f.write(str(mae))
      "
    - {inputPath: model_input}
    - {inputPath: test_data}
    - {outputPath: metrics_output}
    - {outputPath: r2_score}
    - {outputPath: rmse}
    - {outputPath: mae}
"""
    
    # Write YAML files
    components = {
        'data_extraction.yaml': data_extraction_yaml,
        'data_preprocessing.yaml': data_preprocessing_yaml,
        'model_training.yaml': model_training_yaml,
        'model_evaluation.yaml': model_evaluation_yaml
    }
    
    for filename, content in components.items():
        filepath = os.path.join(components_dir, filename)
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"✓ Created: {filename}")
    
    print("\n" + "=" * 60)
    print("✅ SUCCESS! All 4 component YAML files created")
    print("=" * 60)
    print(f"\nComponents saved in: {os.path.abspath(components_dir)}")
    print("\nCreated files:")
    print("  • data_extraction.yaml")
    print("  • data_preprocessing.yaml")
    print("  • model_training.yaml")
    print("  • model_evaluation.yaml")
    
    return True

if __name__ == "__main__":
    try:
        success = create_yaml_components()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)