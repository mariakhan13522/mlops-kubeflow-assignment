# MLOps Pipeline - California Housing Price Prediction

**Course**: Cloud MLOps (BS AI)  
**Date**: November 27, 2025

---

## Project Overview

This project implements a complete **MLOps pipeline** for predicting California housing prices using Random Forest regression. The pipeline demonstrates industry-standard MLOps practices including:

- **Data Versioning** with DVC (Data Version Control)
- **Experiment Tracking** with MLflow
- **Continuous Integration** with Jenkins
- **Automated ML Workflow** from data extraction to model evaluation

### Machine Learning Problem

**Problem Statement**: Predict median house prices in California based on 8 features including median income, house age, average rooms, etc.

**Model**: Random Forest Regressor  
**Dataset**: California Housing Dataset (20,640 samples, 8 features)  
**Performance**: R² Score ~81%, RMSE ~0.52, MAE ~0.34

---

## Tools and Technologies

| Category | Tools |
|----------|-------|
| **Version Control** | Git, GitHub, DVC |
| **ML Framework** | Scikit-learn, Pandas, NumPy |
| **Experiment Tracking** | MLflow |
| **CI/CD** | Jenkins |
| **Language** | Python 3.11 |

---

## Project Structure

```
mlops-kubeflow-assignment/
├── data/
│   ├── raw_data.csv.dvc          # DVC tracked dataset
│   └── download_data.py          # Dataset download script
│
├── src/
│   ├── pipeline_components.py   # Kubeflow-style components
│   └── model_training.py        # Training utilities
│
├── mlflow_pipeline.py            # Main MLflow pipeline
├── mlruns/                       # MLflow experiment tracking data
│
├── Jenkinsfile                   # CI/CD pipeline definition
├── requirements.txt              # Python dependencies
│
├── .dvc/                        # DVC configuration
│   └── config                   # DVC remote storage config
│
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

---

## Setup Instructions

### Prerequisites

- Python 3.9+
- Git
- pip

### Step 1: Clone Repository

```bash
git clone https://github.com/mariakhan13522/mlops-kubeflow-assignment.git
cd mlops-kubeflow-assignment
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Dependencies include:
- `mlflow==2.9.2` - Experiment tracking
- `dvc==3.30.0` - Data version control
- `scikit-learn==1.3.0` - Machine learning
- `pandas==2.0.3` - Data manipulation
- `numpy==1.24.3` - Numerical computing

### Step 3: Setup DVC Remote Storage

```bash
# Initialize DVC
dvc init

# Configure remote storage (example: local)
dvc remote add -d myremote /path/to/storage

# Pull data
dvc pull
```

---

## Running the ML Pipeline

### Execute Pipeline

```bash
python mlflow_pipeline.py
```

This will:
1. Extract California Housing dataset
2. Preprocess data (scale features, split 80/20)
3. Train Random Forest model (100 trees)
4. Evaluate model and log metrics to MLflow

### View Results in MLflow UI

```bash
mlflow ui
```

Then open: **http://localhost:5000**

You'll see:
- Experiment tracking
- Model parameters (n_estimators, max_depth, etc.)
- Performance metrics (R², RMSE, MAE)
- Saved model artifacts

---

## CI/CD Pipeline with Jenkins

### Jenkins Pipeline Stages

The Jenkinsfile defines a 3-stage CI/CD pipeline:

**Stage 1: Environment Setup**
- Checkout code from GitHub
- Install Python dependencies
- Verify installation

**Stage 2: Pipeline Validation**
- Validate Python scripts exist
- Test imports and syntax
- Check for errors

**Stage 3: Pipeline Execution Test**
- Run MLflow pipeline
- Verify outputs created
- Log results

### Trigger Jenkins Build

```bash
# Push to GitHub triggers Jenkins (if webhook configured)
git push origin main

# Or manually trigger in Jenkins UI
```

---

## Pipeline Results

### Model Performance

| Metric | Value |
|--------|-------|
| **R² Score** | 0.8123 (81.23%) |
| **RMSE** | 0.5234 |
| **MAE** | 0.3456 |
| **Training Samples** | 16,512 |
| **Test Samples** | 4,128 |

### Interpretation

- Model explains **81.2%** of variance in house prices
- Average prediction error: **$34,560**
- Typical prediction error: **$52,340**

### Dataset Features

1. **MedInc**: Median income in block
2. **HouseAge**: Median house age
3. **AveRooms**: Average rooms per household
4. **AveBedrms**: Average bedrooms per household
5. **Population**: Block population
6. **AveOccup**: Average house occupancy
7. **Latitude**: Block latitude
8. **Longitude**: Block longitude

**Target**: Median house value (in $100,000s)

---

## MLOps Practices Implemented

### 1. Data Versioning (DVC)
- Track large datasets without storing in Git
- Version data alongside code
- Reproducible data pipelines

### 2. Experiment Tracking (MLflow)
- Log parameters, metrics, and artifacts
- Compare multiple runs
- Model versioning and registry
- Reproducible experiments

### 3. Continuous Integration (Jenkins)
- Automated testing on code changes
- Validate pipeline before deployment
- Ensure code quality
- Fast feedback loop

### 4. Version Control (Git/GitHub)
- Track code changes
- Collaborative development
- Code review process
- Branch management

---

## Development Workflow

```
1. Data scientist modifies pipeline
   ↓
2. Commit changes to GitHub
   ↓
3. Jenkins automatically triggers
   ↓
4. Jenkins runs 3-stage validation
   ↓
5. If successful, pipeline is validated
   ↓
6. MLflow tracks experiment results
   ↓
7. Model ready for deployment
```

---

## Troubleshooting

### Issue: MLflow UI not starting

```bash
# Kill existing MLflow processes
taskkill /F /IM gunicorn.exe
# Start fresh
mlflow ui
```

### Issue: Jenkins build fails at Stage 1

```bash
# Check Python path in Jenkinsfile
# Update PYTHON_HOME variable
```

### Issue: DVC pull fails

```bash
# Check remote storage configuration
dvc remote list
dvc remote modify myremote url /new/path
```

---

## Learning Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DVC Documentation](https://dvc.org/doc)
- [Jenkins Pipeline Syntax](https://www.jenkins.io/doc/book/pipeline/syntax/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

---

## Assignment Tasks Completed

### Task 1: Project Initialization and Data Versioning 
- Created GitHub repository
- Initialized DVC
- Tracked California Housing dataset
- Setup remote storage

### Task 2: Building Pipeline Components 
- Created 4 reusable components (data extraction, preprocessing, training, evaluation)
- Compiled components to YAML files
- Documented component inputs/outputs

### Task 3: Pipeline Orchestration with MLflow 
- Implemented complete ML pipeline
- Logged experiments to MLflow
- Tracked parameters and metrics
- Saved model artifacts
- **Note**: Used MLflow instead of Kubeflow due to deployment issues

### Task 4: Continuous Integration with Jenkins 
- Created Jenkinsfile with 3 stages
- Configured Jenkins Pipeline job
- Linked to GitHub repository
- Successfully executed automated tests

### Task 5: Final Integration and Documentation 
- Comprehensive README.md
- Project documentation
- Setup instructions
- Troubleshooting guide

---

## Author Information

**Name**: Maria Khan  
**Roll Number**: [21i-1352]  
**Section**: [B]  
**Email**: mariakhan13522@gmail.com  
**University**: FAST NUCES Islamabad

---

## License

This project is for educational purposes as part of the Cloud MLOps course assignment.

---

## Acknowledgments

- Course Instructor: Sir Mateen Yaqoob
- FAST NUCES Islamabad
- MLflow Community
- DVC Community

---

**GitHub Repository**: https://github.com/mariakhan13522/mlops-kubeflow-assignment