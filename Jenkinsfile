pipeline {
    agent any
    
    environment {
        // Python executable path - adjust if needed
        PYTHON_HOME = 'C:\\Users\\maria\\AppData\\Local\\Programs\\Python\\Python311'

        PATH = "${PYTHON_HOME};${PYTHON_HOME}\\Scripts;${env.PATH}"
    }
    
    stages {
        stage('Environment Setup') {
            steps {
                echo '=========================================='
                echo 'STAGE 1: ENVIRONMENT SETUP'
                echo '=========================================='
                
                // Display Python version
                echo 'Checking Python installation...'
                bat 'python --version'
                
                // Display pip version
                echo 'Checking pip installation...'
                bat 'pip --version'
                
                // Upgrade pip
                echo 'Upgrading pip...'
                bat 'python -m pip install --upgrade pip'
                
                // Install dependencies from requirements.txt
                echo 'Installing project dependencies...'
                bat 'pip install -r requirements.txt'
                
                echo 'Environment setup completed successfully!'
            }
        }
        
        stage('Pipeline Validation') {
            steps {
                echo '=========================================='
                echo 'STAGE 2: PIPELINE VALIDATION'
                echo '=========================================='
                
                // Validate MLflow pipeline script
                echo 'Validating MLflow pipeline script...'
                bat '''
                    if exist mlflow_pipeline.py (
                        echo [SUCCESS] mlflow_pipeline.py found
                    ) else (
                        echo [ERROR] mlflow_pipeline.py not found!
                        exit 1
                    )
                '''
                
                // Validate pipeline components
                echo 'Validating pipeline components...'
                bat '''
                    if exist src\\pipeline_components.py (
                        echo [SUCCESS] pipeline_components.py found
                    ) else (
                        echo [ERROR] pipeline_components.py not found!
                        exit 1
                    )
                '''
                
                // Test Python imports (syntax check)
                echo 'Testing Python imports and syntax...'
                bat 'python -c "import mlflow; import sklearn; import pandas; print(\'All imports successful!\')"'
                
                echo 'Pipeline validation completed successfully!'
            }
        }
        
        stage('Pipeline Execution Test') {
            steps {
                echo '=========================================='
                echo 'STAGE 3: PIPELINE EXECUTION TEST'
                echo '=========================================='
                
                // Run MLflow pipeline
                echo 'Executing MLflow pipeline...'
                echo 'This will train the model and log to MLflow...'
                bat 'python mlflow_pipeline.py'
                
                // Verify MLflow run directory exists
                echo 'Verifying MLflow outputs...'
                bat '''
                    if exist mlruns (
                        echo [SUCCESS] MLflow runs directory found
                        dir mlruns
                    ) else (
                        echo [ERROR] MLflow runs not created!
                        exit 1
                    )
                '''
                
                // Verify data was created
                echo 'Verifying data outputs...'
                bat '''
                    if exist data\\raw_data.csv (
                        echo [SUCCESS] Data file created
                        for %%I in (data\\raw_data.csv) do echo File size: %%~zI bytes
                    ) else (
                        echo [ERROR] Data file not created!
                        exit 1
                    )
                '''
                
                echo 'Pipeline execution test completed successfully!'
            }
        }
    }
    
    post {
        success {
            echo '=========================================='
            echo 'PIPELINE EXECUTION: SUCCESS'
            echo '=========================================='
            echo 'All stages completed successfully!'
            echo '✓ Stage 1: Environment Setup - PASSED'
            echo '✓ Stage 2: Pipeline Validation - PASSED'
            echo '✓ Stage 3: Pipeline Execution Test - PASSED'
            echo '=========================================='
            echo 'CI/CD Pipeline completed successfully!'
            echo 'MLflow model trained and logged.'
            echo '=========================================='
        }
        failure {
            echo '=========================================='
            echo 'PIPELINE EXECUTION: FAILED'
            echo '=========================================='
            echo 'One or more stages failed.'
            echo 'Please check the console output above.'
            echo '=========================================='
        }
        always {
            echo '=========================================='
            echo 'CLEANUP AND FINAL SUMMARY'
            echo '=========================================='
            echo "Build ID: ${env.BUILD_ID}"
            echo "Build Number: ${env.BUILD_NUMBER}"
            echo "Workspace: ${env.WORKSPACE}"
            echo '=========================================='
        }
    }
}