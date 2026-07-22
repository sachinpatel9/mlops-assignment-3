import pandas as pd
import mlflow
from pycaret.regression import setup, compare_models

def run_pycaret_automl():
    print("Loading processed dataset for PyCaret AutoML...")
    df = pd.read_csv("data/processed/athletes_automl.csv")
    
    print("Initializing PyCaret Setup...")
    
    # 1. Initialize the local SQLite MLflow backend
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    # 2. Initialize the PyCaret environment
    automl_setup = setup(
        data=df, 
        target='total_lift',
        train_size=0.8,   
        fold=5,              
        session_id=42,       
        log_experiment=True,
        experiment_name='Athlete_AutoML_PyCaret'
    )
    
    print("Running AutoML Model Comparison (5-Minute Limit)...")
    # 3. Compare algorithms
    best_model = compare_models(
        budget_time=5, 
        sort='RMSE'
    )
    
    print("AutoML Run Complete!")
    print(f"Top Model Selected: {best_model}")
    
if __name__ == "__main__":
    run_pycaret_automl()