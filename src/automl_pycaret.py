import pandas as pd
import mlflow
import os
from pycaret.regression import setup, compare_models, pull, save_model

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
        train_size=0.8,      # 80/20 train-test split
        fold=5,              # 5-fold cross validation
        session_id=42,       # Fixed random seed
        log_experiment=True, # Automatically log to MLflow
        experiment_name='Athlete_AutoML_PyCaret'
    )
    
    print("Running AutoML Model Comparison (5-Minute Limit)...")
    # 3. Compare algorithms
    best_model = compare_models(
        budget_time=5, 
        sort='RMSE'
    )
    
    print("Saving Leaderboard and Model Summary...")
    # 4. Capture and save outputs (Task 3 Requirements)
    os.makedirs("reports", exist_ok=True)
    
    # pull() grabs the active score grid (leaderboard) as a pandas dataframe
    leaderboard = pull()
    leaderboard.to_csv("reports/pycaret_leaderboard.csv", index=True)
    
    # Save the transformation pipeline and best model object
    save_model(best_model, "reports/pycaret_best_model")
    
    print("AutoML Run Complete!")
    print(f"Top Model Selected: {best_model}")
    print("Leaderboard saved to 'reports/pycaret_leaderboard.csv'")
    
if __name__ == "__main__":
    run_pycaret_automl()