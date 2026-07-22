import pandas as pd
import mlflow
import os
import shutil
from pycaret.regression import setup, compare_models, pull, save_model, plot_model

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
    
    print("Saving Leaderboard and Model Summary...")
    # 4. Capture and save outputs 
    os.makedirs("reports", exist_ok=True)
    
    leaderboard = pull()
    leaderboard.to_csv("reports/pycaret_leaderboard.csv", index=True)
    save_model(best_model, "reports/pycaret_best_model")
    
    print("Generating Feature Importance Plot...")
    # 5. Generate Data Insights / Feature Importance
    try:
        # PyCaret generates and saves the plot as 'Feature Importance.png' in the root directory
        plot_model(best_model, plot='feature', save=True)
        
        # Move the artifact cleanly into the reports folder
        if os.path.exists("Feature Importance.png"):
            shutil.move("Feature Importance.png", "reports/feature_importance.png")
            print("Feature Importance plot saved to 'reports/feature_importance.png'")
    except Exception as e:
        print(f"Warning: Could not generate feature importance plot for {best_model.__class__.__name__}. {e}")
    
    print("AutoML Pipeline Complete!")
    
if __name__ == "__main__":
    run_pycaret_automl()