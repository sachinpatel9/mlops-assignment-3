import pandas as pd
import mlflow
import os
from pycaret.regression import setup, compare_models, pull

def run_reduced_feature_automl():
    print("Loading processed dataset for PyCaret AutoML (Reduced Features)...")
    df = pd.read_csv("data/processed/athletes_automl.csv")
    
    # 1. Isolate the top 3 features identified from the previous run + target
    top_features = ['weight', 'age', 'height', 'total_lift']
    df_reduced = df[top_features]
    
    print("Initializing PyCaret Setup (Top Features Only)...")
    
    # 2. Connect to the existing MLflow tracking database
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    # 3. Initialize PyCaret on the reduced dataset
    automl_setup = setup(
        data=df_reduced, 
        target='total_lift',
        train_size=0.8,      
        fold=5,              
        session_id=42,       
        log_experiment=True, 
        experiment_name='Athlete_AutoML_PyCaret_Reduced_Features' # Separate experiment name
    )
    
    print("Running AutoML Model Comparison (5-Minute Limit)...")
    # 4. Compare algorithms
    best_model = compare_models(
        budget_time=5, 
        sort='RMSE'
    )
    
    print("Saving Reduced Leaderboard...")
    os.makedirs("reports", exist_ok=True)
    leaderboard = pull()
    leaderboard.to_csv("reports/pycaret_leaderboard_top_features.csv", index=True)
    
    print("\n" + "="*50)
    print("DATA FOR README:")
    print("Top 3 Models by Validation Score (RMSE):")
    print(leaderboard[['Model', 'RMSE']].head(3))
    print("\nTop 3 Models by Speed (TT (Sec)):")
    print(leaderboard.sort_values('TT (Sec)')[['Model', 'TT (Sec)']].head(3))
    print("="*50 + "\n")
    
    print("Reduced Feature AutoML Run Complete!")
    print("Leaderboard saved to 'reports/pycaret_leaderboard_top_features.csv'")
    
if __name__ == "__main__":
    run_reduced_feature_automl()