import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import os

def run_h2o_automl():
    print("Initializing local H2O Cluster...")
    # Start the H2O Java virtual machine
    h2o.init()
    
    print("Loading processed dataset into H2O...")
    df_path = "data/processed/athletes_automl.csv"
    hf = h2o.import_file(df_path)
    
    # 1. Define target and predictors
    y = 'total_lift'
    x = hf.columns
    x.remove(y)
    
    # 2. Train/Test split for baseline parity (80/20)
    train, test = hf.split_frame(ratios=[0.8], seed=42)
    
    print("Initializing H2O AutoML Setup (5-Minute Limit)...")
    aml = H2OAutoML(
        max_runtime_secs=300, 
        seed=42,
        sort_metric='RMSE',
        project_name='Athlete_AutoML_H2O'
    )
    
    print("Running H2O AutoML...")
    aml.train(x=x, y=y, training_frame=train)
    
    print("Extracting Leaderboard...")
    os.makedirs("reports", exist_ok=True)
    
    # Extract Leaderboard
    lb_df = aml.leaderboard.as_data_frame()
    lb_df.to_csv("reports/h2o_leaderboard.csv", index=False)
    
    print("\n" + "="*50)
    print("DATA FOR README:")
    print("Top 3 Models by Validation Score (RMSE):")
    print(lb_df[['model_id', 'rmse']].head(3))
    
    print("\nTop 5 Features from H2O:")
    varimp = aml.leader.varimp(use_pandas=True)
    
    # If the leader is a Stacked Ensemble, varimp might be None. 
    # Extract the top base model's varimp instead.
    if varimp is None:
        print("Leader is a Stacked Ensemble. Extracting base model variable importance...")
        for model_id in lb_df['model_id']:
            if 'StackedEnsemble' not in model_id:
                base_model = h2o.get_model(model_id)
                varimp = base_model.varimp(use_pandas=True)
                break
                
    if varimp is not None:
        print(varimp.head(5)[['variable', 'scaled_importance']])
        varimp.to_csv("reports/h2o_top_features.csv", index=False)
        
    print("="*50 + "\n")
    print("H2O AutoML Run Complete!")
    
if __name__ == "__main__":
    run_h2o_automl()