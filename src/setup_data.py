import pandas as pd
import numpy as np 
import os

def prepare_automl_dataset():
    print("Preparing dataset for AutoML with Advanced Preprocessing...")
    df = pd.read_csv("data/raw/athletes.csv")

    # 1. Base Cleaning
    target_cols = ['deadlift', 'candj', 'snatch', 'backsq']
    df = df.dropna(subset=target_cols)

    # 2. Strict Physical Outlier Caps
    df = df[(df['weight'] > 50) & (df['weight'] < 500)]
    df = df[(df['height'] > 48) & (df['height'] < 96)]
    
    df = df[
        ((df['gender'] == 'Male') & (df['deadlift'] <= 1105)) |
        ((df['gender'] == 'Female') & (df['deadlift'] <= 636))
    ]
    
    df = df[(df['candj'] > 0) & (df['candj'] <= 395)]
    df = df[(df['snatch'] > 0) & (df['snatch'] <= 496)]
    df = df[(df['backsq'] > 0) & (df['backsq'] <= 1069)]

    # 3. Target Creation
    df['total_lift'] = df[target_cols].sum(axis=1)

    # 4. Advanced Feature Engineering
    # Map gender to binary
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0}).fillna(-1).astype(int)
    
    # Parse survey text into structured numeric features
    df['experience_years'] = df['howlong'].map({
        'Less than 6 months': 0.5,
        '6-12 months': 1.0,
        '1-2 years': 1.5,
        '2-4 years': 3.0,
        '4+ years': 5.0
    }).fillna(0.0)

    # Create binary training affiliate flag
    df['is_affiliate_trained'] = df['train'].apply(
        lambda x: 1 if isinstance(x, str) and 'CrossFit Affiliate' in x else 0
    )

    # 5. Clean Missing Values & Isolate Core Features
    core_features = ['gender', 'age', 'weight', 'height', 'experience_years', 'is_affiliate_trained']
    df = df.dropna(subset=core_features)

    # Select only the target and the clean engineered features
    final_cols = ['total_lift'] + core_features
    df_clean = df[final_cols]

    # 6. Save as CSV for PyCaret 
    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/athletes_automl.csv"
    df_clean.to_csv(output_path, index=False)
    print(f"Cleaned AutoML data saved to {output_path} (Rows: {len(df_clean)})")

if __name__ == "__main__":
    prepare_automl_dataset()