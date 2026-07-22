import pandas as pd
import numpy as np 
import os

def prepare_automl_dataset():
    print("Preparing dataset for AutoML...")
    df = pd.read_csv("data/raw/athletes.csv")

    # 1. Base Cleaning & Missing Values
    target_cols = ['deadlift', 'candj', 'snatch', 'backsq']
    df = df.dropna(subset=target_cols + ['gender', 'age', 'weight', 'height', 'howlong', 'train'])
    
    # 2. Strict Physical Outlier Caps (From Assignment 1)
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

    # 4. Feature Selection & Engineering
    # Drop the individual lifts (to prevent target leakage) and identifiers
    df = df.drop(columns=target_cols + ['athlete_id', 'name', 'team', 'affiliate', 
                                        'fran', 'helen', 'grace', 'filthy50', 
                                        'fgonebad', 'run400', 'run5k', 'pullups'])
    
    # Clean survey data
    df = df.replace({'Decline to answer|': np.nan}).dropna()

    # Save the strictly processed dataset for AutoML
    output_path = "data/processed/athletes_automl.csv"
    df.to_csv(output_path, index=False)
    print(f"Cleaned AutoML data saved to {output_path} (Rows: {len(df)})")

if __name__ == "__main__":
    prepare_automl_dataset()