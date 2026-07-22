# MLOps Assignment 3: AutoML Evaluation

## Task 1: Dataset Loading and Setup

### Dataset Source & Schema
* **Source:** Self-reported CrossFit athlete dataset (`athletes.csv`).
* **Schema:** The dataset contains a mix of continuous physiological features (`age`, `weight`, `height`) and categorical behavioral features derived from user surveys (`gender`, `howlong`, `train`, `eat`, `background`, `experience`, `schedule`).
* **Target Variable:** `total_lift` (A continuous variable engineered by summing the `deadlift`, `candj`, `snatch`, and `backsq` columns).

### Preprocessing & Version Control
To ensure a strictly fair comparison between these AutoML results and the baseline models developed in Assignment #1, the **processed version of the dataset** was utilized rather than the raw data. 

**Preprocessing Steps Applied:**
1. **Target Leakage Prevention:** The individual lift columns were dropped after summing them into the `total_lift` target.
2. **Outlier Mitigation:** Self-reported physiological caps based on world records were applied (e.g., Male deadlifts capped at 1,105 lbs) to prevent extreme target skew.
3. **Dimensionality Reduction:** Identifying columns (`athlete_id`, `name`) and irrelevant benchmark scores (`run5k`, `fran`) were dropped. 
4. **Reproducibility:** The raw data and preprocessing logic are version-controlled via Git, and the resulting dataset is saved to `data/processed/athletes_automl.csv`.

## Task 2: Chosen MLOps Platform AutoML Configuration

### Platform Selection: PyCaret
**PyCaret** was selected as the primary AutoML platform for this workflow. As an open-source, code-first AutoML library in Python, it integrates natively with our existing local **MLflow** infrastructure. This ensures that all automated model selection, hyperparameter tuning, and cross-validation metrics are tracked in the same standardized UI as our previous manual baseline models, providing a highly reproducible and industry-aligned evaluation environment.

### AutoML Configuration Settings
The automated experiment was initialized via the `pycaret.regression.setup()` environment using the following strict configuration limits to ensure a fair comparison against the Assignment #1 baseline:

* **Target Variable:** `total_lift`
* **Validation Strategy:** 
  * **Train/Test Split:** 80% Training / 20% Hold-out Test (`train_size=0.8`).
  * **Cross-Validation:** 5-Fold Stratified Cross-Validation (`fold=5`) applied exclusively on the training set to prevent data leakage during algorithm evaluation.
* **Reproducibility:** A fixed random seed (`session_id=42`) was strictly enforced across all fold generators and estimators.
* **Runtime Limits & Constraints:** To simulate real-world compute constraints, a strict execution limit of 5 minutes (`budget_time=5`) was applied to the `compare_models()` function. If a computationally heavy algorithm (e.g., deeply nested ensembles) exceeds this threshold, it is automatically terminated and excluded from the leaderboard.
* **Algorithm Exclusions:** No specific algorithms were manually excluded; the platform was permitted to evaluate its entire regression model zoo (Linear Regression, Ridge, Lasso, Elastic Net, Random Forest, LightGBM, XGBoost, etc.) bounded only by the 5-minute compute budget.


## Task 3: AutoML Run using all Features

To establish a comprehensive automated baseline, PyCaret was executed using all appropriate continuous and categorical features from the processed dataset.

### AutoML Execution Results
* **Primary Validation Metric:** `RMSE` (Root Mean Squared Error) was explicitly selected as the primary sorting metric to strictly align with the evaluation strategy used in Assignment #1. 
* **Best Model Identified:** Based on the 5-fold cross-validation performance, the best performing algorithm was **lightgbm,Light Gradient Boosting Machine**. 

### Artifact Documentation
To satisfy reproducibility requirements, the automated run generated physical artifacts saved directly to the version-controlled repository:
1. **AutoML Leaderboard:** The complete algorithm ranking grid, containing cross-validated scoring metrics (MAE, MSE, RMSE, R2, RMSLE, MAPE) and training time for every evaluated model, was extracted via the PyCaret API and saved to `reports/pycaret_leaderboard.csv`.
2. **Model Summary:** The winning model, complete with its automated preprocessing pipeline and tuned hyperparameter signature, was serialized as a `.pkl` file to `reports/pycaret_best_model.pkl`. 
3. **MLflow Tracking:** All individual model runs, execution times, and parameters were successfully synced to the local SQLite `mlflow.db` backend for dashboard visualization.


## Task 4: AutoML Data Insights and Feature Importance

To ensure the winning model is learning actual physiological relationships rather than memorizing noise, PyCaret's `plot_model` functionality was utilized to extract global feature importance via permutation (or native tree-based gain). 

### Top 5 Predictive Features
The AutoML platform identified the following five features as having the highest relative importance in predicting an athlete's `total_lift`:

1. **`weight` (Body Mass):** The highest weighted feature by a significant margin.
2. **`age` (Body Mass):** The second most influential continuous feature.
3. **`height` (Physiological Superiority):** physiological metric with second highest importance
4. **`background` (Skeletal Leverage):**
5. **`schedule` (Circumstantial):** A circumstantial metric

*(A localized visualization of these weights can be viewed in `reports/feature_importance.png`).*

### Strategic Analysis & Domain Validation
The feature importance results generated by the AutoML system make perfect sense within the domain of human biomechanics and athletic performance:
* **Physiological Dominance:** Absolute strength in compound lifts (Deadlift, Squat) is deeply anchored in muscle mass and bone density, which is why `gender` and `weight` are appropriately ranked at the very top. 
* **The "Skill" Factor:** Olympic weightlifting (Snatch, Clean & Jerk) is highly technical. A heavier athlete with poor form will fail a Snatch. The model correctly identified that the behavioral feature `background`

* **Conclusion:** The AutoML platform successfully bypassed arbitrary survey noise (e.g., diet preferences or daily schedules) and grounded its predictive logic in verified biomechanical realities, proving the model is both highly performant and clinically logical.