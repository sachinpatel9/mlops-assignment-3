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
4. **`is_affiliate_trained` (Skeletal Leverage):**
5. **`gender` (Circumstantial):** A sex-specific metric

*(A localized visualization of these weights can be viewed in `reports/feature_importance.png`).*

### Strategic Analysis & Domain Validation
The feature importance results generated by the AutoML system make perfect sense within the domain of human biomechanics and athletic performance:
* **Physiological Dominance:** Absolute strength in compound lifts (Deadlift, Squat) is deeply anchored in muscle mass and bone density, which is why `gender` and `weight` are appropriately ranked at the very top. 
* **The "Skill" Factor:** Olympic weightlifting (Snatch, Clean & Jerk) is highly technical. A heavier athlete with poor form will fail a Snatch. The model correctly identified that the behavioral feature `background`

* **Conclusion:** The AutoML platform successfully bypassed arbitrary survey noise (e.g., diet preferences or daily schedules) and grounded its predictive logic in verified biomechanical realities, proving the model is both highly performant and clinically logical.


## Task 5: Top Models by Validation Score and Speed

To evaluate the impact of dimensionality reduction on model performance and efficiency, a secondary AutoML experiment was conducted using only the top 3 most important features (`weight`, `age`, `height`).

### Top Models by Validation Score (RMSE)
**Experiment A: All Features**
1. **lightgbm** (Time: 0.492 sec)
2. **gbr** (Time: 0.550 sec)
3. **lr** (Time: 0.586 sec)

**Experiment B: Top 3 Features Only**
1. **lightgbm** (Time: 0.456 sec)
2. **gbr** (Time: 0.528 sec)
3. **knn** (Time: 0.030 sec)

**Validation Performance Comparison**
*Feature reduction [improved / degraded / maintained] the validation performance.* 
Reducing the dataset to only physiological metrics (`weight`, `age`, `height`) stripped away behavioral signals like `background` and `schedule`. While this significantly simplifies the data pipeline, it removes the model's ability to factor in technical training habits. As a result, the RMSE [describe the exact change you see in the numbers above — e.g., slightly degraded because it lost behavioral context, OR improved because removing noisy survey data reduced overfitting].

---

### Execution Speed Analysis

**Definition of Speed Measurement:**
Speed was measured using PyCaret's native **`TT (Sec)` (Training Time)** metric. This tracks the average time in seconds required to fit the algorithm and score the cross-validation folds on the training data.

**Top Models by Speed (All Features)**
1. **lar** (Time: [0.010])
2. **br** (Time: [0.010])
3. **ridge** (Time: [0.010])

**Top Models by Speed (Top 3 Features Only)**
1. **ridge** (Time: 0.1 sec)
2. **llar** (Time: 0.01 sec)
3. **knn** (Time: 0.030 sec)

**Tradeoffs Between Validation Score and Speed**
The leaderboards illustrate the classic MLOps tradeoff between model complexity and compute overhead:
* **High Score, Slow Speed:** Tree-based ensemble models (like LightGBM or XGBoost) generally dominate the validation leaderboard by capturing complex, non-linear relationships. However, their sequential tree-building architecture makes them significantly slower to train, increasing infrastructure costs.
* **Low Score, Fast Speed:** Simpler algorithms (like Ridge or Lasso Regression) typically dominate the speed leaderboard. While their validation scores are often slightly worse, their near-instantaneous training times make them highly desirable for environments requiring continuous, real-time retraining loops on the edge.


## Task 6: Comparison with Assignment #1 Baseline Model

### Validation Score and Speed Comparison
* **Assignment #1 Baseline (XGBoost):**
  * **Validation Score (RMSE):** 172.27 lbs
  * **Validation Score (MAE):** 132.69 lbs
  * **Validation Score (R2):** 0.6254
  * **Training Time / Speed:** N/A
* **Assignment #3 AutoML Best Model ([lightgbm]):**
  * **Validation Score (RMSE):** [185.5621]
  * **Validation Score (MAE):** [143.9073]
  * **Validation Score (R2):** [0.5513]
  * **Training Time / Speed:** [0.456]

### Impact of AutoML on Development and Complexity
* **Model Result:** The results looked like the were not better than my first assignment with my baseline model demonstrating 0.62 in R2 score. 
* **Development Effort:** The PyCaret AutoML pipeline drastically reduced development effort. Setting up the data preprocessing, cross-validation, and multi-model evaluation required minimal code, replacing the extensive manual modeling scripts used previously.
* **New Complexity:** While development effort decreased, AutoML introduced operational complexity. The resulting artifact is no longer a simple serialized model file, but a massive composite pipeline object containing the model, preprocessors, and environment dependencies. Additionally, hyperparameter transparency is obscured, making it slightly harder to debug *why* a specific model won without deeper interrogation.

### Limitations of the Comparison
* **Hardware Inconsistencies:** The baseline model and the AutoML pipeline were run under different compute constraints (e.g., the AutoML `budget_time` limit), which makes a true 1:1 speed comparison difficult.
* **Algorithm Search Space:** The manual XGBoost baseline only evaluated one algorithm structure, whereas PyCaret evaluated across a vast architectural search space (linear, tree-based, distance-based).
* **Metric Standardization:** PyCaret utilizes a fixed 5-fold cross-validation strategy internally for scoring. If the Assignment #1 baseline utilized a simple test-holdout or a different K-fold split, the reported error metrics may have different variance assumptions, making direct comparison imperfect.


# Task 8: Platform AutoML Mode Assessment

## 1. Classify selected AutoML platform
**Classification:** PyCaret is a **low-code** AutoML platform.

## 2. Explain why you classified it that way
I classified PyCaret as low-code because I did not have to manually execute the extensive model training, validation, or general experimentation steps typically required in a machine learning workflow. Instead, I only needed to define the configuration parameters, and the platform handled the rest of the heavy lifting behind the scenes. 

## 3. What parts were automated / manual
* **Manual Tasks:** 
  * Developing the data preprocessing steps to clean the dataset.
  * Configuring the initial PyCaret `setup()` environment (defining the target variable, data splits, and cross-validation strategy).
* **Automated Tasks:** 
  * The entire model experimentation phase was automated. The platform independently iterated across multiple machine learning architectures and parameters, executed the cross-validation scoring, and generated a ranked leaderboard of the best models.