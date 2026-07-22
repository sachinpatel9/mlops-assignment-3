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