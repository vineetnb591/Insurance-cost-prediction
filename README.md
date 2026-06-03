# Insurance Cost Prediction

## Overview
A machine learning classification project to predict whether an individual will file 
a car insurance claim, using a dataset of 10,000 customer records with 19 features 
including demographics, vehicle information, and driving history.

## Key Results
- Best model: **XGBoost** with **85.5% accuracy** and **AUC of 0.919**
- Benchmarked 9 models: Logistic Regression, KNN, Naive Bayes, Decision Tree, 
  Random Forest, AdaBoost, Gradient Boost, XGBoost, and SVC
- Addressed class imbalance (31% claimed vs 69% not claimed) using **SMOTE** 
  oversampling, balancing training data to 5,500 samples per class
- Hyperparameter tuning via **Grid Search CV** across all models

## Dataset
- 10,000 rows, 19 features (18 predictors + 1 binary target: OUTCOME)
- Features include: AGE, DRIVING_EXPERIENCE, CREDIT_SCORE, VEHICLE_OWNERSHIP, 
  ANNUAL_MILEAGE, SPEEDING_VIOLATIONS, PAST_ACCIDENTS, and more
- Missing values in CREDIT_SCORE (982) and ANNUAL_MILEAGE (957) handled via mean imputation

## Methodology
1. **EDA** — Correlation heatmaps, pair plots, count plots; identified 
   DRIVING_EXPERIENCE, VEHICLE_OWNERSHIP, AGE, and VEHICLE_YEAR as top predictors
2. **Feature Engineering** — AGE bucketed into young/middle_age/old/very_old; 
   DRIVING_EXPERIENCE into novice/intermediate/experienced/expert
3. **Preprocessing** — Label encoding for categoricals, StandardScaler normalization, 
   80/20 train-test split
4. **Imbalance Handling** — SMOTE applied to training set only
5. **Modeling** — 9 classifiers trained and evaluated
6. **Hyperparameter Tuning** — Grid Search CV to optimize each model
7. **Evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC, Precision-Recall curves

## Model Performance Summary
| Model | Accuracy | ROC-AUC |
|---|---|---|
| XGBoost | **85.5%** | **0.919** |
| Gradient Boost | 85.4% | 0.922 |
| AdaBoost | 84.0% | 0.914 |
| Random Forest | 83.2% | 0.902 |
| SVC | 82.4% | 0.888 |
| Logistic Regression | 82.2% | 0.899 |
| Decision Tree | 79.6% | 0.765 |
| KNeighbors | 79.3% | 0.846 |
| Gaussian NB | 75.4% | 0.849 |

## Key Insights
- Novice drivers (0–9 years experience) are most likely to file claims
- Young drivers (16–25) have significantly higher claim rates
- DRIVING_EXPERIENCE, VEHICLE_OWNERSHIP, and AGE are the strongest predictors

## Tools & Technologies
**Languages:** Python, R  
**Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Imbalanced-learn (SMOTE), 
Matplotlib, Seaborn, Plotly  

## Collaborators
- Vineet Naren Belagod
- Keerthi Sreenidhi Thota
