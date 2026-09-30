# Chronic Kidney Disease Prediction Using Machine Learning

## Overview
An end-to-end, reproducible classical-ML project: data inspection → cleaning → EDA → leakage-free preprocessing pipelines → three models (Logistic Regression, Decision Tree, Random Forest) → cross-validated tuning → medical-style evaluation → explainability → error analysis → a Streamlit demo app. No deep learning is used. All numbers below were produced by the code in this repository.

## Problem statement
Binary classification: given 24 clinical / laboratory features, predict whether a record belongs to the `ckd` (1) or `notckd` (0) class. Questions answered: what problems exist in the raw data, how to handle missing/inconsistent values, which features are associated with CKD, how the models compare, which metrics matter, and which features drive predictions.

## Dataset
Kaggle mirror of the UCI *Chronic Kidney Disease* dataset (`data/raw/kidney_disease.csv`, auto-detected by column names).

| Property | Actual value |
|---|---|
| Records | 400 |
| Raw columns | 26 (`id` + 24 predictors + `classification`) |
| Predictors | 14 numeric/ordinal + 10 categorical |
| Target | 250 `ckd` (62.5%) / 150 `notckd` (37.5%) |
| Exact duplicates | 0 |
| Most-missing features | rbc (152), rc (130), wc (105), pot (88), sod (87) |

**Raw-data problems found (and handled):** tabs/spaces in categories (`'\tno'`, `' yes'`, target `'ckd\t'`), `'\t?'` placeholders that made `pcv`, `wc`, `rc` text columns, empty cells for missing values, and an `id` column. The Kaggle data differs from the prompt's assumptions in one way: 26 columns, not 25 - the extra one is `id`, which was dropped.

<details><summary><b>Data dictionary</b></summary>

| Column | Meaning | Type / unit |
|---|---|---|
| age | Age | years |
| bp | Blood pressure | mm/Hg |
| sg | Urine specific gravity | ordinal 1.005-1.025 |
| al | Urine albumin | ordinal 0-5 |
| su | Urine sugar | ordinal 0-5 |
| rbc / pc | Urine red blood cells / pus cells | normal, abnormal |
| pcc / ba | Pus cell clumps / bacteria | present, notpresent |
| bgr | Random blood glucose | mgs/dl |
| bu | Blood urea | mgs/dl |
| sc | Serum creatinine | mgs/dl |
| sod / pot | Sodium / potassium | mEq/L |
| hemo | Hemoglobin | gms |
| pcv | Packed cell volume | % |
| wc / rc | White / red blood cell count | cells/cumm / millions/cmm |
| htn, dm, cad, pe, ane | Hypertension, diabetes, coronary artery disease, pedal edema, anemia | yes, no |
| appet | Appetite | good, poor |
| classification → target | CKD label | ckd = 1, notckd = 0 |
</details>

## Data preprocessing
- **Stateless cleaning** (`src/data_preprocessing.py`, every change logged): strip whitespace/tabs, lower-case, `?` → NaN, `pd.to_numeric(errors="coerce")`, explicit target mapping, duplicate check (0 removed). No rows are dropped for missing values.
- **Stratified 80/20 split** (`random_state=42`): 320 train / 80 test, done *before* any fitted step.
- **Pipeline (`ColumnTransformer`)**: numeric → median imputation → `StandardScaler`; categorical → most-frequent imputation → `OneHotEncoder(handle_unknown="ignore", drop="if_binary")`. Bundled with each model in a `Pipeline`, so CV re-fits it per fold (no leakage).
- **Outliers are kept**: extreme urea/creatinine values are clinically real; two implausible values (sodium 4.5, potassium 39 and 47) look like entry errors but can't be verified, so they're documented rather than deleted.

## Exploratory data analysis (key findings)
- Hemoglobin, packed cell volume, red-cell count and specific gravity are strongly negatively correlated with CKD (r ≈ -0.70 to -0.77); albumin is positively correlated (r ≈ 0.63).
- Hemoglobin, PCV and RBC count are highly inter-correlated (r up to 0.90).
- **Missingness is class-dependent:** non-CKD patients average 0.69 missing features per row vs 3.63 for CKD patients - a data-collection artifact that models can exploit indirectly.
- Correlation ≠ causation; associations are specific to this sample.

Figures: `reports/figures/` (target distribution, missing values, distributions, boxplots, correlation matrix, outliers).

## Machine-learning models
| Model | Role | Key ideas |
|---|---|---|
| **Logistic Regression** | interpretable baseline | weighted sum → sigmoid → probability; scaling matters for regularisation; coefficients = log-odds |
| **Decision Tree** | interpretable non-linear model | greedy splits minimising Gini/entropy; depth control to avoid overfitting |
| **Random Forest** | ensemble | bootstrap samples + random feature subsets + voting → lower variance |

Tuning: `GridSearchCV` (5-fold stratified, refit on F1) with small grids. Best parameters: `{"Logistic Regression": {"C": 10, "class_weight": "balanced", "solver": "liblinear"}, "Decision Tree": {"class_weight": "balanced", "criterion": "entropy", "max_depth": 5, "min_samples_leaf": 1, "min_samples_split": 2}, "Random Forest": {"class_weight": "balanced", "max_depth": null, "max_features": "sqrt", "min_samples_leaf": 1, "min_samples_split": 2, "n_estimators": 100}}`.

## Evaluation metrics
Accuracy, precision, recall (= sensitivity), specificity, F1, ROC-AUC, confusion matrices, ROC and precision-recall curves. In this setting a false negative (missed CKD case) deserves particular attention, but maximising recall is a trade-off against precision (more false alarms), not a universal goal.

## Results
### Cross-validation (training set, 5-fold stratified, mean ± std) - *not test results*
**Baseline (untuned)**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.991 ± 0.012 | 0.990 ± 0.019 | 0.995 ± 0.010 | 0.993 ± 0.010 | 1.000 ± 0.000 |
| Decision Tree | 0.966 ± 0.006 | 0.976 ± 0.025 | 0.970 ± 0.019 | 0.973 ± 0.004 | 0.964 ± 0.014 |
| Random Forest | 0.991 ± 0.012 | 0.986 ± 0.019 | 1.000 ± 0.000 | 0.993 ± 0.010 | 0.999 ± 0.001 |

**Tuned**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.997 ± 0.006 | 0.995 ± 0.010 | 1.000 ± 0.000 | 0.998 ± 0.005 | 1.000 ± 0.000 |
| Decision Tree | 0.981 ± 0.018 | 0.980 ± 0.018 | 0.990 ± 0.012 | 0.985 ± 0.014 | 0.978 ± 0.021 |
| Random Forest | 0.994 ± 0.008 | 0.990 ± 0.012 | 1.000 ± 0.000 | 0.995 ± 0.006 | 1.000 ± 0.001 |

> Tuned CV scores are slightly optimistic (same folds used for tuning and scoring).

### Model selection
Rule fixed in advance in `src/train.py`, applied to CV results **before** the test set was scored: CV recall → CV F1 → CV ROC-AUC (3 d.p.) → interpretability. **Selected model: Logistic Regression.** (Ranked by CV recall, then CV F1, then CV ROC-AUC (3 d.p.), then interpretability. Logistic Regression: recall=1.000, F1=0.998, ROC-AUC=1.000.)

### Held-out test set (80 patients, threshold 0.50) - evaluated once
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Specificity |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9750 | 1.0000 | 0.9600 | 0.9796 | 1.0000 | 1.0000 |
| Decision Tree | 0.9875 | 1.0000 | 0.9800 | 0.9899 | 0.9900 | 1.0000 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| Logistic Regression | 30 | 0 | 2 | 48 |
| Decision Tree | 30 | 0 | 1 | 49 |
| Random Forest | 30 | 0 | 0 | 50 |

**Reading these results honestly.** The pre-selected model (Logistic Regression) achieved accuracy 0.9750, recall 0.96, precision 1.00 and specificity 1.00 (FN = 2, FP = 0). Random Forest made fewer test errors, but the whole test set is 80 patients (one patient = 1.25 accuracy points) and CV differences between the top models were within one standard deviation, so the models are practically comparable. I did **not** switch models after seeing test results (that would be tuning on the test set). Near-perfect scores on this dataset are well known and do not imply real-world performance.

![Confusion matrices](reports/figures/confusion_matrices.png)
![ROC](reports/figures/roc_curves.png)
![PR](reports/figures/pr_curves.png)

### Threshold analysis (Logistic Regression, out-of-fold on training data)
| Threshold | Precision | Recall | FP | FN |
|---|---|---|---|---|
| 0.100 | 0.980 | 1.000 | 4.000 | 0.000 |
| 0.250 | 0.980 | 1.000 | 4.000 | 0.000 |
| 0.500 | 0.995 | 1.000 | 1.000 | 0.000 |
| 0.750 | 1.000 | 0.970 | 0.000 | 6.000 |
| 0.900 | 1.000 | 0.930 | 0.000 | 14.000 |

Raising the threshold trades false positives for false negatives (precision up, recall down) and vice-versa. The default 0.50 is kept; a different threshold would need a clinical cost rationale and calibrated probabilities (future work).

## Model explainability
**Top features (impurity importances aggregated back from one-hot columns; SHAP for the selected model):**

| Rank | Decision Tree | Random Forest | LR (SHAP, aggregated by column) |
|---|---|---|---|
| 1 | hemo (0.662) | hemo (0.220) | sg (2.72) |
| 2 | sg (0.213) | pcv (0.155) | hemo (2.01) |
| 3 | sc (0.058) | sc (0.137) | pcv (1.98) |
| 4 | htn (0.056) | sg (0.118) | al (1.64) |
| 5 | appet (0.011) | rc (0.099) | htn_yes (1.43) |
| 6 | pcc (0.000) | htn (0.060) | dm_yes (1.31) |

**Logistic Regression coefficients** (scaled numerics: per +1 SD; binary flags: yes vs no; odds ratio = exp(coef); association, not causation):

| Feature | Coefficient | Odds Ratio | Direction |
|---|---|---|---|
| appet_poor | 3.064 | 21.419 | higher odds of CKD |
| htn_yes | 3.064 | 21.418 | higher odds of CKD |
| sg | -2.879 | 0.056 | lower odds of CKD |
| hemo | -2.785 | 0.062 | lower odds of CKD |
| dm_yes | 2.743 | 15.536 | higher odds of CKD |
| pcv | -2.411 | 0.090 | lower odds of CKD |
| sc | 2.098 | 8.154 | higher odds of CKD |
| al | 1.892 | 6.632 | higher odds of CKD |

![Feature importance](reports/figures/feature_importance.png)
![SHAP summary](reports/figures/shap_summary.png)

### Error analysis (selected model; 320 out-of-fold + 80 test predictions)
| outcome | count | sc | hemo | sg | al | bu | n_missing_features |
|---|---|---|---|---|---|---|---|
| FN | 2 | 2.15 | 13.90 | 1.02 | 0.00 | 30.00 | 3.50 |
| FP | 1 | 0.90 | nan | 1.02 | 0.00 | 35.00 | 4.00 |
| TN | 149 | 0.90 | 15.00 | 1.02 | 0.00 | 33.00 | 0.00 |
| TP | 248 | 2.30 | 10.90 | 1.01 | 2.00 | 53.00 | 3.00 |

Only 3 of 400 predictions are wrong (2 missed CKD cases, 1 false alarm). All had predicted probabilities near 0.5 and more missing values than typical correctly-cleared non-CKD patients. With so few errors no medical or statistical conclusion is possible; the observation is descriptive only.

Grouped inputs (patient info, lab measurements, clinical indicators), a **Predict CKD Risk** button, a "Model Prediction" panel (predicted class + model probability), an expandable *About the Model* section, and the disclaimer. The saved pipeline (`models/final_model.pkl`) accepts raw inputs.


## Reproducibility
Python 3.12.3 · pandas 3.0.2 · numpy 2.4.4 · scikit-learn 1.8.0 · matplotlib 3.10.8 · seaborn 0.13.2 · joblib 1.5.3 · streamlit 1.64.0 · shap 0.52.0.
`random_state=42` everywhere; split: 80/20 stratified; CV: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. Details in `reports/environment.json`.

## Limitations
Small dataset (400 records; 80-patient test set → coarse estimates) · single-source, older data of unclear provenance · heavy missingness whose pattern correlates with the label · possible sampling bias (curated CKD/non-CKD groups, not a screening population) · possible data-entry errors · no external, clinical or prospective validation → limited generalisability · probabilities can be overconfident (uncalibrated) · tuned CV scores slightly optimistic · **not a clinical diagnostic system**.

## Future improvements
Larger and more diverse datasets and external validation · probability calibration and cost-based threshold selection · more robust missing-data methods (e.g. model-based or multiple imputation, missingness indicators studied explicitly) · nested CV · richer explainability · bias/fairness analysis · clinical and prospective validation and model monitoring. None of these would by itself make the model clinically valid.

## Dataset attribution
Original data: UCI Machine Learning Repository, *Chronic Kidney Disease* dataset (donated by L. Jerlin Rubini, P. Soundarapandian and P. Eswaran). Kaggle mirror: https://www.kaggle.com/datasets/mansoordaku/ckdisease. UCI repository: Dua, D. and Graff, C. (2019), https://archive.ics.uci.edu/ml. Please consult the original sources for license and citation terms.
# Project By
Shruti Nair
