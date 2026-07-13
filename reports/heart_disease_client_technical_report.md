# Heart Disease Prediction and Detection Using AI Techniques

## Executive Summary

This report summarizes the completed experimental work for the heart disease
prediction thesis project. The work started from the originally provided
Kaggle cardiovascular dataset and was extended into a stronger comparative
experimental framework covering multiple datasets, classical ML models,
tuned gradient boosting models, and hybrid ensemble models.

The strongest repeated cross-validation result was achieved by
**Tuned CatBoostClassifier** with **94.26% accuracy**,
**94.56% F1-score**, and
**96.78% ROC-AUC**.

The strongest discrimination score was achieved by **Tuned Soft Voting Hybrid**
with **97.21% ROC-AUC** and
**96.63% PR-AUC**.

On the final holdout evaluation, the best ROC-AUC model was
**Tuned Stacking Hybrid**, reaching **92.86%
accuracy**, **93.12% F1-score**, and
**97.40% ROC-AUC**.

All metrics in this report are generated from the executed notebook output
files under:

`D:\Freelance work\Mohamed Magied\outputs\clean_notebook`

## What We Worked On

1. Reviewed the proposal topic and translated it into a practical machine
   learning experiment plan.
2. Started with the original Kaggle Cardiovascular Disease dataset.
3. Tested many classical and ensemble machine learning models.
4. Identified that the Kaggle dataset performance ceiling is around the
   low-to-mid 70% accuracy range under clean validation.
5. Added UCI Heart Disease datasets to support a medically standard
   comparison benchmark.
6. Added the OpenML Heart Disease Comprehensive 1190 dataset, which combines
   Cleveland, Hungarian, Switzerland, Long Beach VA, and Statlog sources.
7. Built tuned single models and hybrid ensemble models.
8. Added SHAP explainability to interpret the strongest tree-based model.
9. Added feature selection experiments to test whether reduced clinical
   feature sets improve performance.
10. Added a Feature Tokenizer Transformer deep tabular baseline.
11. Organized the final work into one clean notebook with reproducible outputs,
   visualizations, comparison tables, and printable HTML.

## Main Deliverables

- Main notebook:
  `notebooks/00_clean_heart_disease_experiments.ipynb`
- Executed notebook:
  `outputs/clean_notebook/00_clean_heart_disease_experiments.executed.ipynb`
- Printable HTML:
  `outputs/clean_notebook/00_clean_heart_disease_experiments.html`
- Generated CSV outputs:
  `outputs/clean_notebook/*.csv`
- Reusable code package:
  `src/heart_disease_prediction/`
- Notebook generator:
  `scripts/generate_clean_architecture_notebook.py`

## Datasets Used

### 1. Kaggle Cardiovascular Disease Dataset

- File: `archive/cardio_train.csv`
- Size: approximately 70,000 records before cleaning.
- Target: `cardio`
- Role: Original dataset and baseline experiment source.
- Finding: Good for large-scale experimentation, but the available feature
  set limits clean predictive performance.

### 2. UCI Heart Disease Dataset

- Folder: `data/uci_heart/`
- Sources: Cleveland, Hungarian, Switzerland, and Long Beach VA processed files.
- Role: Classical medical benchmark for heart disease prediction.
- Finding: Smaller than Kaggle, but clinically closer to the standard
  published heart disease ML literature.

### 3. OpenML Heart Disease Comprehensive 1190

- Folder: `data/openml_heart_1190/`
- Size: 1,190 records.
- Sources: Cleveland, Hungarian, Switzerland, Long Beach VA, and Statlog.
- Role: Final strongest benchmark dataset.
- Finding: Best dataset for strong thesis-level results because it combines
  multiple established heart disease sources into one structured benchmark.

## Models Tested

The experiments covered:

- Logistic Regression
- Naive Bayes
- Support Vector Machines
- Random Forest
- Extra Trees
- Gradient Boosting
- AdaBoost
- XGBoost
- LightGBM
- CatBoost
- Multi-layer Perceptron
- Hard Voting ensemble
- Soft Voting hybrid ensemble
- Stacking hybrid ensemble
- Feature-selection-enhanced XGBoost and CatBoost
- Feature Tokenizer Transformer for tabular clinical data

## Hybrid Model Design

Yes, the project now includes true hybrid models.

### Soft Voting Hybrid

The soft voting hybrid combines predicted class probabilities from multiple
strong learners. This is useful when several models are individually strong
but capture slightly different decision boundaries.

### Stacking Hybrid

The stacking hybrid uses several base learners, then trains a meta-model on
their outputs. This allows the final learner to identify when each base
model should be trusted more.

### Why These Hybrids Were Selected

The hybrid strategy was selected based on:

- Strong standalone performance of tree-based boosting models.
- Complementary behavior between CatBoost, LightGBM, XGBoost, Random Forest,
  and Extra Trees.
- Better ROC-AUC and PR-AUC from probability-based ensemble predictions.
- Need for a thesis-level model that goes beyond a single baseline classifier.

## Results Compared Across Datasets

| dataset                               | model                                      | accuracy | f1     | recall | roc_auc |
| ------------------------------------- | ------------------------------------------ | -------- | ------ | ------ | ------- |
| OpenML Heart 1190                     | Tuned Soft Voting Hybrid                   | 94.17%   | 94.52% | 95.23% | 97.21%  |
| OpenML Heart 1190                     | Tuned CatBoostClassifier                   | 94.26%   | 94.56% | 94.86% | 96.78%  |
| OpenML Heart 1190 + Feature Selection | Mutual Information SelectKBest + Tuned XGB | 90.56%   | 91.00% | 90.24% | 95.14%  |
| OpenML Heart 1190 + FT-Transformer    | Feature Tokenizer Transformer              | 88.24%   | 89.23% | 92.06% | 92.80%  |
| UCI Cleveland                         | Random Forest                              | 83.49%   | 81.42% | 79.10% | 91.37%  |
| UCI All Processed                     | SVC RBF                                    | 82.03%   | 83.64% | 83.07% | 87.95%  |
| Kaggle CVD 70k                        | GradientBoostingClassifier                 | 72.23%   | 73.93% | 79.58% | 80.13%  |

## OpenML Baseline Model Results

| model                      | accuracy | accuracy_std | precision | recall | f1     | roc_auc | roc_auc_std | pr_auc |
| -------------------------- | -------- | ------------ | --------- | ------ | ------ | ------- | ----------- | ------ |
| XGBClassifier              | 92.02%   | 1.85%        | 93.05%    | 91.89% | 92.40% | 96.28%  | 1.06%       | 95.58% |
| CatBoostClassifier         | 91.76%   | 1.95%        | 92.28%    | 92.21% | 92.20% | 96.40%  | 1.13%       | 95.41% |
| Random Forest              | 91.09%   | 1.96%        | 91.47%    | 91.89% | 91.60% | 96.36%  | 1.11%       | 95.81% |
| ExtraTreesClassifier       | 88.99%   | 2.36%        | 89.24%    | 90.14% | 89.64% | 95.49%  | 1.14%       | 95.49% |
| LGBMClassifier             | 88.99%   | 2.30%        | 89.88%    | 89.35% | 89.55% | 94.96%  | 1.36%       | 94.65% |
| SVC RBF                    | 87.23%   | 2.31%        | 87.28%    | 88.87% | 88.03% | 93.80%  | 1.64%       | 93.70% |
| GradientBoostingClassifier | 86.39%   | 2.91%        | 86.41%    | 88.23% | 87.26% | 93.37%  | 1.37%       | 93.28% |
| Logistic Regression        | 84.71%   | 2.65%        | 85.23%    | 86.16% | 85.63% | 91.73%  | 2.05%       | 92.43% |

## Tuned Models and Hybrid Ensembles

| model                      | accuracy | accuracy_std | precision | recall | f1     | roc_auc | roc_auc_std | pr_auc |
| -------------------------- | -------- | ------------ | --------- | ------ | ------ | ------- | ----------- | ------ |
| Tuned CatBoostClassifier   | 94.26%   | 2.08%        | 94.38%    | 94.86% | 94.56% | 96.78%  | 1.55%       | 95.86% |
| Tuned Soft Voting Hybrid   | 94.17%   | 1.59%        | 93.89%    | 95.23% | 94.52% | 97.21%  | 1.43%       | 96.63% |
| Tuned Stacking Hybrid      | 94.06%   | 1.68%        | 94.03%    | 94.86% | 94.40% | 97.19%  | 1.44%       | 96.61% |
| Tuned LGBMClassifier       | 93.81%   | 1.72%        | 93.90%    | 94.49% | 94.15% | 96.90%  | 1.60%       | 96.00% |
| Tuned ExtraTreesClassifier | 93.42%   | 1.55%        | 93.13%    | 94.60% | 93.81% | 96.92%  | 1.05%       | 96.50% |
| Tuned XGBClassifier        | 93.28%   | 1.76%        | 93.28%    | 94.12% | 93.66% | 96.37%  | 1.63%       | 95.41% |
| Tuned Random Forest        | 93.05%   | 1.83%        | 92.96%    | 94.06% | 93.47% | 96.75%  | 1.45%       | 96.08% |

## Feature Selection Experiments

| model                                      | accuracy | accuracy_std | precision | recall | f1     | roc_auc | roc_auc_std | pr_auc |
| ------------------------------------------ | -------- | ------------ | --------- | ------ | ------ | ------- | ----------- | ------ |
| Mutual Information SelectKBest + Tuned XGB | 90.56%   | 2.40%        | 91.87%    | 90.24% | 91.00% | 95.14%  | 2.02%       | 94.22% |
| ANOVA SelectKBest + Tuned CatBoost         | 88.49%   | 2.76%        | 89.71%    | 88.49% | 89.00% | 93.74%  | 2.09%       | 93.38% |
| ANOVA SelectKBest + Tuned XGB              | 87.42%   | 2.96%        | 89.12%    | 86.96% | 87.94% | 92.71%  | 2.61%       | 92.72% |

The selected-feature experiments are useful as an analysis contribution. In
the current run, feature selection reduced the feature space but did not
outperform the full tuned boosting and hybrid models. This supports the
final decision to keep the full OpenML feature representation for the main
predictive model.

## Feature Tokenizer Transformer

| model                         | epochs | best_valid_auc | accuracy | precision | recall | f1     | roc_auc | pr_auc |
| ----------------------------- | ------ | -------------- | -------- | --------- | ------ | ------ | ------- | ------ |
| Feature Tokenizer Transformer | 24     | 94.07%         | 88.24%   | 86.57%    | 92.06% | 89.23% | 92.80%  | 90.33% |

The Feature Tokenizer Transformer was added as a modern deep tabular
learning baseline. It converts each clinical variable into a token embedding
and uses self-attention to learn interactions between variables. In this run,
it performed competitively but remained below tuned boosting and hybrid
ensembles, which is expected on small-to-medium structured clinical datasets.

## SHAP Explainability

| feature             | mean_abs_shap      |
| ------------------- | ------------------ |
| chest_pain_type_4.0 | 1.0901115744116048 |
| ST_slope_2.0        | 0.9824976081888284 |
| oldpeak             | 0.8601912077693442 |
| ST_slope_1.0        | 0.8558491137864163 |
| cholesterol         | 0.7732620850698556 |
| max_heart_rate      | 0.5968171613736415 |
| exercise_angina_1.0 | 0.5889404130214758 |
| sex_1.0             | 0.5165384956746343 |
| age                 | 0.494082307021947  |
| resting_bp_s        | 0.4269255127930438 |
| sex_0.0             | 0.354253630880887  |
| resting_ecg_2.0     | 0.3233158089041967 |

SHAP explains which clinical factors most influenced the final tree-based
model. The strongest contributors included chest pain type, ST slope,
oldpeak, cholesterol, maximum heart rate, exercise angina, sex, age, and
resting blood pressure.

## Final Holdout Evaluation

| model                    | accuracy | precision | recall | f1     | roc_auc | pr_auc |
| ------------------------ | -------- | --------- | ------ | ------ | ------- | ------ |
| Tuned Stacking Hybrid    | 92.86%   | 95.04%    | 91.27% | 93.12% | 97.40%  | 96.90% |
| Tuned Soft Voting Hybrid | 92.86%   | 94.31%    | 92.06% | 93.17% | 97.36%  | 96.91% |
| Tuned CatBoostClassifier | 92.44%   | 95.00%    | 90.48% | 92.68% | 96.75%  | 96.08% |

## Comparison With the Provided Published Baseline

| study                                                          | dataset                            | method                            | accuracy | sensitivity_recall           | roc_auc                 | comparison_type                                          | source                                                                         |
| -------------------------------------------------------------- | ---------------------------------- | --------------------------------- | -------- | ---------------------------- | ----------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Ensemble learning with explainable AI, Scientific Reports 2025 | HDDC / D1                          | Stacking ensemble                 | 91%      | Not reported in summary line | 0.92                    | Same family of heart-disease benchmark; protocol differs | https://www.nature.com/articles/s41598-025-97547-6                             |
| Ensemble learning with explainable AI, Scientific Reports 2025 | UHDD / D2                          | Stacking ensemble                 | 98%      | Not reported in summary line | 0.97                    | Same family of heart-disease benchmark; protocol differs | https://www.nature.com/articles/s41598-025-97547-6                             |
| Optimized Ensemble Learning with XAI, Information 2024         | Cleveland                          | Bayesian-optimized XGBoost + SHAP | 98.40%   | 98.90%                       | Not reported in Table 5 | Dataset overlaps with UCI Cleveland; protocol differs    | https://pdfs.semanticscholar.org/3747/6c8130d1b63c9b1894870c9243026c98a24e.pdf |
| Optimized Ensemble Learning with XAI, Information 2024         | Framingham                         | Bayesian-optimized XGBoost + SHAP | 95.90%   | 97.50%                       | Not reported in Table 6 | Different dataset; indirect comparison only              | https://pdfs.semanticscholar.org/3747/6c8130d1b63c9b1894870c9243026c98a24e.pdf |
| This work                                                      | OpenML Heart 1190                  | Tuned CatBoostClassifier          | 94.26%   | 94.86%                       | 0.9678                  | Our reproducible repeated-CV result                      | Generated by this notebook                                                     |
| This work                                                      | OpenML Heart 1190                  | Tuned Soft Voting Hybrid          | 94.17%   | 95.23%                       | 0.9721                  | Our reproducible repeated-CV hybrid result               | Generated by this notebook                                                     |
| This work                                                      | OpenML Heart 1190 + FT-Transformer | Feature Tokenizer Transformer     | 88.24%   | 92.06%                       | 0.9280                  | Our deep tabular baseline                                | Generated by this notebook                                                     |

These rows separate literature-reported metrics from notebook-generated
reproducible metrics. Direct comparison is valid only when the same dataset,
split strategy, preprocessing, and evaluation protocol are used. Therefore,
the published results are used for literature positioning, while our results
are reported as reproducible repeated-CV and holdout outputs.

## Research Contribution Positioning

| issue                                                               | project_response                                                         | evidence                                                                                               |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------ |
| Using multiple datasets alone is not a sufficient contribution.     | Treat multiple datasets as validation evidence, not as the main novelty. | Kaggle, UCI, OpenML, feature-selection, and FT-Transformer results are reported separately.            |
| A clear technical contribution is required.                         | Propose a unified explainable hybrid framework.                          | Soft Voting and Stacking hybrids combine tuned boosting and tree-ensemble learners.                    |
| The model should not be a black box.                                | Add SHAP explainability for the final tree-based model.                  | SHAP identifies chest pain type, ST slope, oldpeak, cholesterol, max heart rate, and exercise angina.  |
| Deep learning should be represented if the thesis claims modern AI. | Add Feature Tokenizer Transformer for tabular clinical data.             | FT-Transformer is evaluated as a deep tabular baseline and compared with boosting/hybrid models.       |
| Feature engineering/selection should be evaluated, not assumed.     | Add ANOVA and mutual-information SelectKBest experiments.                | Feature selection is shown to reduce dimensionality but does not outperform the full hybrid framework. |
| High published accuracies may come from different protocols.        | Separate published results from reproducible notebook-generated results. | Published-work comparison table includes a comparison-type note for each row.                          |

The main contribution should not be presented as using multiple datasets
alone. Multiple datasets are validation evidence. The contribution is the
unified explainable hybrid framework that combines tuned boosting/tree
ensembles, soft voting, stacking, SHAP explainability, feature-selection
analysis, FT-Transformer deep tabular benchmarking, and robust validation.

## Clean Architecture Applied

The final notebook is organized into clear layers:

- Configuration layer: paths, seed, repeated CV settings, and output paths.
- Data layer: dataset loading, validation, cleaning, and dataset metadata.
- Feature layer: preprocessing pipelines for numeric and categorical fields.
- Model layer: baseline models, tuned models, and hybrid ensemble builders.
- Evaluation layer: cross-validation, holdout testing, ROC-AUC, PR-AUC,
  F1-score, precision, recall, and confusion matrix generation.
- Visualization layer: target distributions, feature plots, correlation
  heatmaps, leaderboards, ROC curves, PR curves, and feature importance.

## What Was Added Beyond the Initial Plan

- Multi-dataset experimental comparison.
- UCI and OpenML benchmark integration.
- Tuned boosting models.
- Soft voting hybrid model.
- Stacking hybrid model.
- SHAP explainability.
- Feature selection analysis.
- Feature Tokenizer Transformer deep tabular baseline.
- Repeated stratified cross-validation.
- Final holdout validation.
- Printable HTML output.
- Clean notebook structure.
- Generated result CSVs for traceability.

## Recommended Client Message

The project has moved from a basic single-dataset reproduction into a
stronger thesis-ready experimental framework. The final work compares
multiple heart disease datasets, tests a broad set of machine learning
models, adds tuned hybrid ensemble models, includes SHAP explainability,
evaluates feature selection, and adds a Feature Tokenizer Transformer deep
tabular baseline. The strongest clean results were obtained on the OpenML
Heart Disease Comprehensive 1190 dataset, with approximately 94% accuracy
and above 97% ROC-AUC.

## Next Recommended Improvements

- Add a hyperparameter search section inside the notebook if the client
  wants tuning to be fully visible rather than only using tuned parameters.
- Add nested cross-validation for optimized XGBoost if direct comparison
  with Cleveland-specific published papers is required.
- Add a short final methodology diagram for the thesis chapter.
- Export the final notebook to PDF after the client approves wording.
