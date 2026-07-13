# تقرير النسخة المحسنة - UCI Arrhythmia

## الهدف

تم تثبيت العمل على UCI Arrhythmia Dataset، ثم تحسين النتائج باستخدام advanced models وhybrid models وthreshold optimization، مع الحفاظ على تقييم واضح باستخدام cross-validation بدل إدخال نتائج يدويًا.

## حجم البيانات

- عدد الحالات: 452
- عدد الخصائص: 279
- عدد الفئات الأصلية الموجودة: 13
- عدد الفئات بعد التجميع: 6
- القيم المفقودة: 408

## ما تم تحسينه

- إضافة tuned CatBoost بنسختين: Conservative وDeep.
- إضافة tuned XGBoost وLightGBM مع regularization لتقليل overfitting.
- إضافة tuned Random Forest وExtra Trees.
- إضافة Advanced Soft Voting Hybrid وAdvanced Stacking Hybrid.
- إضافة threshold optimization على out-of-fold probabilities لاختيار أفضل cutoff بدل 0.5.
- الإبقاء على Feature Tokenizer Transformer كتجربة advanced حديثة.

## أفضل نتيجة قبل التحسين المتقدم

- Model: CatBoost
- Accuracy: 84.07%
- Balanced Accuracy: 83.57%
- F1-score: 81.67%
- ROC-AUC: 90.02%

## أفضل نتيجة بعد التحسين

أفضل نتيجة CV حسب ROC-AUC:

- Model: CatBoost Conservative
- Accuracy: 82.08%
- Balanced Accuracy: 81.51%
- F1-score: 79.23%
- ROC-AUC: 89.83%

أفضل نتيجة بعد threshold optimization:

- Model: CatBoost Deep
- Best Threshold: 0.490
- Accuracy: 84.51%
- Balanced Accuracy: 84.14%
- F1-score: 82.50%
- ROC-AUC: 89.44%

## جدول advanced CV

| model                          | accuracy | balanced_accuracy | f1     | roc_auc |
| ------------------------------ | -------- | ----------------- | ------ | ------- |
| CatBoost Conservative          | 82.08%   | 81.51%            | 79.23% | 89.83%  |
| CatBoost Deep                  | 84.07%   | 83.61%            | 81.84% | 89.74%  |
| Advanced Stacking Hybrid       | 82.30%   | 82.05%            | 80.33% | 89.14%  |
| Advanced Soft Voting Hybrid    | 82.31%   | 81.72%            | 79.46% | 88.98%  |
| Extra Trees Tuned              | 81.20%   | 80.35%            | 77.18% | 86.98%  |
| Random Forest Tuned            | 78.77%   | 78.02%            | 74.87% | 86.96%  |
| LightGBM Regularized           | 78.98%   | 78.50%            | 75.80% | 85.31%  |
| SVC Tuned RBF                  | 79.21%   | 78.24%            | 74.52% | 85.22%  |
| XGBoost Regularized            | 77.89%   | 76.72%            | 72.13% | 84.49%  |
| Logistic L2                    | 76.57%   | 76.07%            | 73.00% | 82.78%  |
| Logistic L1                    | 76.57%   | 75.69%            | 71.53% | 82.55%  |
| MI-60 + XGBoost Regularized    | 75.91%   | 74.86%            | 70.37% | 82.19%  |
| ANOVA-70 + XGBoost Regularized | 74.57%   | 73.52%            | 68.67% | 79.44%  |

## جدول threshold optimized

| model                          | best_threshold | accuracy | balanced_accuracy | f1     | roc_auc |
| ------------------------------ | -------------- | -------- | ----------------- | ------ | ------- |
| CatBoost Deep                  | 0.490          | 84.51%   | 84.14%            | 82.50% | 89.44%  |
| CatBoost Conservative          | 0.380          | 82.96%   | 83.24%            | 82.30% | 89.43%  |
| Advanced Stacking Hybrid       | 0.470          | 83.19%   | 83.07%            | 81.64% | 88.75%  |
| Advanced Soft Voting Hybrid    | 0.415          | 82.96%   | 83.01%            | 81.80% | 88.80%  |
| Extra Trees Tuned              | 0.490          | 81.64%   | 80.93%            | 78.33% | 86.92%  |
| Random Forest Tuned            | 0.445          | 79.87%   | 79.86%            | 78.38% | 86.95%  |
| LightGBM Regularized           | 0.525          | 79.65%   | 79.09%            | 76.53% | 85.22%  |
| SVC Tuned RBF                  | 0.545          | 79.65%   | 78.71%            | 75.27% | 84.88%  |
| XGBoost Regularized            | 0.370          | 78.10%   | 78.18%            | 76.81% | 84.48%  |
| MI-60 + XGBoost Regularized    | 0.425          | 77.88%   | 77.57%            | 75.37% | 81.81%  |
| Logistic L1                    | 0.470          | 77.21%   | 76.81%            | 74.31% | 82.24%  |
| Logistic L2                    | 0.470          | 76.99%   | 76.68%            | 74.38% | 82.70%  |
| ANOVA-70 + XGBoost Regularized | 0.400          | 76.33%   | 76.18%            | 74.22% | 79.24%  |

![Optimized threshold comparison](D:/Freelance work/Mohamed Magied/reports/figures/uci_arrhythmia/optimized_threshold_balanced_accuracy.png)

## نتائج التصنيف متعدد الفئات

أفضل نتيجة في grouped multiclass classification:

- Model: CatBoost
- Accuracy: 73.01%
- Balanced Accuracy: 63.75%
- Macro-F1: 61.20%
- Weighted-F1: 72.33%

| model               | accuracy | balanced_accuracy | macro_f1 | weighted_f1 |
| ------------------- | -------- | ----------------- | -------- | ----------- |
| CatBoost            | 73.01%   | 63.75%            | 61.20%   | 72.33%      |
| Random Forest       | 74.55%   | 59.90%            | 59.12%   | 73.12%      |
| LightGBM            | 74.33%   | 57.73%            | 58.27%   | 71.43%      |
| XGBoost             | 73.68%   | 55.02%            | 56.69%   | 70.81%      |
| Logistic Regression | 62.83%   | 48.31%            | 48.28%   | 63.88%      |
| Extra Trees         | 67.71%   | 37.28%            | 39.82%   | 61.35%      |

## نتيجة Feature Tokenizer Transformer

- Accuracy: 79.12%
- Balanced Accuracy: 79.42%
- F1-score: 78.65%
- ROC-AUC: 79.01%

## الخلاصة

تم تحسين أفضل accuracy من 84.07% إلى 84.51% بعد استخدام tuning وthreshold optimization. كما تم تثبيت hybrid models وadvanced boosting models وTransformer baseline داخل نفس خط التجارب. النتيجة ما زالت محدودة بحجم الداتا الصغير وعدم توازن الفئات، لكنها الآن منظمة وقابلة للعرض والدفاع البحثي بشكل أفضل.
