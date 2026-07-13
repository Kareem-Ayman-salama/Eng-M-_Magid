# ملخص تجربة PSO-XGBoost وXGBoost Search

## الهدف

تم تنفيذ تجربة إضافية لمحاولة الاقتراب من نتائج الأبحاث المنشورة التي استخدمت PSO مع XGBoost على UCI Arrhythmia.

## ما تم عمله

- Mutual Information feature prefiltering.
- PSO لاختيار subset من الخصائص.
- PSO لضبط hyperparameters الخاصة بـ XGBoost.
- 5-fold cross-validation للتقييم النهائي.
- Threshold optimization على out-of-fold probabilities.
- RandomizedSearchCV إضافي لـ XGBoost مع SelectKBest كمسار optimization ثاني.

## النتائج

| Model | Accuracy | Balanced Accuracy | F1-score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Current CatBoost Deep | 84.51% | 84.14% | 82.50% | 89.44% |
| PSO-XGBoost | 80.97% | 80.99% | 79.62% | 86.21% |
| RandomizedSearch-XGBoost | 79.20% | 78.91% | 76.85% | 85.30% |
| Published PSO-XGBoost 2025 | 95.24% | 94.81% | 96.30% | Not reported |
| Published MBAR+SMOTE+CatBoost 2022 | 86.33% | Not reported | Not reported | Not reported |

## الاستنتاج

على نفس البروتوكول الحالي لدينا، PSO-XGBoost وRandomizedSearch-XGBoost لم يتفوقا على CatBoost Deep. لذلك النموذج النهائي الأفضل في المشروع يظل CatBoost Deep.

الفرق بيننا وبين ورقة 2025 لا يبدو أنه بسبب اختيار XGBoost فقط، بل غالبًا بسبب تفاصيل إضافية في البروتوكول مثل:

- طريقة feature selection.
- تفاصيل PSO search space.
- طريقة التقسيم أو التحقق.
- preprocessing مختلف.
- احتمالية استخدام binary setup مختلف أو split أكثر تفاؤلًا.

## القرار البحثي

يتم توثيق PSO-XGBoost كتجربة optimization إضافية، لكن لا يتم اعتماده كنموذج نهائي لأنه أقل من CatBoost Deep في كل من Accuracy وROC-AUC.
