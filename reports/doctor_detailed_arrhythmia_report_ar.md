# تقرير تفصيلي للدكتور - Arrhythmia Prediction and Disease Classification

## 1. الهدف من الشغل

الهدف الحالي هو بناء إطار عملي وبحثي على UCI Arrhythmia Dataset يدعم مهمتين أساسيتين:

- Binary prediction: تحديد هل الحالة Normal أم Arrhythmia.
- Disease classification: تصنيف نوع اضطراب النظم، مع تجميع الفئات النادرة بسبب قلة العينات.

## 2. البيانات المتاحة

| البند | القيمة |
| --- | --- |
| Dataset | UCI Arrhythmia |
| عدد الحالات | 452 |
| عدد الخصائص | 279 |
| عدد الفئات الأصلية الموجودة | 13 |
| عدد الفئات بعد التجميع | 6 |
| القيم المفقودة | 408 |

الداتا الأصلية تحتوي على Class 1 كحالة Normal، وباقي الفئات تمثل أنواع مختلفة من Arrhythmia. لكن بعض الفئات تحتوي على عينات قليلة جدًا، لذلك تم استخدام grouped multiclass classification بدل الاعتماد على كل الفئات الأصلية كما هي.

## 3. ما تم تنفيذه

- تجهيز البيانات ومعالجة missing values داخل Pipeline.
- فصل مهمة binary prediction عن مهمة disease classification.
- استخدام Stratified Cross-Validation بدل نتيجة holdout واحدة.
- تجربة نماذج تقليدية: Logistic Regression وSVC وRandom Forest وExtra Trees.
- تجربة boosting models: XGBoost وLightGBM وCatBoost.
- بناء hybrid models: Soft Voting وStacking.
- إضافة Feature Tokenizer Transformer كتجربة advanced للبيانات الجدولية.
- تجربة threshold optimization لتحسين قرار التصنيف.
- تجربة SMOTE وfeature selection داخل cross-validation بدون data leakage.

## 4. أفضل نتيجة Binary Prediction

| Metric | Value |
| --- | --- |
| Best model | CatBoost Deep |
| Best threshold | 0.490 |
| Accuracy | 84.51% |
| Balanced Accuracy | 84.14% |
| F1-score | 82.50% |
| ROC-AUC | 89.44% |

| model                       | best_threshold | accuracy | balanced_accuracy | f1     | roc_auc |
| --------------------------- | -------------- | -------- | ----------------- | ------ | ------- |
| CatBoost Deep               | 0.490          | 84.51%   | 84.14%            | 82.50% | 89.44%  |
| CatBoost Conservative       | 0.380          | 82.96%   | 83.24%            | 82.30% | 89.43%  |
| Advanced Stacking Hybrid    | 0.470          | 83.19%   | 83.07%            | 81.64% | 88.75%  |
| Advanced Soft Voting Hybrid | 0.415          | 82.96%   | 83.01%            | 81.80% | 88.80%  |
| Extra Trees Tuned           | 0.490          | 81.64%   | 80.93%            | 78.33% | 86.92%  |
| Random Forest Tuned         | 0.445          | 79.87%   | 79.86%            | 78.38% | 86.95%  |
| LightGBM Regularized        | 0.525          | 79.65%   | 79.09%            | 76.53% | 85.22%  |
| SVC Tuned RBF               | 0.545          | 79.65%   | 78.71%            | 75.27% | 84.88%  |

![Binary comparison](D:/Freelance work/Mohamed Magied/reports/figures/doctor_detailed_report/binary_best_models.png)

## 5. نتائج Disease Classification

تم تنفيذ classification للمرض كـ grouped multiclass classification. سبب التجميع أن بعض فئات المرض في الداتا الأصلية تحتوي على 2 أو 3 عينات فقط، وهذا غير كاف لتقييم موثوق.

| model               | accuracy | balanced_accuracy | macro_f1 | weighted_f1 |
| ------------------- | -------- | ----------------- | -------- | ----------- |
| CatBoost            | 73.01%   | 63.75%            | 61.20%   | 72.33%      |
| Random Forest       | 74.55%   | 59.90%            | 59.12%   | 73.12%      |
| LightGBM            | 74.33%   | 57.73%            | 58.27%   | 71.43%      |
| XGBoost             | 73.68%   | 55.02%            | 56.69%   | 70.81%      |
| Logistic Regression | 62.83%   | 48.31%            | 48.28%   | 63.88%      |
| Extra Trees         | 67.71%   | 37.28%            | 39.82%   | 61.35%      |

![Multiclass comparison](D:/Freelance work/Mohamed Magied/reports/figures/doctor_detailed_report/classification_best_models.png)

## 6. نتائج SMOTE وFeature Selection

تمت تجربة SMOTE مع feature selection داخل cross-validation. النتيجة لم تتفوق على CatBoost Deep الأساسي، ولذلك تم اعتبارها تجربة بحثية مساعدة وليست النموذج النهائي.

| model                                                | best_threshold | accuracy | balanced_accuracy | f1     | roc_auc |
| ---------------------------------------------------- | -------------- | -------- | ----------------- | ------ | ------- |
| SMOTE + ExtraTrees-SFM + CatBoost                    | 0.415          | 82.74%   | 82.70%            | 81.34% | 88.48%  |
| SMOTE + ExtraTrees-SFM + SMOTE Feature Voting Hybrid | 0.445          | 81.86%   | 81.73%            | 80.19% | 88.54%  |
| SMOTE + ExtraTrees-SFM + Extra Trees                 | 0.475          | 81.86%   | 81.50%            | 79.60% | 87.47%  |
| SMOTE + ExtraTrees-SFM + Random Forest               | 0.505          | 81.19%   | 80.56%            | 78.04% | 87.64%  |
| SMOTE + MI-96 + Random Forest                        | 0.485          | 80.53%   | 80.35%            | 78.64% | 86.36%  |
| SMOTE + MI-96 + SMOTE Feature Voting Hybrid          | 0.465          | 80.31%   | 80.15%            | 78.45% | 85.88%  |

## 7. نتيجة Transformer

| Metric | Value |
| --- | --- |
| Model | Feature Tokenizer Transformer |
| Accuracy | 79.12% |
| Balanced Accuracy | 79.42% |
| F1-score | 78.65% |
| ROC-AUC | 79.01% |

## 8. شكل الـ Output للمستخدم

التطبيق المقترح باستخدام Streamlit سيعرض للمستخدم:

- Upload CSV.
- Validate schema وعدد الصفوف والأعمدة والقيم المفقودة.
- اختيار نوع المهمة: Prediction أو Classification.
- جدول نتائج يحتوي على predicted class وprobability وconfidence وrisk level.
- عند وجود target column يتم عرض Accuracy وF1-score وROC-AUC وConfusion Matrix.
- إمكانية تنزيل النتائج كملف CSV.

![Output flow](D:/Freelance work/Mohamed Magied/reports/figures/doctor_detailed_report/streamlit_output_flow.png)

## 9. الخلاصة

المشروع حاليًا يحتوي على مسارين واضحين: prediction لوجود Arrhythmia، وclassification مجمع لأنواع المرض. أفضل نتيجة حالية في binary prediction هي Accuracy = 84.51% وROC-AUC = 89.44%. أما disease classification فهي أصعب بسبب قلة العينات وعدم توازن الفئات، ولذلك تم استخدام grouped classes للحصول على تقييم أكثر منطقية.
