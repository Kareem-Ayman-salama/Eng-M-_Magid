# تقرير تحديث تجارب أمراض القلب واضطراب النظم

## الهدف من الإضافة الجديدة

تم إضافة مسار تجريبي جديد بجانب شغل Heart Disease Prediction الأساسي، والهدف منه أن المشروع لا يظل مقتصرًا على التنبؤ بوجود المرض فقط، بل يدعم أيضًا تصنيف نوع الحالة الطبية عند توفر Dataset مناسبة لذلك.

المسار الجديد يعمل على UCI Arrhythmia Dataset لأنها Dataset tabular مناسبة للتجربة السريعة داخل نفس بيئة العمل، وتحتوي على قياسات ECG وخصائص رقمية تساعد في تنفيذ مهمتين:

- Binary Prediction: التنبؤ هل الحالة Normal أم Arrhythmia.
- Grouped Multiclass Classification: تصنيف نوع الاضطراب بعد تجميع الفئات النادرة لتقليل أثر عدم توازن البيانات.

## ملخص البيانات

| البند | القيمة |
| --- | --- |
| عدد الصفوف | 452 |
| عدد الخصائص | 279 |
| عدد الفئات الأصلية الموجودة | 13 |
| عدد الفئات بعد التجميع | 6 |
| عدد القيم المفقودة | 408 |

## ما تم تنفيذه

- تجهيز البيانات وقراءة القيم المفقودة ومعالجتها داخل Pipeline.
- فصل مسار Binary Prediction عن مسار Multiclass Classification.
- استخدام cross-validation بدل نتيجة holdout واحدة حتى تكون النتائج أقل تحيزًا.
- تجربة نماذج كلاسيكية وقوية: Logistic Regression, Random Forest, Extra Trees, SVC.
- تجربة Boosting Models: XGBoost, LightGBM, CatBoost.
- بناء Hybrid Models باستخدام Soft Voting وStacking.
- تجربة Feature Selection قبل XGBoost باستخدام Mutual Information وANOVA.
- إضافة Feature Tokenizer Transformer كتجربة حديثة للبيانات الجدولية.

## نتائج Binary Prediction

أفضل نتيجة في مهمة Normal vs Arrhythmia كانت:

- Model: CatBoost
- Accuracy: 84.07%
- Balanced Accuracy: 83.57%
- F1-score: 81.67%
- ROC-AUC: 90.02%

| model                       | accuracy | balanced_accuracy | f1     | roc_auc |
| --------------------------- | -------- | ----------------- | ------ | ------- |
| CatBoost                    | 84.07%   | 83.57%            | 81.67% | 90.02%  |
| Stacking Hybrid             | 82.96%   | 82.71%            | 81.08% | 89.27%  |
| Soft Voting Hybrid          | 80.98%   | 80.31%            | 77.52% | 88.78%  |
| LightGBM                    | 80.74%   | 80.17%            | 77.67% | 88.36%  |
| Extra Trees                 | 80.54%   | 79.65%            | 76.34% | 87.42%  |
| Random Forest               | 80.54%   | 80.03%            | 77.70% | 87.36%  |
| XGBoost                     | 78.76%   | 78.04%            | 74.74% | 86.71%  |
| SVC RBF                     | 79.65%   | 78.68%            | 74.93% | 85.83%  |
| MI SelectKBest + XGBoost    | 77.68%   | 77.09%            | 74.23% | 83.94%  |
| ANOVA SelectKBest + XGBoost | 75.24%   | 74.43%            | 70.46% | 79.39%  |
| Logistic Regression         | 72.15%   | 71.60%            | 68.07% | 78.80%  |

![Binary model comparison](D:/Freelance work/Mohamed Magied/reports/figures/uci_arrhythmia/binary_roc_auc_comparison.png)

## نتائج Grouped Multiclass Classification

أفضل نتيجة حسب Macro-F1 كانت:

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

![Multiclass model comparison](D:/Freelance work/Mohamed Magied/reports/figures/uci_arrhythmia/multiclass_macro_f1_comparison.png)

## نتيجة Feature Tokenizer Transformer

تمت إضافة Transformer للبيانات الجدولية كاتجاه بحثي حديث، لكنه لم يتفوق على CatBoost أو Hybrid Models في هذه النسخة بسبب صغر حجم البيانات وعدم توازن الفئات.

| Metric | Value |
| --- | --- |
| Epochs | 56 |
| Accuracy | 79.12% |
| Balanced Accuracy | 79.42% |
| F1-score | 78.65% |
| ROC-AUC | 79.01% |
| PR-AUC | 71.08% |

## تفسير النتائج

النتائج الحالية توضح أن إضافة التصنيف ممكنة عمليًا، لكن UCI Arrhythmia Dataset صغيرة جدًا وغير متوازنة، لذلك لا يصح الاعتماد عليها وحدها للوصول إلى أرقام أعلى من المنشورين الأقوياء. أهم قيمة في هذه المرحلة أنها تثبت أن النظام يمكنه التعامل مع prediction وclassification في نفس المشروع.

أفضل أداء في binary prediction جاء من CatBoost لأنه مناسب للبيانات الجدولية الصغيرة والمتوسطة ويتعامل جيدًا مع العلاقات غير الخطية. Hybrid Stacking جاء قريبًا منه، وهذا مهم لأنه يثبت أن دمج الموديلات قابل للتطوير في النسخة النهائية.

في multiclass classification كانت المهمة أصعب بسبب قلة عدد العينات في بعض فئات اضطراب النظم، لذلك تم تجميع الفئات النادرة بدل تدريب نموذج على فئات شديدة الندرة.

## الإضافة العلمية المقترحة

المساهمة الحالية يمكن صياغتها كالتالي:

- المشروع لا يقدم heart disease prediction فقط، بل يضيف arrhythmia classification عند توفر بيانات متعددة الفئات.
- تم بناء pipeline موحد يمكنه تشغيل binary prediction أو grouped multiclass classification.
- تم مقارنة نماذج تقليدية وboosting وhybrid وtransformer داخل نفس إطار التقييم.
- تم توضيح أن Transformer ليس دائمًا الأفضل في الداتا الصغيرة، وأن CatBoost/Hybrid Models أكثر استقرارًا في هذا النوع من البيانات.
- تم تجهيز اتجاه تطبيقي لاحق باستخدام Streamlit بحيث يستطيع المستخدم رفع Dataset واختيار مهمة prediction أو classification.

## خطة Streamlit المقترحة

النسخة التطبيقية يمكن أن تحتوي على:

- رفع ملف CSV من المستخدم.
- فحص الأعمدة والقيم المفقودة تلقائيًا.
- اختيار نوع المهمة: binary prediction أو multiclass classification.
- تشغيل Pipeline محفوظ أو إعادة تدريب نموذج عند الحاجة.
- عرض احتمالية المرض أو نوع التصنيف المتوقع.
- عرض confidence score وملخص لأهم الخصائص المؤثرة في القرار.

## الخلاصة

تم توسيع المشروع من مجرد Heart Disease Prediction إلى إطار أوسع يدعم Prediction وClassification، وتمت إضافة Hybrid Models وFeature Tokenizer Transformer وتجارب Boosting قوية. النتائج الحالية جيدة كإثبات اتجاه وتجربة بحثية منظمة، لكن للحصول على أرقام أعلى ومقارنة أقوى مع المنشورين يجب الانتقال في المرحلة التالية إلى Dataset أكبر مثل PTB-XL أو Chapman ECG لأنها تحتوي على عدد أكبر من المرضى والفئات وتسمح بتدريب نماذج deep learning أقوى.
