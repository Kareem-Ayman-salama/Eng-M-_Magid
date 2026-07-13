# تعليمات تشغيل نوتبوك AF الكامل على Kaggle

## الهدف

النوتبوك `notebooks/02_kaggle_af_full_pipeline.ipynb` يبني تجربة كاملة على ثلاث قواعد بيانات ECG خاصة بالـ Atrial Fibrillation، ويغطي:

- Binary prediction: التنبؤ بوجود AF/AFL مقابل non-AF.
- Rhythm classification: تصنيف نوع الإيقاع/المرض من الـ annotations المتاحة.
- Preprocessing كامل لإشارات WFDB والـ annotations.
- Feature extraction من RR intervals ومن الإشارة نفسها.
- تدريب موديلات تقليدية ومتقدمة وهايبرد.
- تقييم داخل كل dataset، تقييم pooled، وتقييم inter-dataset generalization.

## Kaggle Inputs المطلوبة

يجب إضافة الثلاث datasets التالية إلى Kaggle Notebook بنفس المسارات:

```text
/kaggle/input/long-term-af-database
/kaggle/input/mit-bih-atrial-fibrillation
/kaggle/input/shdb-af-atrial-fibrillation/shdb-af-a-japanese-holter-ecg-database-of-atrial-fibrillation-1.0.0
```

## الموديلات الموجودة في النوتبوك

Binary prediction:

- Logistic Regression
- SVC RBF
- Random Forest
- Extra Trees
- XGBoost إذا كان متاحا في Kaggle
- LightGBM إذا كان متاحا في Kaggle
- CatBoost إذا كان متاحا في Kaggle
- Soft Voting Hybrid
- Stacking Hybrid

Rhythm classification:

- Random Forest
- Extra Trees
- XGBoost
- LightGBM
- CatBoost

## الملفات التي سيتم إخراجها

كل النتائج تحفظ في:

```text
/kaggle/working/af_full_pipeline_outputs
```

أهم الملفات:

- `record_inventory.csv`: حصر السجلات والـ annotation files.
- `af_window_features_v4_best_rhythm_source_prediction_classification.csv`: features المستخرجة لكل window بعد اختيار أفضل مصدر rhythm وفصل prediction عن classification.
- `binary_prediction_results.csv`: نتائج prediction للـ AF.
- `rhythm_classification_results.csv`: نتائج multiclass classification.
- `inter_dataset_binary_results.csv`: تدريب على dataset واختبار على dataset أخرى.
- `binary_holdout_confusion_matrix.png`: confusion matrix لأفضل موديل binary.
- `dataset_summary_plots.png`: توزيع البيانات والـ labels.
- `binary_results_plot.png`: أفضل نتائج prediction.
- `classification_results_plot.png`: أفضل نتائج classification.
- `best_binary_model.joblib`: أفضل موديل binary محفوظ.
- `experiment_summary.csv`: ملخص سريع لأفضل النتائج.

## ملاحظات تشغيل

- أول تشغيل سيكون أبطأ لأنه يستخرج features من ملفات WFDB.
- بعد أول تشغيل، النوتبوك سيقرأ ملف features الخاص بالنسخة الحالية من الكاش داخل `/kaggle/working` إذا كان موجودا.
- النسخة الحالية تنظف control characters من labels، وتجمع rhythm labels في مجموعات طبية أوضح: `NORMAL`, `AFIB`, `AFL`, `ATRIAL_TACHYCARDIA`, `OTHER`.
- النسخة الحالية تضيف fallback لـ MIT-BIH AF حتى تدخل في التجربة عندما لا تكون rhythm segments صريحة في الـ annotations.
- النسخة الحالية تفصل rhythm annotations عن beat/QRS annotations حتى لا تظهر features فاضية عند قواعد WFDB التي تستخدم ملفات annotation مختلفة.
- النسخة الحالية تختار أفضل annotation source للـ rhythm labels لكل record، ثم تستخدم beat/QRS source منفصل لاستخراج RR features.
- يمكن تقليل زمن التشغيل بتعديل `max_windows_per_record` أو `max_records_per_dataset` داخل خلية Configuration.
- المقارنة البحثية الأقوى هنا ليست accuracy فقط، بل inter-dataset validation لأنها تثبت أن الموديل لا يحفظ dataset واحدة فقط.
