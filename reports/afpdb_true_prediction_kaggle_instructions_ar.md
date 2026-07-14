# تعليمات نوتبوك True AF Onset Prediction

## اسم الداتا

الاسم الرسمي:

```text
PAF Prediction Challenge Database (AFPDB)
```

على PhysioNet:

```text
https://physionet.org/content/afpdb/1.0.0/
```

على Kaggle ابحث عن:

```text
Paraoxymal Atrial Fibrillation Prediction Database
```

ملاحظة: اسم Kaggle فيه خطأ إملائي في كلمة `Paraoxymal` بدلا من `Paroxysmal`.

المسار المتوقع إذا أضفتها إلى Kaggle:

```text
/kaggle/input/paraoxymal-atrial-fibrillation-prediction-database
```

## النوتبوك

النوتبوك:

```text
notebooks/03_afpdb_true_onset_prediction.ipynb
```

## الهدف البحثي

هذا النوتبوك يضيف مهمة prediction حقيقية مختلفة عن detection:

- Detection: هل النافذة الحالية تحتوي AF؟
- True prediction: هل نافذة ECG الحالية، قبل ظهور AF، تسبق episode قريبة من Paroxysmal AF؟

## Labels المستخدمة

يعتمد النوتبوك على قواعد AFPDB:

- `p` records: من مرضى لديهم PAF.
- الرقم الزوجي من `p` records يمثل ECG قبل onset مباشرة.
- الرقم الفردي من `p` records يمثل segment بعيد عن onset.
- `n` records تمثل negative/no imminent PAF.
- `t` records يتم تقييمها إذا كانت `event-2-answers` موجودة.

يتم استبعاد continuation records المنتهية بـ `c` لأنها قد تحتوي الحدث نفسه وليست pre-onset input.

## المخرجات

كل النتائج تحفظ في:

```text
/kaggle/working/afpdb_true_prediction_outputs
```

أهم الملفات:

- `afpdb_record_inventory.csv`
- `afpdb_true_prediction_features.csv`
- `afpdb_subject_level_cv_results.csv`
- `afpdb_challenge_test_results.csv`
- `afpdb_holdout_metrics.csv`
- `afpdb_experiment_summary.csv`
- `best_afpdb_prediction_model.joblib`
- `afpdb_holdout_confusion_matrix.png`
- `afpdb_holdout_roc_curve.png`
- `afpdb_holdout_precision_recall_curve.png`
- `afpdb_feature_importance.png`

## ملاحظات مهمة

- إذا لم تضف الداتا على Kaggle، سيحاول النوتبوك تنزيلها من PhysioNet باستخدام `wfdb.dl_database`.
- لو Kaggle internet مغلق، يجب إضافة الداتا كـ input.
- التقييم يستخدم subject-level grouping لتقليل data leakage.
- هذه التجربة يجب عرضها كـ true onset prediction، بينما النوتبوك السابق يعرض AF detection و rhythm classification.

