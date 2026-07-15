# تشغيل AFPDB True Prediction محليا على VS Code

## الداتا المحلية

تم تجهيز النوتبوك ليقرأ الداتا من المسار المحلي التالي داخل المشروع:

```text
paraoxymal-atrial-fibrillation-prediction-database/
```

والمسار الداخلي الفعلي للإشارات:

```text
paraoxymal-atrial-fibrillation-prediction-database/
  paf-prediction-challenge-database-1.0.0/
    paf-prediction-challenge-database-1.0.0/
```

## النوتبوك

شغل النوتبوك التالي من VS Code:

```text
notebooks/03_afpdb_true_onset_prediction.ipynb
```

## خطوات التشغيل

1. افتح فولدر المشروع في VS Code:

```text
D:\Freelance work\Mohamed Magied
```

2. اختار Python kernel الخاص بالـ venv:

```text
.venv\Scripts\python.exe
```

3. لو `wfdb` غير مثبت:

```powershell
.\.venv\Scripts\python.exe -m pip install wfdb
```

أو ثبت كل المتطلبات:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

4. شغل النوتبوك Run All.

## مخرجات التشغيل المحلي

كل النتائج ستظهر هنا:

```text
outputs/afpdb_true_prediction/
```

أهم الملفات:

- `afpdb_true_prediction_features.csv`
- `afpdb_record_level_features.csv`
- `afpdb_subject_level_cv_results.csv`
- `afpdb_record_level_cv_results.csv`
- `afpdb_pairwise_paf_onset_results.csv`
- `afpdb_holdout_metrics.csv`
- `afpdb_experiment_summary.csv`
- `best_afpdb_prediction_model.joblib`
- `afpdb_holdout_confusion_matrix.png`
- `afpdb_holdout_roc_curve.png`
- `afpdb_holdout_precision_recall_curve.png`
- `afpdb_feature_importance.png`

## ملاحظة مهمة

هذا النوتبوك خاص بمهمة prediction الحقيقية:

- input: ECG segment قبل حدوث AF.
- output: هل هذا segment يسبق AF onset أم لا.

أما النوتبوك السابق الخاص بثلاث قواعد ECG فهو Detection + Rhythm Classification.

