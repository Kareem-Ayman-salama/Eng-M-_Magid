# شكل الـ Output المقترح للمستخدم في تطبيق Streamlit

## الهدف

التطبيق يكون موجه لطبيب/باحث أو مستخدم يرفع ملف CSV، ثم يحصل على نتيجة واضحة: هل الحالة Normal أم Arrhythmia، أو تصنيف نوع الاضطراب عند تشغيل multiclass classification.

## الشاشة الرئيسية

### 1. رفع البيانات

العناصر الظاهرة:

- زر رفع ملف CSV.
- جدول معاينة أول 5 صفوف.
- ملخص سريع:
  - عدد الصفوف.
  - عدد الأعمدة.
  - عدد القيم المفقودة.
  - هل الأعمدة متوافقة مع نموذج UCI Arrhythmia أم لا.

مثال Output:

| Item | Value |
| --- | --- |
| Uploaded rows | 12 |
| Uploaded features | 279 |
| Missing values | 4 |
| Schema status | Compatible |

## اختيار نوع المهمة

### 2. Task Mode

المستخدم يختار واحد من:

- Binary Prediction: Normal vs Arrhythmia.
- Grouped Multiclass Classification: نوع اضطراب النظم.

لو الأعمدة غير متوافقة، التطبيق يظهر رسالة:

> The uploaded dataset does not match the required 279-feature schema.

## Output في حالة Binary Prediction

### 3. Prediction Summary

لكل حالة في الملف، يظهر جدول:

| Patient ID | Predicted Class | Arrhythmia Probability | Confidence | Risk Level |
| --- | --- | --- | --- | --- |
| 1 | Arrhythmia | 0.82 | High | High Risk |
| 2 | Normal | 0.21 | Medium | Low Risk |

### 4. حالة واحدة بالتفصيل

عند اختيار حالة من الجدول، يظهر:

- Predicted class: Arrhythmia أو Normal.
- Probability of Arrhythmia.
- Confidence score.
- Recommended decision threshold.
- Model used.

مثال:

| Field | Value |
| --- | --- |
| Prediction | Arrhythmia |
| Probability | 82% |
| Confidence | High |
| Threshold | 0.49 |
| Model | CatBoost Deep |

### 5. Risk Card

كارت مختصر:

```text
Prediction: Arrhythmia
Probability: 82%
Risk Level: High
Model Confidence: High
```

## Output في حالة Multiclass Classification

### 6. Multiclass Result

لو المستخدم اختار classification، يظهر:

| Patient ID | Predicted Group | Group Probability | Confidence |
| --- | --- | --- | --- |
| 1 | Class 10 | 0.61 | Medium |
| 2 | Normal | 0.78 | High |

مع ملاحظة واضحة:

> Multiclass output is grouped because rare arrhythmia classes have very few samples in the UCI dataset.

## Explainability Output

### 7. أهم الخصائص المؤثرة

يظهر للمستخدم جدول أو bar chart:

| Feature | Impact |
| --- | --- |
| feature_090 | High |
| feature_004 | Medium |
| feature_176 | Medium |

في النسخة الأولى ممكن نستخدم feature importance من CatBoost/Extra Trees، وبعدها نضيف SHAP.

## Batch Summary

### 8. ملخص الملف المرفوع

لو المستخدم رفع أكثر من حالة:

| Metric | Value |
| --- | --- |
| Total cases | 12 |
| Predicted Normal | 5 |
| Predicted Arrhythmia | 7 |
| Average Arrhythmia Probability | 64% |
| High Risk Cases | 4 |

## Charts

الرسوم المقترحة:

- Pie chart: Normal vs Arrhythmia count.
- Histogram: Probability distribution.
- Bar chart: Top influential features.
- Confusion matrix في صفحة evaluation لو المستخدم رفع target column.

## Evaluation Mode

لو الملف المرفوع يحتوي على target column، نعرض:

| Metric | Value |
| --- | --- |
| Accuracy | 84.51% |
| Balanced Accuracy | 84.14% |
| F1-score | 82.50% |
| ROC-AUC | 89.44% |

ونعرض:

- Confusion matrix.
- ROC curve.
- PR curve.
- Classification report.

## أفضل شكل للتطبيق

التطبيق يكون فيه 3 tabs:

1. Upload & Validate
2. Prediction Results
3. Model Evaluation

وفي sidebar:

- Task mode.
- Model selection.
- Decision threshold.
- Download results button.

## Output النهائي القابل للتحميل

المستخدم يقدر ينزل CSV فيه:

| patient_id | predicted_class | probability_arrhythmia | confidence | risk_level | model_name |
| --- | --- | --- | --- | --- | --- |
| 1 | Arrhythmia | 0.82 | High | High Risk | CatBoost Deep |

## ملاحظات مهمة

- التطبيق لا يقدم تشخيص طبي نهائي.
- النتيجة تكون decision-support فقط.
- لازم يظهر تنبيه أن النموذج مدرب على UCI Arrhythmia وهي dataset صغيرة.
- عند استخدام بيانات خارجية، يجب التأكد من توافق الأعمدة مع schema التدريب.
