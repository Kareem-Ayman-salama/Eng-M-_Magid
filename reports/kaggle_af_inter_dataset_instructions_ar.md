# تعليمات تشغيل Notebook داتا AF على Kaggle

## الملف

Notebook:

`notebooks/01_kaggle_af_inter_dataset_experiments.ipynb`

## تضيف إيه في Kaggle Input

من صفحة Kaggle Notebook اضغط **Add Input** وأضف:

1. `yahiahedna/shdb-af-atrial-fibrillation`
2. `sahilshahare12/mit-bih-atrial-fibrillation`
3. `sahilshahare12/long-term-af-database`

لو المساحة أو وقت التشغيل قليل، ابدأ بـ:

1. `sahilshahare12/mit-bih-atrial-fibrillation`
2. `sahilshahare12/long-term-af-database`

ثم أضف SHDB-AF لاحقًا لأنها الأكبر.

## الفكرة

نترك شغل UCI القديم كما هو كـ tabular baseline، ونضيف Notebook جديدة لتجربة ECG signal/annotation datasets.

## المهام داخل النوتبوك

- اكتشاف كل ملفات `.hea`, `.atr`, `.qrs`, `.dat` تلقائيًا من `/kaggle/input`.
- استخراج RR interval features من annotations بدل تحميل الإشارة كاملة في البداية.
- Binary prediction:
  - AF vs non-AF.
- Rhythm classification:
  - AFIB / AFL / AT / OTHER حسب labels المتاحة.
- Inter-dataset validation:
  - تدريب على Dataset واختبار على Dataset أخرى.

## ليه ده أقوى بحثيًا

Inter-dataset validation يقيس قدرة النموذج على التعميم بين قواعد بيانات ECG مختلفة، وده أقوى من تدريب واختبار على نفس dataset فقط.

## ملاحظات

- لو Dataset فيها AF-positive فقط، binary classification داخل نفس الداتا مش هيكون مفيد.
- في الحالة دي نستخدمها في rhythm classification أو كـ external validation.
- أول نسخة تعتمد على annotation/RR features لأنها أخف وأسرع على Kaggle.
- بعد النتائج الأولية ممكن نضيف CNN/Transformer على raw ECG windows.
