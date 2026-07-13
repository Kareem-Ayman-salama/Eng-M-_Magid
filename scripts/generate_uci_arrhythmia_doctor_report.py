"""Generate Arabic doctor-ready report for UCI Arrhythmia experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import arabic_reshaper
import matplotlib.pyplot as plt
import pandas as pd
from bidi.algorithm import get_display
from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


@dataclass(frozen=True)
class ReportConfig:
    """Report generation configuration.

    Attributes:
        project_root: Root directory of the project.
        result_dir: Directory containing experiment CSV outputs.
        report_dir: Directory where report files are saved.
        figure_dir: Directory where generated charts are saved.
    """

    project_root: Path
    result_dir: Path
    report_dir: Path
    figure_dir: Path


def rtl(text: object) -> str:
    """Prepare Arabic text for ReportLab rendering.

    Args:
        text: Any value to render.

    Returns:
        Reshaped RTL-safe string.
    """

    return get_display(arabic_reshaper.reshape(str(text)))


def pct(value: object) -> str:
    """Format numeric metric as percentage.

    Args:
        value: Numeric metric.

    Returns:
        Percentage string, or the original text when conversion fails.
    """

    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return str(value)


def read_csv(path: Path) -> pd.DataFrame:
    """Read a required CSV file.

    Args:
        path: CSV path.

    Returns:
        Loaded dataframe.
    """

    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def prepare_percent_table(frame: pd.DataFrame, metric_columns: Iterable[str]) -> pd.DataFrame:
    """Create display copy with selected metrics formatted as percentages."""

    output = frame.copy()
    for column in metric_columns:
        if column in output.columns:
            output[column] = output[column].map(pct)
    return output


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a dataframe as a markdown table without optional dependencies."""

    string_frame = frame.astype(str)
    headers = [str(column) for column in string_frame.columns]
    rows = string_frame.values.tolist()
    widths = [
        max(len(header), *(len(str(row[index])) for row in rows))
        for index, header in enumerate(headers)
    ]
    header = "| " + " | ".join(
        value.ljust(widths[index]) for index, value in enumerate(headers)
    ) + " |"
    divider = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [
        "| " + " | ".join(
            str(value).ljust(widths[index]) for index, value in enumerate(row)
        ) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    """Build a right-aligned Arabic paragraph."""

    return Paragraph(rtl(text), style)


def pdf_table(frame: pd.DataFrame, style: ParagraphStyle, max_rows: int = 12) -> Table:
    """Build a compact RTL table for the PDF."""

    visible = frame.head(max_rows).copy()
    values = [[Paragraph(rtl(value), style) for value in reversed(visible.columns)]]
    for row in visible.astype(str).values.tolist():
        values.append([Paragraph(rtl(value), style) for value in reversed(row)])

    table = Table(values, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, -1), "Arabic"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.0),
                ("ALIGN", (0, 0), (-1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def save_bar_chart(
    frame: pd.DataFrame,
    metric: str,
    title: str,
    output_path: Path,
    top_n: int = 8,
) -> None:
    """Save a horizontal bar chart for model comparison."""

    chart_frame = frame.sort_values(metric, ascending=True).tail(top_n)
    plt.figure(figsize=(8, 4.8))
    bars = plt.barh(chart_frame["model"], chart_frame[metric] * 100, color="#2563eb")
    plt.xlabel(f"{metric} (%)")
    plt.title(title)
    plt.grid(axis="x", linestyle="--", alpha=0.35)
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.4, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%", va="center", fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def build_markdown(
    summary: pd.DataFrame,
    binary_table: pd.DataFrame,
    multiclass_table: pd.DataFrame,
    transformer_table: pd.DataFrame,
    binary_chart: Path,
    multiclass_chart: Path,
) -> str:
    """Build Arabic markdown report content."""

    best_binary = binary_table.iloc[0]
    best_multiclass = multiclass_table.iloc[0]
    transformer = transformer_table.iloc[0]

    return f"""# تقرير تحديث تجارب أمراض القلب واضطراب النظم

## الهدف من الإضافة الجديدة

تم إضافة مسار تجريبي جديد بجانب شغل Heart Disease Prediction الأساسي، والهدف منه أن المشروع لا يظل مقتصرًا على التنبؤ بوجود المرض فقط، بل يدعم أيضًا تصنيف نوع الحالة الطبية عند توفر Dataset مناسبة لذلك.

المسار الجديد يعمل على UCI Arrhythmia Dataset لأنها Dataset tabular مناسبة للتجربة السريعة داخل نفس بيئة العمل، وتحتوي على قياسات ECG وخصائص رقمية تساعد في تنفيذ مهمتين:

- Binary Prediction: التنبؤ هل الحالة Normal أم Arrhythmia.
- Grouped Multiclass Classification: تصنيف نوع الاضطراب بعد تجميع الفئات النادرة لتقليل أثر عدم توازن البيانات.

## ملخص البيانات

| البند | القيمة |
| --- | --- |
| عدد الصفوف | {int(summary.loc[0, "rows"])} |
| عدد الخصائص | {int(summary.loc[0, "features"])} |
| عدد الفئات الأصلية الموجودة | {int(summary.loc[0, "original_classes"])} |
| عدد الفئات بعد التجميع | {int(summary.loc[0, "grouped_classes"])} |
| عدد القيم المفقودة | {int(summary.loc[0, "missing_cells"])} |

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

- Model: {best_binary["model"]}
- Accuracy: {best_binary["accuracy"]}
- Balanced Accuracy: {best_binary["balanced_accuracy"]}
- F1-score: {best_binary["f1"]}
- ROC-AUC: {best_binary["roc_auc"]}

{markdown_table(binary_table)}

![Binary model comparison]({binary_chart.as_posix()})

## نتائج Grouped Multiclass Classification

أفضل نتيجة حسب Macro-F1 كانت:

- Model: {best_multiclass["model"]}
- Accuracy: {best_multiclass["accuracy"]}
- Balanced Accuracy: {best_multiclass["balanced_accuracy"]}
- Macro-F1: {best_multiclass["macro_f1"]}
- Weighted-F1: {best_multiclass["weighted_f1"]}

{markdown_table(multiclass_table)}

![Multiclass model comparison]({multiclass_chart.as_posix()})

## نتيجة Feature Tokenizer Transformer

تمت إضافة Transformer للبيانات الجدولية كاتجاه بحثي حديث، لكنه لم يتفوق على CatBoost أو Hybrid Models في هذه النسخة بسبب صغر حجم البيانات وعدم توازن الفئات.

| Metric | Value |
| --- | --- |
| Epochs | {int(transformer["epochs"])} |
| Accuracy | {transformer["accuracy"]} |
| Balanced Accuracy | {transformer["balanced_accuracy"]} |
| F1-score | {transformer["f1"]} |
| ROC-AUC | {transformer["roc_auc"]} |
| PR-AUC | {transformer["pr_auc"]} |

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
"""


def build_pdf(
    output_path: Path,
    summary: pd.DataFrame,
    binary_table: pd.DataFrame,
    multiclass_table: pd.DataFrame,
    transformer_table: pd.DataFrame,
    binary_chart: Path,
    multiclass_chart: Path,
) -> None:
    """Build the Arabic PDF report."""

    pdfmetrics.registerFont(TTFont("Arabic", "C:/Windows/Fonts/arial.ttf"))
    pdfmetrics.registerFont(TTFont("ArabicBold", "C:/Windows/Fonts/arialbd.ttf"))

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ArabicTitle",
        parent=styles["Title"],
        fontName="ArabicBold",
        fontSize=17,
        leading=23,
        alignment=TA_RIGHT,
        spaceAfter=10,
    )
    heading_style = ParagraphStyle(
        "ArabicHeading",
        parent=styles["Heading2"],
        fontName="ArabicBold",
        fontSize=12.5,
        leading=18,
        alignment=TA_RIGHT,
        spaceBefore=8,
        spaceAfter=5,
    )
    body_style = ParagraphStyle(
        "ArabicBody",
        parent=styles["BodyText"],
        fontName="Arabic",
        fontSize=10,
        leading=15,
        alignment=TA_RIGHT,
        spaceAfter=5,
    )
    table_style = ParagraphStyle(
        "ArabicTable",
        parent=body_style,
        fontSize=7.0,
        leading=9,
        alignment=TA_RIGHT,
    )

    best_binary = binary_table.iloc[0]
    best_multiclass = multiclass_table.iloc[0]
    transformer = transformer_table.iloc[0]
    dataset_rows = pd.DataFrame(
        [
            ["عدد الصفوف", int(summary.loc[0, "rows"])],
            ["عدد الخصائص", int(summary.loc[0, "features"])],
            ["الفئات الأصلية الموجودة", int(summary.loc[0, "original_classes"])],
            ["الفئات بعد التجميع", int(summary.loc[0, "grouped_classes"])],
            ["القيم المفقودة", int(summary.loc[0, "missing_cells"])],
        ],
        columns=["البند", "القيمة"],
    )
    transformer_display = pd.DataFrame(
        [
            ["Epochs", int(transformer["epochs"])],
            ["Accuracy", transformer["accuracy"]],
            ["Balanced Accuracy", transformer["balanced_accuracy"]],
            ["F1-score", transformer["f1"]],
            ["ROC-AUC", transformer["roc_auc"]],
            ["PR-AUC", transformer["pr_auc"]],
        ],
        columns=["Metric", "Value"],
    )

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=0.5 * inch,
        leftMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )
    story = [
        paragraph("تقرير تحديث تجارب أمراض القلب واضطراب النظم", title_style),
        paragraph(
            "تم إضافة مسار تجريبي جديد بجانب شغل Heart Disease Prediction الأساسي، بحيث يدعم المشروع "
            "Binary Prediction وGrouped Multiclass Classification عند توفر Dataset مناسبة.",
            body_style,
        ),
        paragraph("ملخص البيانات", heading_style),
        pdf_table(dataset_rows, table_style, max_rows=10),
        Spacer(1, 0.12 * inch),
        paragraph("ما تم تنفيذه", heading_style),
        paragraph(
            "تم تجهيز البيانات داخل Pipeline واضح، ومعالجة القيم المفقودة، واستخدام cross-validation، "
            "وتجربة Logistic Regression وRandom Forest وExtra Trees وSVC وXGBoost وLightGBM وCatBoost، "
            "بالإضافة إلى Soft Voting Hybrid وStacking Hybrid وFeature Selection وFeature Tokenizer Transformer.",
            body_style,
        ),
        paragraph("نتائج Binary Prediction", heading_style),
        paragraph(
            f"أفضل نتيجة كانت مع {best_binary['model']} بدقة {best_binary['accuracy']} "
            f"وROC-AUC {best_binary['roc_auc']}.",
            body_style,
        ),
        pdf_table(binary_table, table_style, max_rows=12),
        Spacer(1, 0.12 * inch),
        Image(str(binary_chart), width=6.8 * inch, height=4.0 * inch),
        PageBreak(),
        paragraph("نتائج Grouped Multiclass Classification", heading_style),
        paragraph(
            f"أفضل نتيجة حسب Macro-F1 كانت مع {best_multiclass['model']} "
            f"بـMacro-F1 {best_multiclass['macro_f1']} وAccuracy {best_multiclass['accuracy']}.",
            body_style,
        ),
        pdf_table(multiclass_table, table_style, max_rows=8),
        Spacer(1, 0.12 * inch),
        Image(str(multiclass_chart), width=6.8 * inch, height=4.0 * inch),
        paragraph("Feature Tokenizer Transformer", heading_style),
        paragraph(
            "تمت إضافته كاتجاه بحثي حديث للبيانات الجدولية، لكنه لم يتفوق على CatBoost أو Stacking "
            "بسبب صغر حجم البيانات وعدم توازن الفئات.",
            body_style,
        ),
        pdf_table(transformer_display, table_style, max_rows=10),
        paragraph("الإضافة العلمية", heading_style),
        paragraph(
            "الإضافة الأساسية هي توسيع المشروع من prediction فقط إلى إطار يدعم prediction وclassification، "
            "مع مقارنة نماذج boosting وhybrid وtransformer داخل نفس التقييم. هذا يجعل الاتجاه البحثي أقوى، "
            "خصوصًا عند استكماله لاحقًا على Dataset أكبر مثل PTB-XL أو Chapman ECG.",
            body_style,
        ),
        paragraph("خطة Streamlit المقترحة", heading_style),
        paragraph(
            "يمكن بناء واجهة Streamlit تسمح برفع CSV، فحص الأعمدة والقيم المفقودة، اختيار مهمة prediction "
            "أو classification، ثم عرض النتيجة والاحتمالية والثقة وأهم الخصائص المؤثرة.",
            body_style,
        ),
        paragraph("الخلاصة", heading_style),
        paragraph(
            "تم تنفيذ مسار إضافي كامل للتنبؤ والتصنيف، وإضافة Hybrid Models وFeature Tokenizer Transformer. "
            "النتائج الحالية مناسبة لإثبات الفكرة وتنظيم الاتجاه البحثي، أما المنافسة الرقمية القوية فتحتاج "
            "Dataset أكبر ومتعددة الفئات في المرحلة التالية.",
            body_style,
        ),
    ]
    document.build(story)


def main() -> None:
    """Generate markdown, figures, and PDF report."""

    project_root = Path.cwd()
    config = ReportConfig(
        project_root=project_root,
        result_dir=project_root / "outputs" / "uci_arrhythmia_advanced",
        report_dir=project_root / "reports",
        figure_dir=project_root / "reports" / "figures" / "uci_arrhythmia",
    )
    config.report_dir.mkdir(parents=True, exist_ok=True)
    config.figure_dir.mkdir(parents=True, exist_ok=True)

    summary = read_csv(config.result_dir / "dataset_summary.csv")
    binary = read_csv(config.result_dir / "advanced_binary_results.csv").sort_values(
        "roc_auc",
        ascending=False,
    )
    multiclass = read_csv(config.result_dir / "advanced_grouped_multiclass_results.csv").sort_values(
        "macro_f1",
        ascending=False,
    )
    transformer = read_csv(config.result_dir / "ft_transformer_binary_holdout.csv")

    binary_display = prepare_percent_table(
        binary[["model", "accuracy", "balanced_accuracy", "f1", "roc_auc"]],
        ["accuracy", "balanced_accuracy", "f1", "roc_auc"],
    )
    multiclass_display = prepare_percent_table(
        multiclass[["model", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]],
        ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"],
    )
    transformer_display = prepare_percent_table(
        transformer[["epochs", "accuracy", "balanced_accuracy", "f1", "roc_auc", "pr_auc"]],
        ["accuracy", "balanced_accuracy", "f1", "roc_auc", "pr_auc"],
    )

    binary_chart = config.figure_dir / "binary_roc_auc_comparison.png"
    multiclass_chart = config.figure_dir / "multiclass_macro_f1_comparison.png"
    save_bar_chart(binary, "roc_auc", "Binary Prediction ROC-AUC", binary_chart)
    save_bar_chart(multiclass, "macro_f1", "Grouped Multiclass Macro-F1", multiclass_chart)

    markdown = build_markdown(
        summary=summary,
        binary_table=binary_display,
        multiclass_table=multiclass_display,
        transformer_table=transformer_display,
        binary_chart=binary_chart,
        multiclass_chart=multiclass_chart,
    )
    markdown_path = config.report_dir / "uci_arrhythmia_prediction_classification_report_ar.md"
    pdf_path = config.report_dir / "uci_arrhythmia_prediction_classification_report_ar.pdf"
    markdown_path.write_text(markdown, encoding="utf-8")
    build_pdf(
        output_path=pdf_path,
        summary=summary,
        binary_table=binary_display,
        multiclass_table=multiclass_display,
        transformer_table=transformer_display,
        binary_chart=binary_chart,
        multiclass_chart=multiclass_chart,
    )
    print(markdown_path.resolve())
    print(pdf_path.resolve())


if __name__ == "__main__":
    main()
