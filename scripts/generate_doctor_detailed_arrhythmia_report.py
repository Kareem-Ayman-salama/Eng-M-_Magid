"""Generate detailed Arabic PDF report for the arrhythmia project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
class DoctorReportConfig:
    """Configuration for the doctor report.

    Attributes:
        project_root: Project root path.
        report_dir: Report output directory.
        figure_dir: Generated figures directory.
    """

    project_root: Path
    report_dir: Path
    figure_dir: Path


def rtl(text: object) -> str:
    """Prepare Arabic text for ReportLab RTL rendering."""

    return get_display(arabic_reshaper.reshape(str(text)))


def pct(value: object) -> str:
    """Format a metric as a percentage string."""

    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return str(value)


def read_csv(path: Path) -> pd.DataFrame:
    """Read a required CSV file."""

    if not path.exists():
        raise FileNotFoundError(f"Missing required result file: {path}")
    return pd.read_csv(path)


def format_table(frame: pd.DataFrame, percent_columns: list[str]) -> pd.DataFrame:
    """Format selected metric columns as percentages."""

    output = frame.copy()
    for column in percent_columns:
        if column in output.columns:
            output[column] = output[column].map(pct)
    if "best_threshold" in output.columns:
        output["best_threshold"] = output["best_threshold"].map(lambda value: f"{float(value):.3f}")
    return output


def markdown_table(frame: pd.DataFrame) -> str:
    """Convert a dataframe to a markdown table."""

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
    """Build an RTL paragraph."""

    return Paragraph(rtl(text), style)


def pdf_table(frame: pd.DataFrame, style: ParagraphStyle, max_rows: int = 10) -> Table:
    """Build a compact right-to-left PDF table."""

    visible = frame.head(max_rows).copy()
    rows = [[Paragraph(rtl(value), style) for value in reversed(visible.columns)]]
    for row in visible.astype(str).values.tolist():
        rows.append([Paragraph(rtl(value), style) for value in reversed(row)])

    table = Table(rows, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, -1), "Arabic"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
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


def save_bar_chart(frame: pd.DataFrame, metric: str, title: str, output_path: Path) -> None:
    """Save a horizontal bar chart for model comparison."""

    chart_frame = frame.sort_values(metric, ascending=True).tail(8)
    plt.figure(figsize=(8.8, 4.8))
    bars = plt.barh(chart_frame["model"], chart_frame[metric] * 100, color="#2563eb")
    plt.xlabel(f"{metric} (%)")
    plt.title(title)
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.35, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%", va="center", fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_output_flow_chart(output_path: Path) -> None:
    """Save a simple product-output workflow chart."""

    labels = [
        "Upload CSV",
        "Validate schema",
        "Select task",
        "Run model",
        "Show prediction",
        "Download results",
    ]
    x_values = range(len(labels))
    plt.figure(figsize=(9, 2.8))
    plt.plot(list(x_values), [1] * len(labels), color="#0f766e", linewidth=2)
    plt.scatter(list(x_values), [1] * len(labels), s=520, color="#0f766e")
    for index, label in enumerate(labels):
        plt.text(index, 1, str(index + 1), color="white", ha="center", va="center", fontsize=12, weight="bold")
        plt.text(index, 0.72, label, ha="center", va="top", fontsize=9)
    plt.ylim(0.45, 1.35)
    plt.axis("off")
    plt.title("Streamlit Output Flow")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def build_markdown(
    summary: pd.DataFrame,
    best_binary: pd.Series,
    binary_results: pd.DataFrame,
    multiclass_results: pd.DataFrame,
    resampling_results: pd.DataFrame,
    transformer: pd.Series,
    binary_chart: Path,
    multiclass_chart: Path,
    flow_chart: Path,
) -> str:
    """Build the markdown version of the report."""

    return f"""# تقرير تفصيلي للدكتور - Arrhythmia Prediction and Disease Classification

## 1. الهدف من الشغل

الهدف الحالي هو بناء إطار عملي وبحثي على UCI Arrhythmia Dataset يدعم مهمتين أساسيتين:

- Binary prediction: تحديد هل الحالة Normal أم Arrhythmia.
- Disease classification: تصنيف نوع اضطراب النظم، مع تجميع الفئات النادرة بسبب قلة العينات.

## 2. البيانات المتاحة

| البند | القيمة |
| --- | --- |
| Dataset | UCI Arrhythmia |
| عدد الحالات | {int(summary.loc[0, "rows"])} |
| عدد الخصائص | {int(summary.loc[0, "features"])} |
| عدد الفئات الأصلية الموجودة | {int(summary.loc[0, "original_classes"])} |
| عدد الفئات بعد التجميع | {int(summary.loc[0, "grouped_classes"])} |
| القيم المفقودة | {int(summary.loc[0, "missing_cells"])} |

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
| Best model | {best_binary["model"]} |
| Best threshold | {best_binary["best_threshold"]} |
| Accuracy | {best_binary["accuracy"]} |
| Balanced Accuracy | {best_binary["balanced_accuracy"]} |
| F1-score | {best_binary["f1"]} |
| ROC-AUC | {best_binary["roc_auc"]} |

{markdown_table(binary_results)}

![Binary comparison]({binary_chart.as_posix()})

## 5. نتائج Disease Classification

تم تنفيذ classification للمرض كـ grouped multiclass classification. سبب التجميع أن بعض فئات المرض في الداتا الأصلية تحتوي على 2 أو 3 عينات فقط، وهذا غير كاف لتقييم موثوق.

{markdown_table(multiclass_results)}

![Multiclass comparison]({multiclass_chart.as_posix()})

## 6. نتائج SMOTE وFeature Selection

تمت تجربة SMOTE مع feature selection داخل cross-validation. النتيجة لم تتفوق على CatBoost Deep الأساسي، ولذلك تم اعتبارها تجربة بحثية مساعدة وليست النموذج النهائي.

{markdown_table(resampling_results)}

## 7. نتيجة Transformer

| Metric | Value |
| --- | --- |
| Model | Feature Tokenizer Transformer |
| Accuracy | {transformer["accuracy"]} |
| Balanced Accuracy | {transformer["balanced_accuracy"]} |
| F1-score | {transformer["f1"]} |
| ROC-AUC | {transformer["roc_auc"]} |

## 8. شكل الـ Output للمستخدم

التطبيق المقترح باستخدام Streamlit سيعرض للمستخدم:

- Upload CSV.
- Validate schema وعدد الصفوف والأعمدة والقيم المفقودة.
- اختيار نوع المهمة: Prediction أو Classification.
- جدول نتائج يحتوي على predicted class وprobability وconfidence وrisk level.
- عند وجود target column يتم عرض Accuracy وF1-score وROC-AUC وConfusion Matrix.
- إمكانية تنزيل النتائج كملف CSV.

![Output flow]({flow_chart.as_posix()})

## 9. الخلاصة

المشروع حاليًا يحتوي على مسارين واضحين: prediction لوجود Arrhythmia، وclassification مجمع لأنواع المرض. أفضل نتيجة حالية في binary prediction هي Accuracy = {best_binary["accuracy"]} وROC-AUC = {best_binary["roc_auc"]}. أما disease classification فهي أصعب بسبب قلة العينات وعدم توازن الفئات، ولذلك تم استخدام grouped classes للحصول على تقييم أكثر منطقية.
"""


def build_pdf(
    output_path: Path,
    summary: pd.DataFrame,
    best_binary: pd.Series,
    binary_results: pd.DataFrame,
    multiclass_results: pd.DataFrame,
    resampling_results: pd.DataFrame,
    transformer: pd.Series,
    binary_chart: Path,
    multiclass_chart: Path,
    flow_chart: Path,
) -> None:
    """Build the final detailed PDF report."""

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
        spaceAfter=8,
    )
    heading_style = ParagraphStyle(
        "ArabicHeading",
        parent=styles["Heading2"],
        fontName="ArabicBold",
        fontSize=12.5,
        leading=17,
        alignment=TA_RIGHT,
        spaceBefore=8,
        spaceAfter=5,
    )
    body_style = ParagraphStyle(
        "ArabicBody",
        parent=styles["BodyText"],
        fontName="Arabic",
        fontSize=9.8,
        leading=14.5,
        alignment=TA_RIGHT,
        spaceAfter=5,
    )
    table_style = ParagraphStyle(
        "ArabicTable",
        parent=body_style,
        fontSize=6.8,
        leading=8.8,
        alignment=TA_RIGHT,
    )

    dataset_table = pd.DataFrame(
        [
            ["Dataset", "UCI Arrhythmia"],
            ["عدد الحالات", int(summary.loc[0, "rows"])],
            ["عدد الخصائص", int(summary.loc[0, "features"])],
            ["الفئات الأصلية", int(summary.loc[0, "original_classes"])],
            ["الفئات بعد التجميع", int(summary.loc[0, "grouped_classes"])],
            ["القيم المفقودة", int(summary.loc[0, "missing_cells"])],
        ],
        columns=["البند", "القيمة"],
    )
    best_table = pd.DataFrame(
        [
            ["Best model", best_binary["model"]],
            ["Best threshold", best_binary["best_threshold"]],
            ["Accuracy", best_binary["accuracy"]],
            ["Balanced Accuracy", best_binary["balanced_accuracy"]],
            ["F1-score", best_binary["f1"]],
            ["ROC-AUC", best_binary["roc_auc"]],
        ],
        columns=["Metric", "Value"],
    )
    transformer_table = pd.DataFrame(
        [
            ["Model", "Feature Tokenizer Transformer"],
            ["Accuracy", transformer["accuracy"]],
            ["Balanced Accuracy", transformer["balanced_accuracy"]],
            ["F1-score", transformer["f1"]],
            ["ROC-AUC", transformer["roc_auc"]],
        ],
        columns=["Metric", "Value"],
    )
    output_table = pd.DataFrame(
        [
            ["Upload & Validate", "رفع CSV وفحص عدد الصفوف والأعمدة والقيم المفقودة"],
            ["Task Mode", "اختيار Prediction أو Classification"],
            ["Prediction Output", "predicted class, probability, confidence, risk level"],
            ["Evaluation Output", "Accuracy, F1-score, ROC-AUC, Confusion Matrix عند وجود target"],
            ["Download", "تنزيل النتائج كملف CSV"],
        ],
        columns=["Section", "User Output"],
    )

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=0.45 * inch,
        leftMargin=0.45 * inch,
        topMargin=0.45 * inch,
        bottomMargin=0.45 * inch,
    )
    story = [
        paragraph("تقرير تفصيلي للدكتور - Arrhythmia Prediction and Classification", title_style),
        paragraph("الهدف من الشغل", heading_style),
        paragraph(
            "تم بناء إطار بحثي وعملي على UCI Arrhythmia Dataset يدعم مسارين: "
            "Binary Prediction لتحديد Normal أو Arrhythmia، وDisease Classification لتصنيف نوع الاضطراب "
            "بعد تجميع الفئات النادرة بسبب قلة العينات.",
            body_style,
        ),
        paragraph("البيانات المتاحة", heading_style),
        pdf_table(dataset_table, table_style, max_rows=8),
        paragraph("ما تم تنفيذه", heading_style),
        paragraph(
            "تم تجهيز البيانات داخل Pipeline، معالجة missing values، استخدام Stratified Cross-Validation، "
            "وتجربة Logistic Regression وSVC وRandom Forest وExtra Trees وXGBoost وLightGBM وCatBoost. "
            "كما تم بناء Soft Voting وStacking Hybrid وإضافة Feature Tokenizer Transformer وتجربة SMOTE "
            "مع feature selection داخل cross-validation بدون data leakage.",
            body_style,
        ),
        paragraph("أفضل نتيجة Binary Prediction", heading_style),
        pdf_table(best_table, table_style, max_rows=8),
        Spacer(1, 0.1 * inch),
        pdf_table(binary_results, table_style, max_rows=8),
        Spacer(1, 0.1 * inch),
        Image(str(binary_chart), width=6.8 * inch, height=3.7 * inch),
        PageBreak(),
        paragraph("Disease Classification", heading_style),
        paragraph(
            "الداتا الأصلية تحتوي على عدة فئات لاضطراب النظم، لكن بعض الفئات قليلة جدًا. لذلك تم استخدام "
            "grouped multiclass classification حتى يكون التقييم أكثر منطقية بدل تدريب نموذج على فئات بها "
            "عينات نادرة جدًا.",
            body_style,
        ),
        pdf_table(multiclass_results, table_style, max_rows=8),
        Spacer(1, 0.1 * inch),
        Image(str(multiclass_chart), width=6.8 * inch, height=3.7 * inch),
        paragraph("SMOTE وFeature Selection", heading_style),
        paragraph(
            "تمت تجربة SMOTE مع عدة طرق feature selection داخل cross-validation. هذه التجربة لم تتفوق على "
            "CatBoost Deep، لذلك تم توثيقها كتجربة تحسين بحثية وليست النموذج النهائي.",
            body_style,
        ),
        pdf_table(resampling_results, table_style, max_rows=6),
        paragraph("Feature Tokenizer Transformer", heading_style),
        pdf_table(transformer_table, table_style, max_rows=6),
        PageBreak(),
        paragraph("شكل الـ Output للمستخدم", heading_style),
        paragraph(
            "في تطبيق Streamlit، المستخدم سيرفع CSV، ثم يتم فحص توافق الأعمدة، وبعدها يختار نوع المهمة. "
            "في النهاية يحصل على جدول نتائج واحتمالات وثقة النموذج ومستوى الخطورة، مع إمكانية تنزيل النتائج.",
            body_style,
        ),
        pdf_table(output_table, table_style, max_rows=8),
        Spacer(1, 0.1 * inch),
        Image(str(flow_chart), width=6.8 * inch, height=2.25 * inch),
        paragraph("الخلاصة", heading_style),
        paragraph(
            f"أفضل نتيجة حالية في Binary Prediction هي Accuracy = {best_binary['accuracy']} "
            f"وROC-AUC = {best_binary['roc_auc']} باستخدام {best_binary['model']}. "
            "أما Disease Classification فهي موجودة لكنها grouped بسبب عدم توازن الفئات وصغر حجم الداتا. "
            "شكل الـ output النهائي واضح للمستخدم ويغطي upload, validation, prediction, evaluation, and download.",
            body_style,
        ),
    ]
    document.build(story)


def main() -> None:
    """Generate markdown, charts, and PDF report."""

    root = Path.cwd()
    config = DoctorReportConfig(
        project_root=root,
        report_dir=root / "reports",
        figure_dir=root / "reports" / "figures" / "doctor_detailed_report",
    )
    config.report_dir.mkdir(parents=True, exist_ok=True)
    config.figure_dir.mkdir(parents=True, exist_ok=True)

    summary = read_csv(root / "outputs" / "uci_arrhythmia_advanced" / "dataset_summary.csv")
    binary_raw = read_csv(root / "outputs" / "uci_arrhythmia_optimized" / "optimized_binary_threshold_results.csv")
    multiclass_raw = read_csv(root / "outputs" / "uci_arrhythmia_advanced" / "advanced_grouped_multiclass_results.csv")
    resampling_raw = read_csv(root / "outputs" / "uci_arrhythmia_resampling" / "resampling_feature_selection_threshold_results.csv")
    transformer_raw = read_csv(root / "outputs" / "uci_arrhythmia_advanced" / "ft_transformer_binary_holdout.csv")

    binary = binary_raw.sort_values(["balanced_accuracy", "f1"], ascending=False)
    multiclass = multiclass_raw.sort_values("macro_f1", ascending=False)
    resampling = resampling_raw.sort_values(["balanced_accuracy", "f1"], ascending=False)

    binary_display = format_table(
        binary[["model", "best_threshold", "accuracy", "balanced_accuracy", "f1", "roc_auc"]].head(8),
        ["accuracy", "balanced_accuracy", "f1", "roc_auc"],
    )
    multiclass_display = format_table(
        multiclass[["model", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]],
        ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"],
    )
    resampling_display = format_table(
        resampling[["model", "best_threshold", "accuracy", "balanced_accuracy", "f1", "roc_auc"]].head(6),
        ["accuracy", "balanced_accuracy", "f1", "roc_auc"],
    )
    transformer_display = format_table(
        transformer_raw[["accuracy", "balanced_accuracy", "f1", "roc_auc"]],
        ["accuracy", "balanced_accuracy", "f1", "roc_auc"],
    )
    best_binary = binary_display.iloc[0]
    transformer = transformer_display.iloc[0]

    binary_chart = config.figure_dir / "binary_best_models.png"
    multiclass_chart = config.figure_dir / "classification_best_models.png"
    flow_chart = config.figure_dir / "streamlit_output_flow.png"
    save_bar_chart(binary.head(8), "balanced_accuracy", "Binary Prediction - Balanced Accuracy", binary_chart)
    save_bar_chart(multiclass, "macro_f1", "Disease Classification - Macro F1", multiclass_chart)
    save_output_flow_chart(flow_chart)

    markdown = build_markdown(
        summary=summary,
        best_binary=best_binary,
        binary_results=binary_display,
        multiclass_results=multiclass_display,
        resampling_results=resampling_display,
        transformer=transformer,
        binary_chart=binary_chart,
        multiclass_chart=multiclass_chart,
        flow_chart=flow_chart,
    )
    markdown_path = config.report_dir / "doctor_detailed_arrhythmia_report_ar.md"
    pdf_path = config.report_dir / "doctor_detailed_arrhythmia_report_ar.pdf"
    markdown_path.write_text(markdown, encoding="utf-8")
    build_pdf(
        output_path=pdf_path,
        summary=summary,
        best_binary=best_binary,
        binary_results=binary_display,
        multiclass_results=multiclass_display,
        resampling_results=resampling_display,
        transformer=transformer,
        binary_chart=binary_chart,
        multiclass_chart=multiclass_chart,
        flow_chart=flow_chart,
    )
    print(markdown_path.resolve())
    print(pdf_path.resolve())


if __name__ == "__main__":
    main()
