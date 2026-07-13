"""Generate final Arabic optimized UCI Arrhythmia report."""

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
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


@dataclass(frozen=True)
class FinalReportConfig:
    """Final report paths."""

    project_root: Path
    advanced_dir: Path
    optimized_dir: Path
    report_dir: Path
    figure_dir: Path


def rtl(text: object) -> str:
    """Prepare Arabic text for RTL PDF rendering."""

    return get_display(arabic_reshaper.reshape(str(text)))


def pct(value: object) -> str:
    """Format metric value as percentage."""

    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return str(value)


def read_csv(path: Path) -> pd.DataFrame:
    """Read a required CSV file."""

    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def format_metrics(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Format selected metric columns as percentages."""

    output = frame.copy()
    for column in columns:
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


def save_chart(frame: pd.DataFrame, metric: str, title: str, output_path: Path) -> None:
    """Save a clean horizontal comparison chart."""

    chart_frame = frame.sort_values(metric, ascending=True).tail(8)
    plt.figure(figsize=(8.5, 4.8))
    bars = plt.barh(chart_frame["model"], chart_frame[metric] * 100, color="#0f766e")
    plt.xlabel(f"{metric} (%)")
    plt.title(title)
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.3, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%", va="center", fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def pdf_paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    """Create an RTL paragraph."""

    return Paragraph(rtl(text), style)


def pdf_table(frame: pd.DataFrame, style: ParagraphStyle, max_rows: int = 10) -> Table:
    """Create a compact PDF table."""

    visible = frame.head(max_rows)
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


def build_markdown(
    summary: pd.DataFrame,
    advanced_binary: pd.DataFrame,
    optimized_cv: pd.DataFrame,
    optimized_threshold: pd.DataFrame,
    multiclass: pd.DataFrame,
    transformer: pd.DataFrame,
    chart_path: Path,
) -> str:
    """Build final Arabic markdown report."""

    best_old = advanced_binary.iloc[0]
    best_cv = optimized_cv.iloc[0]
    best_threshold = optimized_threshold.iloc[0]
    best_multi = multiclass.iloc[0]
    ft_row = transformer.iloc[0]
    return f"""# تقرير النسخة المحسنة - UCI Arrhythmia

## الهدف

تم تثبيت العمل على UCI Arrhythmia Dataset، ثم تحسين النتائج باستخدام advanced models وhybrid models وthreshold optimization، مع الحفاظ على تقييم واضح باستخدام cross-validation بدل إدخال نتائج يدويًا.

## حجم البيانات

- عدد الحالات: {int(summary.loc[0, "rows"])}
- عدد الخصائص: {int(summary.loc[0, "features"])}
- عدد الفئات الأصلية الموجودة: {int(summary.loc[0, "original_classes"])}
- عدد الفئات بعد التجميع: {int(summary.loc[0, "grouped_classes"])}
- القيم المفقودة: {int(summary.loc[0, "missing_cells"])}

## ما تم تحسينه

- إضافة tuned CatBoost بنسختين: Conservative وDeep.
- إضافة tuned XGBoost وLightGBM مع regularization لتقليل overfitting.
- إضافة tuned Random Forest وExtra Trees.
- إضافة Advanced Soft Voting Hybrid وAdvanced Stacking Hybrid.
- إضافة threshold optimization على out-of-fold probabilities لاختيار أفضل cutoff بدل 0.5.
- الإبقاء على Feature Tokenizer Transformer كتجربة advanced حديثة.

## أفضل نتيجة قبل التحسين المتقدم

- Model: {best_old["model"]}
- Accuracy: {best_old["accuracy"]}
- Balanced Accuracy: {best_old["balanced_accuracy"]}
- F1-score: {best_old["f1"]}
- ROC-AUC: {best_old["roc_auc"]}

## أفضل نتيجة بعد التحسين

أفضل نتيجة CV حسب ROC-AUC:

- Model: {best_cv["model"]}
- Accuracy: {best_cv["accuracy"]}
- Balanced Accuracy: {best_cv["balanced_accuracy"]}
- F1-score: {best_cv["f1"]}
- ROC-AUC: {best_cv["roc_auc"]}

أفضل نتيجة بعد threshold optimization:

- Model: {best_threshold["model"]}
- Best Threshold: {best_threshold["best_threshold"]}
- Accuracy: {best_threshold["accuracy"]}
- Balanced Accuracy: {best_threshold["balanced_accuracy"]}
- F1-score: {best_threshold["f1"]}
- ROC-AUC: {best_threshold["roc_auc"]}

## جدول advanced CV

{markdown_table(optimized_cv)}

## جدول threshold optimized

{markdown_table(optimized_threshold)}

![Optimized threshold comparison]({chart_path.as_posix()})

## نتائج التصنيف متعدد الفئات

أفضل نتيجة في grouped multiclass classification:

- Model: {best_multi["model"]}
- Accuracy: {best_multi["accuracy"]}
- Balanced Accuracy: {best_multi["balanced_accuracy"]}
- Macro-F1: {best_multi["macro_f1"]}
- Weighted-F1: {best_multi["weighted_f1"]}

{markdown_table(multiclass)}

## نتيجة Feature Tokenizer Transformer

- Accuracy: {ft_row["accuracy"]}
- Balanced Accuracy: {ft_row["balanced_accuracy"]}
- F1-score: {ft_row["f1"]}
- ROC-AUC: {ft_row["roc_auc"]}

## الخلاصة

تم تحسين أفضل accuracy من {best_old["accuracy"]} إلى {best_threshold["accuracy"]} بعد استخدام tuning وthreshold optimization. كما تم تثبيت hybrid models وadvanced boosting models وTransformer baseline داخل نفس خط التجارب. النتيجة ما زالت محدودة بحجم الداتا الصغير وعدم توازن الفئات، لكنها الآن منظمة وقابلة للعرض والدفاع البحثي بشكل أفضل.
"""


def build_pdf(
    pdf_path: Path,
    markdown_tables: dict[str, pd.DataFrame],
    summary_text: list[str],
    chart_path: Path,
) -> None:
    """Build final Arabic PDF."""

    pdfmetrics.registerFont(TTFont("Arabic", "C:/Windows/Fonts/arial.ttf"))
    pdfmetrics.registerFont(TTFont("ArabicBold", "C:/Windows/Fonts/arialbd.ttf"))
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleArabic",
        parent=styles["Title"],
        fontName="ArabicBold",
        fontSize=17,
        leading=23,
        alignment=TA_RIGHT,
        spaceAfter=10,
    )
    heading_style = ParagraphStyle(
        "HeadingArabic",
        parent=styles["Heading2"],
        fontName="ArabicBold",
        fontSize=12.5,
        leading=18,
        alignment=TA_RIGHT,
        spaceBefore=8,
        spaceAfter=5,
    )
    body_style = ParagraphStyle(
        "BodyArabic",
        parent=styles["BodyText"],
        fontName="Arabic",
        fontSize=10,
        leading=15,
        alignment=TA_RIGHT,
        spaceAfter=5,
    )
    table_style = ParagraphStyle(
        "TableArabic",
        parent=body_style,
        fontSize=7,
        leading=9,
        alignment=TA_RIGHT,
    )
    document = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=0.5 * inch,
        leftMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )
    story = [pdf_paragraph("تقرير النسخة المحسنة - UCI Arrhythmia", title_style)]
    for text in summary_text:
        story.append(pdf_paragraph(text, body_style))
    story.append(pdf_paragraph("أفضل نتائج التحسين", heading_style))
    story.append(pdf_table(markdown_tables["best"], table_style, max_rows=5))
    story.append(Spacer(1, 0.12 * inch))
    story.append(Image(str(chart_path), width=6.8 * inch, height=4.0 * inch))
    story.append(pdf_paragraph("Advanced CV Results", heading_style))
    story.append(pdf_table(markdown_tables["cv"], table_style, max_rows=8))
    story.append(pdf_paragraph("Threshold Optimized Results", heading_style))
    story.append(pdf_table(markdown_tables["threshold"], table_style, max_rows=8))
    story.append(pdf_paragraph("Grouped Multiclass Results", heading_style))
    story.append(pdf_table(markdown_tables["multiclass"], table_style, max_rows=6))
    document.build(story)


def main() -> None:
    """Generate final optimized markdown and PDF reports."""

    root = Path.cwd()
    config = FinalReportConfig(
        project_root=root,
        advanced_dir=root / "outputs" / "uci_arrhythmia_advanced",
        optimized_dir=root / "outputs" / "uci_arrhythmia_optimized",
        report_dir=root / "reports",
        figure_dir=root / "reports" / "figures" / "uci_arrhythmia",
    )
    config.report_dir.mkdir(parents=True, exist_ok=True)
    config.figure_dir.mkdir(parents=True, exist_ok=True)

    summary = read_csv(config.advanced_dir / "dataset_summary.csv")
    advanced_binary = read_csv(config.advanced_dir / "advanced_binary_results.csv").sort_values("roc_auc", ascending=False)
    optimized_cv = read_csv(config.optimized_dir / "optimized_binary_cv_results.csv").sort_values("roc_auc", ascending=False)
    optimized_threshold = read_csv(config.optimized_dir / "optimized_binary_threshold_results.csv").sort_values(
        ["balanced_accuracy", "f1"],
        ascending=False,
    )
    multiclass = read_csv(config.advanced_dir / "advanced_grouped_multiclass_results.csv").sort_values("macro_f1", ascending=False)
    transformer = read_csv(config.advanced_dir / "ft_transformer_binary_holdout.csv")

    metric_columns = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]
    advanced_binary_display = format_metrics(
        advanced_binary[["model", "accuracy", "balanced_accuracy", "f1", "roc_auc"]],
        ["accuracy", "balanced_accuracy", "f1", "roc_auc"],
    )
    optimized_cv_display = format_metrics(
        optimized_cv[["model", "accuracy", "balanced_accuracy", "f1", "roc_auc"]],
        metric_columns,
    )
    optimized_threshold_display = format_metrics(
        optimized_threshold[["model", "best_threshold", "accuracy", "balanced_accuracy", "f1", "roc_auc"]],
        metric_columns,
    )
    multiclass_display = format_metrics(
        multiclass[["model", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]],
        ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"],
    )
    transformer_display = format_metrics(
        transformer[["accuracy", "balanced_accuracy", "f1", "roc_auc"]],
        ["accuracy", "balanced_accuracy", "f1", "roc_auc"],
    )

    chart_path = config.figure_dir / "optimized_threshold_balanced_accuracy.png"
    save_chart(optimized_threshold, "balanced_accuracy", "Threshold Optimized Balanced Accuracy", chart_path)

    markdown = build_markdown(
        summary=summary,
        advanced_binary=advanced_binary_display,
        optimized_cv=optimized_cv_display,
        optimized_threshold=optimized_threshold_display,
        multiclass=multiclass_display,
        transformer=transformer_display,
        chart_path=chart_path,
    )
    markdown_path = config.report_dir / "uci_arrhythmia_optimized_final_report_ar.md"
    pdf_path = config.report_dir / "uci_arrhythmia_optimized_final_report_ar.pdf"
    markdown_path.write_text(markdown, encoding="utf-8")

    best_table = pd.DataFrame(
        [
            ["قبل التحسين", advanced_binary_display.iloc[0]["model"], advanced_binary_display.iloc[0]["accuracy"], advanced_binary_display.iloc[0]["roc_auc"]],
            [
                "بعد التحسين",
                optimized_threshold_display.iloc[0]["model"],
                optimized_threshold_display.iloc[0]["accuracy"],
                optimized_threshold_display.iloc[0]["roc_auc"],
            ],
        ],
        columns=["المرحلة", "الموديل", "Accuracy", "ROC-AUC"],
    )
    build_pdf(
        pdf_path=pdf_path,
        markdown_tables={
            "best": best_table,
            "cv": optimized_cv_display,
            "threshold": optimized_threshold_display,
            "multiclass": multiclass_display,
        },
        summary_text=[
            "تم تثبيت العمل على UCI Arrhythmia Dataset وتحسين النتائج باستخدام advanced boosting models وhybrid models.",
            "تم استخدام threshold optimization على out-of-fold probabilities لتحسين قرار التصنيف بدون إدخال نتائج يدويًا.",
            f"حجم البيانات: {int(summary.loc[0, 'rows'])} حالة و{int(summary.loc[0, 'features'])} خاصية.",
        ],
        chart_path=chart_path,
    )
    print(markdown_path.resolve())
    print(pdf_path.resolve())


if __name__ == "__main__":
    main()
