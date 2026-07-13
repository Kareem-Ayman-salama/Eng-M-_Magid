"""Generate a concise Arabic update summary PDF for the heart disease work."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import arabic_reshaper
import pandas as pd
from bidi.algorithm import get_display
from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


@dataclass(frozen=True)
class SummaryConfig:
    """PDF generation configuration.

    Attributes:
        project_root: Project root path.
        results_dir: Directory containing generated experiment outputs.
        output_dir: Directory where the report is saved.
        font_regular: Arabic-capable regular font path.
        font_bold: Arabic-capable bold font path.
    """

    project_root: Path
    results_dir: Path
    output_dir: Path
    font_regular: Path
    font_bold: Path


def rtl(text: object) -> str:
    """Prepare Arabic text for ReportLab rendering."""

    return get_display(arabic_reshaper.reshape(str(text)))


def pct(value: float) -> str:
    """Format decimal metric as percentage."""

    return f"{value * 100:.2f}%"


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    """Create a right-to-left paragraph."""

    return Paragraph(rtl(text), style)


def bullet(text: str, style: ParagraphStyle) -> Paragraph:
    """Create a right-to-left bullet paragraph."""

    return paragraph(f"- {text}", style)


def read_csv(path: Path) -> pd.DataFrame:
    """Read a required CSV file."""

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def make_table(
    rows: list[list[str]],
    header: list[str],
    table_style: ParagraphStyle,
) -> Table:
    """Create a styled RTL table."""

    values = [[Paragraph(rtl(value), table_style) for value in reversed(header)]]
    for row in rows:
        values.append([Paragraph(rtl(value), table_style) for value in reversed(row)])

    table = Table(values, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, -1), "Arabic"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("ALIGN", (0, 0), (-1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def build_summary_pdf(config: SummaryConfig) -> Path:
    """Build the organized Arabic update summary PDF."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.output_dir / "heart_disease_project_update_summary_ar.pdf"

    tuned = read_csv(config.results_dir / "openml_tuned_repeated_cv_results.csv")
    holdout = read_csv(config.results_dir / "final_holdout_results.csv")
    feature_selection = read_csv(config.results_dir / "feature_selection_repeated_cv_results.csv")
    ft_results = read_csv(config.results_dir / "ft_transformer_holdout_results.csv")
    published = read_csv(config.results_dir / "published_work_comparison.csv")

    best_accuracy = tuned.sort_values("accuracy", ascending=False).iloc[0]
    best_auc = tuned.sort_values("roc_auc", ascending=False).iloc[0]
    best_holdout = holdout.sort_values("roc_auc", ascending=False).iloc[0]
    best_feature_selection = feature_selection.sort_values("roc_auc", ascending=False).iloc[0]
    ft_row = ft_results.iloc[0]

    pdfmetrics.registerFont(TTFont("Arabic", str(config.font_regular)))
    pdfmetrics.registerFont(TTFont("ArabicBold", str(config.font_bold)))

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleArabic",
        parent=styles["Title"],
        fontName="ArabicBold",
        fontSize=19,
        leading=25,
        alignment=TA_RIGHT,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=12,
    )
    heading_style = ParagraphStyle(
        "HeadingArabic",
        parent=styles["Heading2"],
        fontName="ArabicBold",
        fontSize=13,
        leading=18,
        alignment=TA_RIGHT,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=9,
        spaceAfter=5,
    )
    body_style = ParagraphStyle(
        "BodyArabic",
        parent=styles["BodyText"],
        fontName="Arabic",
        fontSize=10,
        leading=15,
        alignment=TA_RIGHT,
        textColor=colors.HexColor("#1f2937"),
        spaceAfter=5,
    )
    table_style = ParagraphStyle(
        "TableArabic",
        parent=body_style,
        fontSize=7.5,
        leading=10,
        alignment=TA_RIGHT,
    )

    story = [
        paragraph("تقرير تحديث مشروع التنبؤ بأمراض القلب", title_style),
        paragraph(
            "يوضح هذا التقرير ملاحظات الدكتور على الشغل، وكيف تم التعامل معها، "
            "وما الإضافات التي تم تنفيذها داخل النوتبوك والتقارير مع النتائج الأساسية.",
            body_style,
        ),
        Spacer(1, 0.08 * inch),
        paragraph("1. ملاحظات الدكتور وكيف تم حلها", heading_style),
        make_table(
            rows=[
                [
                    "استخدام أكثر من Dataset لا يكفي كـ contribution.",
                    "تمت إعادة صياغة الإسهام البحثي بحيث تصبح الـ datasets وسيلة validation وليست الإضافة الأساسية.",
                    "تمت إضافة Research Contribution Matrix داخل النوتبوك والتقرير.",
                ],
                [
                    "لازم يكون فيه technique أو framework جديد واضح.",
                    "تم تقديم الشغل كـ Unified Explainable Hybrid Framework يجمع بين النماذج الهجينة وSHAP وFT-Transformer وFeature Selection.",
                    "تمت إضافة قسم Research Contribution Positioning.",
                ],
                [
                    "المقارنة مع المنشور لا ينفع تكون كلام فقط.",
                    "تمت إضافة جدول Comparison With Published Work يضم أبحاث منشورة ونتائجنا.",
                    "تم إنشاء published_work_comparison.csv وإضافته للنوتبوك والتقارير.",
                ],
                [
                    "المقارنة لازم توضح هل هي على نفس Dataset أم لا.",
                    "تمت إضافة عمود comparison_type لتوضيح هل المقارنة مباشرة أو غير مباشرة بسبب اختلاف الداتا أو البروتوكول.",
                    "تمت إضافة ملاحظة منهجية عن شروط المقارنة المباشرة.",
                ],
                [
                    "لازم الشغل يبين إضافة بحثية فعلية.",
                    "تم ربط الإضافة البحثية بالهايبرد موديل، قابلية التفسير، FT-Transformer، Feature Selection، والتقييم القوي.",
                    "تم تحديث التقرير العربي والإنجليزي والنوتبوك المتنفذة.",
                ],
            ],
            header=["ملاحظة الدكتور", "طريقة الحل", "ما تم إضافته"],
            table_style=table_style,
        ),
        paragraph("2. الهدف من التحديث", heading_style),
        paragraph(
            "تم تعديل طريقة عرض المشروع بحيث لا يتم تقديم استخدام أكثر من Dataset كإسهام بحثي مستقل، "
            "بل يتم اعتباره وسيلة تحقق من ثبات الإطار المقترح. الإسهام الأساسي أصبح إطارًا موحدًا قابلًا "
            "للتفسير يجمع بين النماذج الهجينة، SHAP، Feature Selection، وFeature Tokenizer Transformer.",
            body_style,
        ),
        paragraph("3. الإضافات التي تمت على الشغل", heading_style),
    ]

    for item in [
        "إضافة جدول مقارنة مع الأعمال المنشورة داخل النوتبوك.",
        "إضافة جدول يوضح الإسهام البحثي الحقيقي وكيف يعالج كل ملاحظة.",
        "إعادة صياغة الـ contribution بحيث لا تعتمد على multiple datasets فقط.",
        "نماذج boosting قوية مثل CatBoost وXGBoost وLightGBM.",
        "نماذج هجينة تشمل Soft Voting وStacking.",
        "تفسير قرارات النموذج باستخدام SHAP.",
        "تحليل اختيار الخصائص باستخدام ANOVA وMutual Information.",
        "إضافة Feature Tokenizer Transformer كنموذج deep tabular learning.",
        "تقييم باستخدام repeated cross-validation واختبار holdout.",
        "مقارنة منظمة مع أعمال منشورة مع توضيح حدود المقارنة.",
    ]:
        story.append(bullet(item, body_style))

    story.extend(
        [
            paragraph("4. النتائج الأساسية", heading_style),
            make_table(
                rows=[
                    [
                        str(best_accuracy["model"]),
                        pct(float(best_accuracy["accuracy"])),
                        pct(float(best_accuracy["recall"])),
                        pct(float(best_accuracy["f1"])),
                        pct(float(best_accuracy["roc_auc"])),
                    ],
                    [
                        str(best_auc["model"]),
                        pct(float(best_auc["accuracy"])),
                        pct(float(best_auc["recall"])),
                        pct(float(best_auc["f1"])),
                        pct(float(best_auc["roc_auc"])),
                    ],
                    [
                        str(best_holdout["model"]) + " - Holdout",
                        pct(float(best_holdout["accuracy"])),
                        pct(float(best_holdout["recall"])),
                        pct(float(best_holdout["f1"])),
                        pct(float(best_holdout["roc_auc"])),
                    ],
                ],
                header=["النموذج", "Accuracy", "Recall", "F1-score", "ROC-AUC"],
                table_style=table_style,
            ),
            paragraph("5. Feature Selection وFT-Transformer", heading_style),
            make_table(
                rows=[
                    [
                        str(best_feature_selection["model"]),
                        pct(float(best_feature_selection["accuracy"])),
                        pct(float(best_feature_selection["recall"])),
                        pct(float(best_feature_selection["roc_auc"])),
                    ],
                    [
                        str(ft_row["model"]),
                        pct(float(ft_row["accuracy"])),
                        pct(float(ft_row["recall"])),
                        pct(float(ft_row["roc_auc"])),
                    ],
                ],
                header=["التجربة", "Accuracy", "Recall", "ROC-AUC"],
                table_style=table_style,
            ),
            paragraph(
                "Feature Selection كان مفيدًا كتحليل بحثي لكنه لم يتفوق على الإطار الهجين الكامل. "
                "أما Feature Tokenizer Transformer فتمت إضافته كخط أساس حديث للتعلم العميق على البيانات الجدولية، "
                "لكنه ظل أقل من نماذج boosting والهجين، وهو أمر متوقع مع البيانات الطبية الجدولية محدودة الحجم.",
                body_style,
            ),
            PageBreak(),
            paragraph("6. المقارنة مع الأعمال المنشورة", heading_style),
            make_table(
                rows=[
                    [
                        str(row["study"]),
                        str(row["dataset"]),
                        str(row["method"]),
                        str(row["accuracy"]),
                        str(row["roc_auc"]),
                        str(row["comparison_type"]),
                    ]
                    for _, row in published.iterrows()
                ],
                header=["الدراسة", "Dataset", "Method", "Accuracy", "ROC-AUC", "نوع المقارنة"],
                table_style=table_style,
            ),
            paragraph(
                "تم فصل نتائج الأعمال المنشورة عن نتائجنا المتولدة من النوتبوك. المقارنة المباشرة لا تكون صحيحة "
                "إلا عند استخدام نفس Dataset ونفس preprocessing ونفس evaluation protocol.",
                body_style,
            ),
            paragraph("7. الإسهام البحثي النهائي", heading_style),
            paragraph(
                "الإسهام البحثي لا يتمثل في استخدام أكثر من Dataset فقط. استخدام أكثر من مصدر بيانات هو validation "
                "لإثبات الثبات وقابلية التعميم. الإسهام الأساسي هو Unified Explainable Hybrid Framework الذي يجمع "
                "بين tuned ensemble learning، النماذج الهجينة، قابلية التفسير باستخدام SHAP، تحليل الخصائص، ونموذج "
                "FT-Transformer.",
                body_style,
            ),
            paragraph("8. الخلاصة", heading_style),
            paragraph(
                "أصبح المشروع الآن أكثر قوة وتنظيمًا من الناحية البحثية. النتائج الأساسية وصلت إلى حوالي 94% Accuracy "
                "وأكثر من 97% ROC-AUC على OpenML Heart 1190، مع وجود مقارنة منشورة واضحة وتحديد دقيق للـ contribution. "
                "الشغل الحالي يقدم إطارًا قابلًا للتفسير والتكرار، وليس مجرد تجربة نماذج على Dataset واحدة.",
                body_style,
            ),
        ]
    )

    document.build(story)
    return output_path


def main() -> None:
    """Generate the Arabic update summary PDF."""

    project_root = Path(__file__).resolve().parents[1]
    config = SummaryConfig(
        project_root=project_root,
        results_dir=project_root / "outputs" / "clean_notebook",
        output_dir=project_root / "reports",
        font_regular=Path("C:/Windows/Fonts/arial.ttf"),
        font_bold=Path("C:/Windows/Fonts/arialbd.ttf"),
    )
    output_path = build_summary_pdf(config)
    print(output_path.resolve())


if __name__ == "__main__":
    main()
