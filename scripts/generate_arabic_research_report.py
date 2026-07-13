"""Generate a complete Arabic research report for the heart disease project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Iterable

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
class ArabicReportConfig:
    """Configuration for Arabic report generation.

    Attributes:
        project_root: Project root path.
        results_dir: Directory containing executed notebook CSV outputs.
        output_dir: Directory where Arabic reports are written.
        font_regular: Arabic-capable regular font path.
        font_bold: Arabic-capable bold font path.
    """

    project_root: Path
    results_dir: Path
    output_dir: Path
    font_regular: Path
    font_bold: Path


def format_percent(value: float) -> str:
    """Format a decimal metric as a percentage.

    Args:
        value: Decimal metric value.

    Returns:
        Percentage string.
    """

    return f"{value * 100:.2f}%"


def read_csv(path: Path) -> pd.DataFrame:
    """Read a required CSV output file.

    Args:
        path: CSV path.

    Returns:
        Loaded DataFrame.
    """

    if not path.exists():
        raise FileNotFoundError(f"Missing required result file: {path}")
    return pd.read_csv(path)


def prepare_metric_table(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Prepare a metric table with formatted percentage columns.

    Args:
        frame: Source DataFrame.
        columns: Metric columns to format.

    Returns:
        Formatted DataFrame.
    """

    output = frame.copy()
    for column in columns:
        if column in output.columns:
            output[column] = output[column].map(format_percent)
    return output


def markdown_table(frame: pd.DataFrame) -> str:
    """Convert a DataFrame to a markdown table without optional dependencies."""

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
        "| " + " | ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def clean_markdown_indentation(text: str) -> str:
    """Remove template indentation from generated markdown."""

    lines = []
    for line in text.splitlines():
        if line.startswith("        "):
            lines.append(line[8:])
        else:
            lines.append(line)
    return "\n".join(lines).strip() + "\n"


def ar(text: object) -> str:
    """Reshape Arabic text for ReportLab rendering.

    Args:
        text: Text value.

    Returns:
        Display-ready RTL text.
    """

    value = str(text)
    return get_display(arabic_reshaper.reshape(value))


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    """Create an Arabic-aware paragraph."""

    return Paragraph(ar(text), style)


def bullet_lines(items: Iterable[str], style: ParagraphStyle) -> list[Paragraph]:
    """Create Arabic bullet paragraphs."""

    return [paragraph(f"- {item}", style) for item in items]


def pdf_table(frame: pd.DataFrame, style: ParagraphStyle, max_rows: int = 10) -> Table:
    """Create a right-to-left PDF table.

    Args:
        frame: Source table.
        style: Paragraph style.
        max_rows: Maximum row count.

    Returns:
        ReportLab table.
    """

    visible = frame.head(max_rows).copy()
    values = []
    header = [Paragraph(ar(column), style) for column in reversed(visible.columns)]
    values.append(header)
    for row in visible.astype(str).values.tolist():
        values.append([Paragraph(ar(value), style) for value in reversed(row)])

    table = Table(values, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
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


def build_arabic_markdown(
    dataset_comparison: pd.DataFrame,
    tuned_results: pd.DataFrame,
    holdout_results: pd.DataFrame,
    feature_selection: pd.DataFrame,
    ft_results: pd.DataFrame,
    shap_importance: pd.DataFrame,
    published_work: pd.DataFrame,
    contribution_matrix: pd.DataFrame,
) -> str:
    """Build the Arabic markdown report."""

    dataset_table = prepare_metric_table(
        dataset_comparison,
        ["accuracy", "f1", "recall", "roc_auc"],
    )
    tuned_table = prepare_metric_table(
        tuned_results,
        ["accuracy", "accuracy_std", "precision", "recall", "f1", "roc_auc", "roc_auc_std", "pr_auc"],
    )
    holdout_table = prepare_metric_table(
        holdout_results,
        ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"],
    )
    feature_selection_table = prepare_metric_table(
        feature_selection,
        ["accuracy", "accuracy_std", "precision", "recall", "f1", "roc_auc", "roc_auc_std", "pr_auc"],
    )
    ft_table = prepare_metric_table(
        ft_results,
        ["best_valid_auc", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"],
    )

    best_accuracy = tuned_results.sort_values("accuracy", ascending=False).iloc[0]
    best_auc = tuned_results.sort_values("roc_auc", ascending=False).iloc[0]
    best_holdout = holdout_results.sort_values("roc_auc", ascending=False).iloc[0]

    markdown_text = dedent(
        f"""
        # تقرير مشروع التنبؤ بأمراض القلب باستخدام تقنيات الذكاء الاصطناعي

        ## 1. ملخص تنفيذي

        يهدف هذا العمل إلى بناء إطار تجريبي قوي للتنبؤ بأمراض القلب اعتمادًا على بيانات طبية منظمة ونماذج تعلم آلي متعددة. بدأ العمل من تجربة أولية على بيانات Kaggle Cardiovascular Disease، ثم تم توسيعه ليشمل بيانات UCI Heart Disease وبيانات OpenML Heart Disease Comprehensive 1190، مع إضافة نماذج هجينة، وتحليل اختيار الخصائص، ونموذج Feature Tokenizer Transformer، وتفسير النتائج باستخدام SHAP.

        أفضل نتيجة من حيث الدقة في التحقق المتقاطع المتكرر كانت لنموذج **{best_accuracy['model']}** بدقة **{format_percent(best_accuracy['accuracy'])}**، وقيمة F1 تساوي **{format_percent(best_accuracy['f1'])}**، وROC-AUC تساوي **{format_percent(best_accuracy['roc_auc'])}**.

        أفضل نتيجة من حيث ROC-AUC كانت لنموذج **{best_auc['model']}** بقيمة **{format_percent(best_auc['roc_auc'])}**، وPR-AUC تساوي **{format_percent(best_auc['pr_auc'])}**.

        في اختبار holdout النهائي، حقق نموذج **{best_holdout['model']}** قيمة ROC-AUC قدرها **{format_percent(best_holdout['roc_auc'])}**، مع دقة **{format_percent(best_holdout['accuracy'])}** وقيمة F1 تساوي **{format_percent(best_holdout['f1'])}**.

        ## 2. هدف الدراسة

        الهدف الأساسي هو تطوير نظام قابل للتكرار للتنبؤ بخطر الإصابة بأمراض القلب، مع التركيز على:

        - مقارنة أكثر من مصدر بيانات بدل الاعتماد على مجموعة بيانات واحدة.
        - اختبار نماذج تقليدية ونماذج boosting ونماذج هجينة.
        - تفسير قرارات النموذج باستخدام SHAP.
        - دراسة أثر اختيار الخصائص على الأداء.
        - إضافة نموذج عميق حديث للبيانات الجدولية باستخدام Feature Tokenizer Transformer.
        - تقديم نتائج قابلة للدفاع العلمي من خلال repeated stratified cross-validation واختبار holdout.

        ## 3. البيانات المستخدمة

        ### 3.1 بيانات Kaggle Cardiovascular Disease

        تحتوي هذه البيانات على حوالي 70 ألف سجل قبل التنظيف، وتشمل خصائص مثل العمر، الطول، الوزن، ضغط الدم، الكوليسترول، الجلوكوز، التدخين، النشاط البدني، والهدف cardio. تم استخدامها كبداية للتجربة ولتقييم حدود الأداء على البيانات الأصلية.

        أوضحت النتائج أن هذه البيانات، رغم حجمها الكبير، لديها سقف أداء محدود عند التقييم النظيف، ويرجع ذلك إلى طبيعة الخصائص المتاحة وغياب بعض المؤشرات السريرية الأكثر تفصيلًا.

        ### 3.2 بيانات UCI Heart Disease

        تم استخدام ملفات Cleveland وHungarian وSwitzerland وLong Beach VA كمعيار طبي معروف في أبحاث أمراض القلب. هذه البيانات أصغر حجمًا لكنها أكثر ارتباطًا بالأدبيات البحثية الطبية الخاصة بتوقع أمراض القلب.

        ### 3.3 بيانات OpenML Heart Disease Comprehensive 1190

        تم استخدام هذه البيانات كأقوى معيار نهائي لأنها تجمع مصادر متعددة: Cleveland وHungarian وSwitzerland وLong Beach VA وStatlog. تحتوي على 1190 سجلًا وتوفر توازنًا جيدًا بين الحجم والقيمة السريرية للخصائص.

        ## 4. معالجة البيانات والهندسة المبدئية للخصائص

        تم تنفيذ خطوات تنظيف وتجهيز مختلفة حسب كل مجموعة بيانات:

        - إزالة القيم غير المنطقية في ضغط الدم والطول والوزن والعمر في بيانات Kaggle.
        - حساب مؤشرات مشتقة مثل BMI وpulse pressure وmean arterial pressure.
        - تحويل الهدف إلى تصنيف ثنائي في بيانات UCI.
        - التعامل مع القيم الناقصة باستخدام median للخصائص الرقمية وmost frequent للخصائص الفئوية.
        - استخدام StandardScaler للخصائص الرقمية وOneHotEncoder للخصائص الفئوية.
        - الحفاظ على فصل واضح بين طبقة البيانات، طبقة التجهيز، طبقة النمذجة، وطبقة التقييم.

        ## 5. النماذج التي تم اختبارها

        شملت التجارب النماذج التالية:

        - Logistic Regression
        - Naive Bayes
        - Support Vector Machines
        - Random Forest
        - Extra Trees
        - Gradient Boosting
        - AdaBoost
        - XGBoost
        - LightGBM
        - CatBoost
        - Multi-layer Perceptron
        - Soft Voting Hybrid
        - Stacking Hybrid
        - Feature Selection + XGBoost/CatBoost
        - Feature Tokenizer Transformer

        ## 6. النماذج الهجينة

        تم بناء نموذجين هجينين أساسيين:

        ### 6.1 Soft Voting Hybrid

        يعتمد هذا النموذج على دمج احتمالات التنبؤ من عدة نماذج قوية. الفكرة أن كل نموذج قد يتعلم نمطًا مختلفًا من البيانات، وبالتالي فإن دمج الاحتمالات يساعد على تحسين الثبات ورفع ROC-AUC.

        ### 6.2 Stacking Hybrid

        يعتمد هذا النموذج على مجموعة نماذج أساسية، ثم يستخدم نموذجًا نهائيًا يتعلم من مخرجات هذه النماذج. الهدف هو الاستفادة من نقاط قوة كل نموذج بدل الاعتماد على مصنف واحد فقط.

        ## 7. النتائج عبر مجموعات البيانات

        {markdown_table(dataset_table)}

        توضح النتائج أن بيانات Kaggle حققت أداءً أقل نسبيًا، بينما قدمت بيانات OpenML Heart 1190 أفضل نتائج عامة. هذا يدعم قرار توسيع الدراسة بدل الاكتفاء بمجموعة بيانات واحدة.

        ## 8. نتائج النماذج المحسنة والهجينة

        {markdown_table(tuned_table)}

        حقق CatBoost أعلى دقة تقريبًا، بينما حقق Soft Voting Hybrid أعلى ROC-AUC. لذلك يمكن عرض CatBoost كأقوى نموذج منفرد، وSoft Voting/Stacking كأقوى إطار هجين.

        ## 9. اختبار Holdout النهائي

        {markdown_table(holdout_table)}

        يوضح اختبار holdout أن النماذج الهجينة احتفظت بأداء قوي خارج التحقق المتقاطع، خاصة من حيث ROC-AUC وPR-AUC.

        ## 10. تحليل اختيار الخصائص

        {markdown_table(feature_selection_table)}

        تم اختبار ANOVA SelectKBest وMutual Information SelectKBest مع XGBoost وCatBoost. أظهرت النتائج أن تقليل عدد الخصائص لم يتفوق على استخدام جميع الخصائص مع النماذج المحسنة والهجينة. هذه نتيجة مهمة لأنها توضح أن الخصائص الكاملة في OpenML تحمل معلومات مفيدة، وأن حذف جزء منها قد يقلل الأداء.

        ## 11. Feature Tokenizer Transformer

        {markdown_table(ft_table)}

        تمت إضافة Feature Tokenizer Transformer كنموذج تعلم عميق للبيانات الجدولية. يقوم النموذج بتحويل كل خاصية طبية إلى token embedding ثم يستخدم self-attention لتعلم العلاقات بين الخصائص.

        حقق النموذج أداءً جيدًا كخط أساس عميق، لكنه لم يتفوق على نماذج boosting والهجين. وهذا متوقع في البيانات الطبية الجدولية الصغيرة أو المتوسطة، حيث تكون CatBoost وXGBoost وLightGBM غالبًا أكثر كفاءة في التعلم من عدد محدود من السجلات.

        ## 12. تفسير النموذج باستخدام SHAP

        {markdown_table(shap_importance.head(12))}

        أظهر تحليل SHAP أن أهم العوامل المؤثرة في قرارات النموذج تشمل chest pain type وST slope وoldpeak وcholesterol وmax heart rate وexercise angina وsex وage وresting blood pressure. هذه النتائج مهمة لأنها تربط أداء النموذج بعوامل طبية يمكن تفسيرها بدل الاكتفاء برقم الدقة فقط.

        ## 13. المقارنة مع الأعمال المنشورة

        {markdown_table(published_work)}

        بعض الأعمال المنشورة تعرض أرقامًا مرتفعة جدًا مثل 98% على Cleveland أو UHDD. الفرق الأساسي أن هذه الأعمال قد تعتمد على مجموعة بيانات صغيرة، أو split واحد، أو تحسين مكثف لنموذج معين مثل XGBoost، أو اختيار خصائص قد يرفع النتيجة على بيانات محددة.

        المقارنة المباشرة لا تكون صحيحة إلا عند استخدام نفس مجموعة البيانات، ونفس طريقة التقسيم، ونفس خطوات التجهيز، ونفس بروتوكول التقييم. لذلك تم فصل نتائج الأعمال المنشورة عن النتائج التي تم توليدها من النوتبوك.

        في هذا العمل تم التركيز على إطار أوسع وأكثر قابلية للتكرار:

        - استخدام أكثر من مجموعة بيانات.
        - مقارنة مصادر بيانات مختلفة.
        - استخدام repeated stratified cross-validation بدل الاعتماد فقط على split واحد.
        - اختبار نماذج منفردة وهجينة.
        - إضافة تفسير SHAP.
        - إضافة Feature Tokenizer Transformer.
        - توضيح حدود كل Dataset بدل الاكتفاء بأفضل رقم فقط.

        لذلك فإن قوة العمل ليست فقط في الوصول إلى رقم عالٍ، بل في بناء إطار بحثي شامل يوضح متى ولماذا تتحسن النتائج، وما حدود كل مجموعة بيانات.

        ## 14. الإسهام البحثي

        {markdown_table(contribution_matrix)}

        يمكن تلخيص الإسهام البحثي في النقاط التالية:

        1. بناء إطار تجريبي متعدد البيانات للتنبؤ بأمراض القلب.
        2. تحليل حدود بيانات Kaggle وإظهار أن انخفاض الأداء مرتبط بطبيعة الخصائص وليس فقط باختيار النموذج.
        3. استخدام OpenML Heart 1190 كمصدر أقوى يجمع أكثر من benchmark طبي.
        4. بناء نماذج هجينة Soft Voting وStacking اعتمادًا على نماذج boosting وtree ensembles.
        5. إضافة SHAP explainability لتفسير العوامل الطبية المؤثرة.
        6. دراسة أثر feature selection على الأداء.
        7. إضافة Feature Tokenizer Transformer كخط أساس حديث للتعلم العميق على البيانات الجدولية.
        8. اعتماد تقييم أكثر ثباتًا من خلال repeated stratified cross-validation واختبار holdout.

        استخدام أكثر من Dataset لا يتم تقديمه كإسهام بحثي مستقل، ولكنه دليل تحقق validation على ثبات الإطار المقترح. الإسهام الأساسي هو الإطار الموحد القابل للتفسير الذي يجمع بين النماذج الهجينة، وSHAP، وFeature Selection، وFeature Tokenizer Transformer، والتقييم القوي.

        ## 15. الخلاصة

        أظهرت النتائج أن أفضل أداء تحقق على بيانات OpenML Heart Disease Comprehensive 1190. حقق نموذج CatBoost المحسن أعلى دقة تقريبًا، بينما حقق Soft Voting Hybrid أعلى ROC-AUC. كما أوضح تحليل SHAP أن قرارات النموذج تعتمد على عوامل طبية منطقية، مما يزيد قابلية تفسير النتائج.

        إضافة Feature Tokenizer Transformer تمثل امتدادًا حديثًا للعمل، حتى وإن لم يتفوق على boosting models، لأنها تثبت أن الدراسة لا تقتصر على النماذج التقليدية فقط، بل تقارن أيضًا اتجاهات حديثة في deep tabular learning.

        بناءً على ذلك، أصبح العمل إطارًا بحثيًا متكاملًا يجمع بين الأداء، المقارنة بين البيانات، النماذج الهجينة، قابلية التفسير، وتحليل الخصائص.
        """
    )
    return clean_markdown_indentation(markdown_text)


def build_pdf(
    config: ArabicReportConfig,
    markdown_text: str,
    tables: dict[str, pd.DataFrame],
) -> Path:
    """Build a complete Arabic PDF report from markdown content."""

    pdfmetrics.registerFont(TTFont("Arabic", str(config.font_regular)))
    pdfmetrics.registerFont(TTFont("ArabicBold", str(config.font_bold)))

    output_path = config.output_dir / "heart_disease_arabic_research_report.pdf"
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
        "ArabicTitle",
        parent=styles["Title"],
        fontName="ArabicBold",
        fontSize=18,
        leading=24,
        alignment=TA_RIGHT,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=12,
    )
    heading_style = ParagraphStyle(
        "ArabicHeading",
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
        "ArabicBody",
        parent=styles["BodyText"],
        fontName="Arabic",
        fontSize=9.5,
        leading=15,
        alignment=TA_RIGHT,
        textColor=colors.HexColor("#1f2937"),
        spaceAfter=5,
    )
    table_style = ParagraphStyle(
        "ArabicTable",
        parent=body_style,
        fontSize=7,
        leading=9,
        alignment=TA_RIGHT,
    )
    code_style = ParagraphStyle(
        "CodeTable",
        parent=styles["BodyText"],
        fontName="Courier",
        fontSize=5.8,
        leading=7,
        alignment=0,
        textColor=colors.HexColor("#111827"),
        spaceAfter=1,
    )

    story = []
    table_buffer: list[str] = []

    def flush_table() -> None:
        if not table_buffer:
            return
        for table_line in table_buffer:
            if set(table_line.replace("|", "").replace(" ", "")) == {"-"}:
                continue
            story.append(Paragraph(table_line.replace("|", " | "), code_style))
        story.append(Spacer(1, 0.08 * inch))
        table_buffer.clear()

    for raw_line in markdown_text.splitlines():
        line = raw_line.strip()
        if not line:
            flush_table()
            story.append(Spacer(1, 0.04 * inch))
            continue
        if line.startswith("|"):
            table_buffer.append(line)
            continue

        flush_table()
        cleaned = line.replace("**", "")
        if cleaned.startswith("# "):
            story.append(paragraph(cleaned[2:], title_style))
        elif cleaned.startswith("## "):
            story.append(paragraph(cleaned[3:], heading_style))
        elif cleaned.startswith("### "):
            story.append(paragraph(cleaned[4:], heading_style))
        elif cleaned.startswith("- "):
            story.append(paragraph(cleaned, body_style))
        elif len(cleaned) > 2 and cleaned[0].isdigit() and cleaned[1] == ".":
            story.append(paragraph(cleaned, body_style))
        else:
            story.append(paragraph(cleaned, body_style))

    flush_table()

    document.build(story)
    return output_path


def generate_report(config: ArabicReportConfig) -> tuple[Path, Path]:
    """Generate Arabic markdown and PDF reports."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = read_csv(config.results_dir / "final_dataset_comparison.csv")
    tuned = read_csv(config.results_dir / "openml_tuned_repeated_cv_results.csv")
    holdout = read_csv(config.results_dir / "final_holdout_results.csv")
    feature_selection = read_csv(config.results_dir / "feature_selection_repeated_cv_results.csv")
    ft_results = read_csv(config.results_dir / "ft_transformer_holdout_results.csv")
    shap_importance = read_csv(config.results_dir / "shap_feature_importance.csv")
    published_work = read_csv(config.results_dir / "published_work_comparison.csv")
    contribution_matrix = read_csv(config.results_dir / "research_contribution_matrix.csv")

    markdown_text = build_arabic_markdown(
        dataset,
        tuned,
        holdout,
        feature_selection,
        ft_results,
        shap_importance,
        published_work,
        contribution_matrix,
    )
    markdown_path = config.output_dir / "heart_disease_arabic_research_report.md"
    markdown_path.write_text(markdown_text, encoding="utf-8")

    tables = {
        "dataset": prepare_metric_table(dataset, ["accuracy", "f1", "recall", "roc_auc"]),
        "tuned": prepare_metric_table(
            tuned,
            ["accuracy", "accuracy_std", "precision", "recall", "f1", "roc_auc", "roc_auc_std", "pr_auc"],
        ),
        "holdout": prepare_metric_table(holdout, ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]),
        "feature_selection": prepare_metric_table(
            feature_selection,
            ["accuracy", "accuracy_std", "precision", "recall", "f1", "roc_auc", "roc_auc_std", "pr_auc"],
        ),
        "ft": prepare_metric_table(
            ft_results,
            ["best_valid_auc", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"],
        ),
        "shap": shap_importance.head(12),
    }
    pdf_path = build_pdf(config, markdown_text, tables)
    return markdown_path, pdf_path


def main() -> None:
    """Generate the Arabic research report."""

    project_root = Path(__file__).resolve().parents[1]
    config = ArabicReportConfig(
        project_root=project_root,
        results_dir=project_root / "outputs" / "clean_notebook",
        output_dir=project_root / "reports",
        font_regular=Path("C:/Windows/Fonts/arial.ttf"),
        font_bold=Path("C:/Windows/Fonts/arialbd.ttf"),
    )
    markdown_path, pdf_path = generate_report(config)
    print(f"Arabic Markdown report: {markdown_path}")
    print(f"Arabic PDF report: {pdf_path}")


if __name__ == "__main__":
    main()
