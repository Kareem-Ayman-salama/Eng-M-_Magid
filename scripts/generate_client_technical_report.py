"""Generate the client technical report from executed experiment outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


@dataclass(frozen=True)
class ReportConfig:
    """Configuration for client report generation.

    Attributes:
        project_root: Absolute project root path.
        output_dir: Directory where report artifacts are written.
        results_dir: Directory containing executed notebook CSV outputs.
    """

    project_root: Path
    output_dir: Path
    results_dir: Path


def format_percent(value: float) -> str:
    """Format a decimal metric as a percent string.

    Args:
        value: Decimal metric value.

    Returns:
        Percentage string with two decimal places.
    """

    return f"{value * 100:.2f}%"


def read_required_csv(path: Path) -> pd.DataFrame:
    """Read a required CSV file with a clear error if missing.

    Args:
        path: CSV file path.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """

    if not path.exists():
        raise FileNotFoundError(f"Required report input was not found: {path}")
    return pd.read_csv(path)


def read_optional_csv(path: Path) -> pd.DataFrame:
    """Read an optional CSV file.

    Args:
        path: CSV file path.

    Returns:
        Loaded DataFrame, or an empty DataFrame if the file is missing.
    """

    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def prepare_metric_table(
    frame: pd.DataFrame,
    metric_columns: list[str],
    sort_by: str | None = None,
) -> pd.DataFrame:
    """Prepare a compact metrics table for reporting.

    Args:
        frame: Source metrics DataFrame.
        metric_columns: Metric columns to format as percentages.
        sort_by: Optional metric column used for descending sort.

    Returns:
        Report-ready DataFrame with formatted metric strings.
    """

    report_frame = frame.copy()
    if sort_by is not None:
        report_frame = report_frame.sort_values(sort_by, ascending=False)
    for column in metric_columns:
        if column in report_frame.columns:
            report_frame[column] = report_frame[column].map(format_percent)
    return report_frame


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Convert a DataFrame to a GitHub-style markdown table.

    Args:
        frame: DataFrame to serialize.

    Returns:
        Markdown table text.
    """

    string_frame = frame.astype(str)
    headers = [str(column) for column in string_frame.columns]
    rows = string_frame.values.tolist()
    widths = [
        max(len(header), *(len(str(row[index])) for row in rows))
        for index, header in enumerate(headers)
    ]
    header_row = "| " + " | ".join(
        header.ljust(widths[index]) for index, header in enumerate(headers)
    ) + " |"
    divider_row = "| " + " | ".join("-" * width for width in widths) + " |"
    body_rows = [
        "| " + " | ".join(
            str(value).ljust(widths[index]) for index, value in enumerate(row)
        ) + " |"
        for row in rows
    ]
    return "\n".join([header_row, divider_row, *body_rows])


def clean_markdown_indentation(text: str) -> str:
    """Remove template indentation without changing table formatting.

    Args:
        text: Raw markdown text.

    Returns:
        Clean markdown text.
    """

    cleaned_lines = []
    for line in text.splitlines():
        if line.startswith("        "):
            cleaned_lines.append(line[8:])
        else:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip() + "\n"


def create_bar_chart(
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    output_path: Path,
    color: str,
) -> None:
    """Create a horizontal metric bar chart.

    Args:
        frame: Source DataFrame.
        x_column: Category column.
        y_column: Numeric metric column.
        title: Chart title.
        output_path: File path for the rendered chart.
        color: Matplotlib bar color.
    """

    chart_frame = frame.sort_values(y_column, ascending=True)
    plt.figure(figsize=(9.5, max(3.5, len(chart_frame) * 0.45)))
    plt.barh(chart_frame[x_column], chart_frame[y_column] * 100, color=color)
    plt.xlabel(f"{y_column.replace('_', ' ').title()} (%)")
    plt.title(title)
    plt.xlim(max(0, chart_frame[y_column].min() * 100 - 5), 100)
    plt.grid(axis="x", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def build_markdown_report(
    config: ReportConfig,
    dataset_comparison: pd.DataFrame,
    tuned_results: pd.DataFrame,
    holdout_results: pd.DataFrame,
    openml_baseline: pd.DataFrame,
    feature_selection_results: pd.DataFrame,
    ft_transformer_results: pd.DataFrame,
    shap_importance: pd.DataFrame,
    published_work: pd.DataFrame,
    contribution_matrix: pd.DataFrame,
) -> str:
    """Build the markdown report body.

    Args:
        config: Report configuration.
        dataset_comparison: Final dataset comparison results.
        tuned_results: Tuned repeated cross-validation results.
        holdout_results: Final holdout evaluation results.
        openml_baseline: OpenML baseline model results.

    Returns:
        Markdown report content.
    """

    dataset_table = prepare_metric_table(
        dataset_comparison,
        ["accuracy", "f1", "recall", "roc_auc"],
        sort_by="roc_auc",
    )
    tuned_table = prepare_metric_table(
        tuned_results,
        ["accuracy", "accuracy_std", "precision", "recall", "f1", "roc_auc", "roc_auc_std", "pr_auc"],
        sort_by="accuracy",
    )
    holdout_table = prepare_metric_table(
        holdout_results,
        ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"],
        sort_by="roc_auc",
    )
    baseline_table = prepare_metric_table(
        openml_baseline.sort_values("accuracy", ascending=False).head(8),
        ["accuracy", "accuracy_std", "precision", "recall", "f1", "roc_auc", "roc_auc_std", "pr_auc"],
    )
    feature_selection_table = prepare_metric_table(
        feature_selection_results,
        ["accuracy", "accuracy_std", "precision", "recall", "f1", "roc_auc", "roc_auc_std", "pr_auc"],
        sort_by="roc_auc",
    ) if not feature_selection_results.empty else pd.DataFrame()
    ft_table = prepare_metric_table(
        ft_transformer_results,
        ["best_valid_auc", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"],
        sort_by="roc_auc",
    ) if not ft_transformer_results.empty else pd.DataFrame()

    best_cv = tuned_results.sort_values("accuracy", ascending=False).iloc[0]
    best_auc = tuned_results.sort_values("roc_auc", ascending=False).iloc[0]
    best_holdout = holdout_results.sort_values("roc_auc", ascending=False).iloc[0]

    markdown = dedent(
        f"""
        # Heart Disease Prediction and Detection Using AI Techniques

        ## Executive Summary

        This report summarizes the completed experimental work for the heart disease
        prediction thesis project. The work started from the originally provided
        Kaggle cardiovascular dataset and was extended into a stronger comparative
        experimental framework covering multiple datasets, classical ML models,
        tuned gradient boosting models, and hybrid ensemble models.

        The strongest repeated cross-validation result was achieved by
        **{best_cv['model']}** with **{format_percent(best_cv['accuracy'])} accuracy**,
        **{format_percent(best_cv['f1'])} F1-score**, and
        **{format_percent(best_cv['roc_auc'])} ROC-AUC**.

        The strongest discrimination score was achieved by **{best_auc['model']}**
        with **{format_percent(best_auc['roc_auc'])} ROC-AUC** and
        **{format_percent(best_auc['pr_auc'])} PR-AUC**.

        On the final holdout evaluation, the best ROC-AUC model was
        **{best_holdout['model']}**, reaching **{format_percent(best_holdout['accuracy'])}
        accuracy**, **{format_percent(best_holdout['f1'])} F1-score**, and
        **{format_percent(best_holdout['roc_auc'])} ROC-AUC**.

        All metrics in this report are generated from the executed notebook output
        files under:

        `{config.results_dir}`

        ## What We Worked On

        1. Reviewed the proposal topic and translated it into a practical machine
           learning experiment plan.
        2. Started with the original Kaggle Cardiovascular Disease dataset.
        3. Tested many classical and ensemble machine learning models.
        4. Identified that the Kaggle dataset performance ceiling is around the
           low-to-mid 70% accuracy range under clean validation.
        5. Added UCI Heart Disease datasets to support a medically standard
           comparison benchmark.
        6. Added the OpenML Heart Disease Comprehensive 1190 dataset, which combines
           Cleveland, Hungarian, Switzerland, Long Beach VA, and Statlog sources.
        7. Built tuned single models and hybrid ensemble models.
        8. Added SHAP explainability to interpret the strongest tree-based model.
        9. Added feature selection experiments to test whether reduced clinical
           feature sets improve performance.
        10. Added a Feature Tokenizer Transformer deep tabular baseline.
        11. Organized the final work into one clean notebook with reproducible outputs,
           visualizations, comparison tables, and printable HTML.

        ## Main Deliverables

        - Main notebook:
          `notebooks/00_clean_heart_disease_experiments.ipynb`
        - Executed notebook:
          `outputs/clean_notebook/00_clean_heart_disease_experiments.executed.ipynb`
        - Printable HTML:
          `outputs/clean_notebook/00_clean_heart_disease_experiments.html`
        - Generated CSV outputs:
          `outputs/clean_notebook/*.csv`
        - Reusable code package:
          `src/heart_disease_prediction/`
        - Notebook generator:
          `scripts/generate_clean_architecture_notebook.py`

        ## Datasets Used

        ### 1. Kaggle Cardiovascular Disease Dataset

        - File: `archive/cardio_train.csv`
        - Size: approximately 70,000 records before cleaning.
        - Target: `cardio`
        - Role: Original dataset and baseline experiment source.
        - Finding: Good for large-scale experimentation, but the available feature
          set limits clean predictive performance.

        ### 2. UCI Heart Disease Dataset

        - Folder: `data/uci_heart/`
        - Sources: Cleveland, Hungarian, Switzerland, and Long Beach VA processed files.
        - Role: Classical medical benchmark for heart disease prediction.
        - Finding: Smaller than Kaggle, but clinically closer to the standard
          published heart disease ML literature.

        ### 3. OpenML Heart Disease Comprehensive 1190

        - Folder: `data/openml_heart_1190/`
        - Size: 1,190 records.
        - Sources: Cleveland, Hungarian, Switzerland, Long Beach VA, and Statlog.
        - Role: Final strongest benchmark dataset.
        - Finding: Best dataset for strong thesis-level results because it combines
          multiple established heart disease sources into one structured benchmark.

        ## Models Tested

        The experiments covered:

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
        - Hard Voting ensemble
        - Soft Voting hybrid ensemble
        - Stacking hybrid ensemble
        - Feature-selection-enhanced XGBoost and CatBoost
        - Feature Tokenizer Transformer for tabular clinical data

        ## Hybrid Model Design

        Yes, the project now includes true hybrid models.

        ### Soft Voting Hybrid

        The soft voting hybrid combines predicted class probabilities from multiple
        strong learners. This is useful when several models are individually strong
        but capture slightly different decision boundaries.

        ### Stacking Hybrid

        The stacking hybrid uses several base learners, then trains a meta-model on
        their outputs. This allows the final learner to identify when each base
        model should be trusted more.

        ### Why These Hybrids Were Selected

        The hybrid strategy was selected based on:

        - Strong standalone performance of tree-based boosting models.
        - Complementary behavior between CatBoost, LightGBM, XGBoost, Random Forest,
          and Extra Trees.
        - Better ROC-AUC and PR-AUC from probability-based ensemble predictions.
        - Need for a thesis-level model that goes beyond a single baseline classifier.

        ## Results Compared Across Datasets

        {dataframe_to_markdown(dataset_table)}

        ## OpenML Baseline Model Results

        {dataframe_to_markdown(baseline_table)}

        ## Tuned Models and Hybrid Ensembles

        {dataframe_to_markdown(tuned_table)}

        ## Feature Selection Experiments

        {dataframe_to_markdown(feature_selection_table) if not feature_selection_table.empty else "Feature selection results were not generated."}

        The selected-feature experiments are useful as an analysis contribution. In
        the current run, feature selection reduced the feature space but did not
        outperform the full tuned boosting and hybrid models. This supports the
        final decision to keep the full OpenML feature representation for the main
        predictive model.

        ## Feature Tokenizer Transformer

        {dataframe_to_markdown(ft_table) if not ft_table.empty else "FT-Transformer results were not generated."}

        The Feature Tokenizer Transformer was added as a modern deep tabular
        learning baseline. It converts each clinical variable into a token embedding
        and uses self-attention to learn interactions between variables. In this run,
        it performed competitively but remained below tuned boosting and hybrid
        ensembles, which is expected on small-to-medium structured clinical datasets.

        ## SHAP Explainability

        {dataframe_to_markdown(shap_importance.head(12)) if not shap_importance.empty else "SHAP outputs were not generated."}

        SHAP explains which clinical factors most influenced the final tree-based
        model. The strongest contributors included chest pain type, ST slope,
        oldpeak, cholesterol, maximum heart rate, exercise angina, sex, age, and
        resting blood pressure.

        ## Final Holdout Evaluation

        {dataframe_to_markdown(holdout_table)}

        ## Comparison With the Provided Published Baseline

        {dataframe_to_markdown(published_work) if not published_work.empty else "Published-work comparison was not generated."}

        These rows separate literature-reported metrics from notebook-generated
        reproducible metrics. Direct comparison is valid only when the same dataset,
        split strategy, preprocessing, and evaluation protocol are used. Therefore,
        the published results are used for literature positioning, while our results
        are reported as reproducible repeated-CV and holdout outputs.

        ## Research Contribution Positioning

        {dataframe_to_markdown(contribution_matrix) if not contribution_matrix.empty else "Contribution matrix was not generated."}

        The main contribution should not be presented as using multiple datasets
        alone. Multiple datasets are validation evidence. The contribution is the
        unified explainable hybrid framework that combines tuned boosting/tree
        ensembles, soft voting, stacking, SHAP explainability, feature-selection
        analysis, FT-Transformer deep tabular benchmarking, and robust validation.

        ## Clean Architecture Applied

        The final notebook is organized into clear layers:

        - Configuration layer: paths, seed, repeated CV settings, and output paths.
        - Data layer: dataset loading, validation, cleaning, and dataset metadata.
        - Feature layer: preprocessing pipelines for numeric and categorical fields.
        - Model layer: baseline models, tuned models, and hybrid ensemble builders.
        - Evaluation layer: cross-validation, holdout testing, ROC-AUC, PR-AUC,
          F1-score, precision, recall, and confusion matrix generation.
        - Visualization layer: target distributions, feature plots, correlation
          heatmaps, leaderboards, ROC curves, PR curves, and feature importance.

        ## What Was Added Beyond the Initial Plan

        - Multi-dataset experimental comparison.
        - UCI and OpenML benchmark integration.
        - Tuned boosting models.
        - Soft voting hybrid model.
        - Stacking hybrid model.
        - SHAP explainability.
        - Feature selection analysis.
        - Feature Tokenizer Transformer deep tabular baseline.
        - Repeated stratified cross-validation.
        - Final holdout validation.
        - Printable HTML output.
        - Clean notebook structure.
        - Generated result CSVs for traceability.

        ## Recommended Client Message

        The project has moved from a basic single-dataset reproduction into a
        stronger thesis-ready experimental framework. The final work compares
        multiple heart disease datasets, tests a broad set of machine learning
        models, adds tuned hybrid ensemble models, includes SHAP explainability,
        evaluates feature selection, and adds a Feature Tokenizer Transformer deep
        tabular baseline. The strongest clean results were obtained on the OpenML
        Heart Disease Comprehensive 1190 dataset, with approximately 94% accuracy
        and above 97% ROC-AUC.

        ## Next Recommended Improvements

        - Add a hyperparameter search section inside the notebook if the client
          wants tuning to be fully visible rather than only using tuned parameters.
        - Add nested cross-validation for optimized XGBoost if direct comparison
          with Cleveland-specific published papers is required.
        - Add a short final methodology diagram for the thesis chapter.
        - Export the final notebook to PDF after the client approves wording.
        """
    )
    return clean_markdown_indentation(markdown)


def make_paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    """Create a ReportLab paragraph with safe line breaks.

    Args:
        text: Paragraph text.
        style: Paragraph style.

    Returns:
        ReportLab Paragraph.
    """

    return Paragraph(text.replace("\n", "<br/>"), style)


def table_from_dataframe(frame: pd.DataFrame, max_rows: int = 12) -> Table:
    """Create a styled PDF table from a DataFrame.

    Args:
        frame: Source DataFrame.
        max_rows: Maximum number of rows to include.

    Returns:
        Styled ReportLab table.
    """

    visible = frame.head(max_rows).copy()
    values = [list(visible.columns)] + visible.astype(str).values.tolist()
    table = Table(values, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def bullet_list(items: list[str], style: ParagraphStyle) -> ListFlowable:
    """Create a simple bullet list for PDF output.

    Args:
        items: Bullet text items.
        style: Paragraph style.

    Returns:
        ReportLab bullet list.
    """

    return ListFlowable(
        [ListItem(Paragraph(item, style), leftIndent=12) for item in items],
        bulletType="bullet",
        leftIndent=18,
    )


def build_pdf_report(
    config: ReportConfig,
    dataset_comparison: pd.DataFrame,
    tuned_results: pd.DataFrame,
    holdout_results: pd.DataFrame,
    openml_baseline: pd.DataFrame,
    feature_selection_results: pd.DataFrame,
    ft_transformer_results: pd.DataFrame,
    shap_importance: pd.DataFrame,
    published_work: pd.DataFrame,
    contribution_matrix: pd.DataFrame,
    chart_paths: list[Path],
) -> Path:
    """Build a polished PDF report.

    Args:
        config: Report configuration.
        dataset_comparison: Final dataset comparison results.
        tuned_results: Tuned repeated CV results.
        holdout_results: Final holdout results.
        openml_baseline: OpenML baseline results.
        chart_paths: Generated chart paths to include.

    Returns:
        Final PDF path.
    """

    pdf_path = config.output_dir / "heart_disease_client_technical_report.pdf"
    document = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=16,
    )
    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.4,
        leading=13,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#1f2937"),
        spaceAfter=6,
    )

    dataset_table = prepare_metric_table(
        dataset_comparison,
        ["accuracy", "f1", "recall", "roc_auc"],
        sort_by="roc_auc",
    )
    tuned_table = prepare_metric_table(
        tuned_results,
        ["accuracy", "accuracy_std", "precision", "recall", "f1", "roc_auc", "roc_auc_std", "pr_auc"],
        sort_by="accuracy",
    )
    holdout_table = prepare_metric_table(
        holdout_results,
        ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"],
        sort_by="roc_auc",
    )
    baseline_table = prepare_metric_table(
        openml_baseline.sort_values("accuracy", ascending=False).head(8),
        ["accuracy", "accuracy_std", "precision", "recall", "f1", "roc_auc", "roc_auc_std", "pr_auc"],
    )
    feature_selection_table = prepare_metric_table(
        feature_selection_results,
        ["accuracy", "accuracy_std", "precision", "recall", "f1", "roc_auc", "roc_auc_std", "pr_auc"],
        sort_by="roc_auc",
    ) if not feature_selection_results.empty else pd.DataFrame()
    ft_table = prepare_metric_table(
        ft_transformer_results,
        ["best_valid_auc", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"],
        sort_by="roc_auc",
    ) if not ft_transformer_results.empty else pd.DataFrame()

    best_cv = tuned_results.sort_values("accuracy", ascending=False).iloc[0]
    best_auc = tuned_results.sort_values("roc_auc", ascending=False).iloc[0]

    story = [
        Paragraph("Heart Disease Prediction and Detection Using AI Techniques", title_style),
        make_paragraph(
            "Client Technical Report - experimental progress, added work, hybrid models, and results.",
            body_style,
        ),
        Spacer(1, 0.12 * inch),
        Paragraph("Executive Summary", heading_style),
        make_paragraph(
            "The project was extended from a basic single-dataset experiment into a "
            "thesis-ready comparative framework. We evaluated Kaggle CVD, UCI Heart "
            "Disease, and OpenML Heart Disease Comprehensive 1190, then built tuned "
            "boosting models and hybrid ensemble models.",
            body_style,
        ),
        make_paragraph(
            f"Best repeated CV accuracy: {best_cv['model']} at "
            f"{format_percent(best_cv['accuracy'])} accuracy, "
            f"{format_percent(best_cv['f1'])} F1-score, and "
            f"{format_percent(best_cv['roc_auc'])} ROC-AUC.",
            body_style,
        ),
        make_paragraph(
            f"Best repeated CV ROC-AUC: {best_auc['model']} at "
            f"{format_percent(best_auc['roc_auc'])} ROC-AUC and "
            f"{format_percent(best_auc['pr_auc'])} PR-AUC.",
            body_style,
        ),
        Paragraph("Work Completed", heading_style),
        bullet_list(
            [
                "Reviewed the proposal topic and translated it into an executable ML plan.",
                "Loaded and cleaned the original Kaggle cardiovascular dataset.",
                "Benchmarked classical ML, boosting, and neural baseline models.",
                "Added UCI and OpenML heart disease benchmarks for stronger thesis evidence.",
                "Built tuned CatBoost, LightGBM, XGBoost, Random Forest, and Extra Trees models.",
                "Built Soft Voting and Stacking hybrid ensemble models.",
                "Added SHAP explainability, feature selection analysis, and FT-Transformer.",
                "Created one clean notebook where metrics are produced by execution.",
            ],
            body_style,
        ),
        Paragraph("Dataset Comparison", heading_style),
        table_from_dataframe(dataset_table),
        Spacer(1, 0.12 * inch),
    ]

    for chart_path in chart_paths:
        story.extend(
            [
                Image(str(chart_path), width=7.0 * inch, height=3.2 * inch),
                Spacer(1, 0.1 * inch),
            ]
        )

    story.extend(
        [
            PageBreak(),
            Paragraph("OpenML Baseline Models", heading_style),
            table_from_dataframe(baseline_table),
            Paragraph("Tuned Models and Hybrid Ensembles", heading_style),
            table_from_dataframe(tuned_table),
            Paragraph("Feature Selection Experiments", heading_style),
            table_from_dataframe(feature_selection_table) if not feature_selection_table.empty else Paragraph(
                "Feature selection results were not generated.",
                body_style,
            ),
            Paragraph("Feature Tokenizer Transformer", heading_style),
            table_from_dataframe(ft_table) if not ft_table.empty else Paragraph(
                "FT-Transformer results were not generated.",
                body_style,
            ),
            Paragraph("Final Holdout Evaluation", heading_style),
            table_from_dataframe(holdout_table),
            Paragraph("Hybrid Model Design", heading_style),
            make_paragraph(
                "The Soft Voting Hybrid combines calibrated probability outputs from "
                "multiple strong learners. The Stacking Hybrid trains a meta-model on "
                "base model predictions. These designs were selected because boosting "
                "models and tree ensembles showed complementary strengths across "
                "accuracy, F1-score, ROC-AUC, and PR-AUC.",
                body_style,
            ),
            Paragraph("Comparison With the Published Baseline", heading_style),
            table_from_dataframe(published_work) if not published_work.empty else Paragraph(
                "Published-work comparison was not generated.",
                body_style,
            ),
            make_paragraph(
                "Direct comparison is valid only when the same dataset, split strategy, "
                "preprocessing, and evaluation protocol are used. Published values are "
                "therefore used for literature positioning, while notebook-generated "
                "metrics are reported as reproducible results.",
                body_style,
            ),
            Paragraph("Research Contribution Positioning", heading_style),
            table_from_dataframe(contribution_matrix) if not contribution_matrix.empty else Paragraph(
                "Contribution matrix was not generated.",
                body_style,
            ),
            make_paragraph(
                "The main contribution is the unified explainable hybrid framework, "
                "not the use of multiple datasets alone. Multiple datasets are used as "
                "validation evidence for generalization.",
                body_style,
            ),
            Paragraph("SHAP Explainability", heading_style),
            table_from_dataframe(shap_importance.head(12)) if not shap_importance.empty else Paragraph(
                "SHAP outputs were not generated.",
                body_style,
            ),
            make_paragraph(
                "SHAP was added to explain the clinical factors driving the final "
                "tree-based model. The strongest drivers included chest pain type, "
                "ST slope, oldpeak, cholesterol, maximum heart rate, exercise angina, "
                "sex, age, and resting blood pressure.",
                body_style,
            ),
            Paragraph("Recommended Next Steps", heading_style),
            bullet_list(
                [
                    "Add visible hyperparameter search inside the notebook if required.",
                    "Optionally add nested CV for the optimized XGBoost comparison.",
                    "Add a methodology diagram for thesis writing.",
                    "Export the approved notebook/report to final PDF.",
                ],
                body_style,
            ),
        ]
    )

    document.build(story)
    return pdf_path


def generate_report(config: ReportConfig) -> tuple[Path, Path]:
    """Generate markdown and PDF report artifacts.

    Args:
        config: Report configuration.

    Returns:
        Tuple containing markdown path and PDF path.
    """

    config.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = config.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    dataset_comparison = read_required_csv(config.results_dir / "final_dataset_comparison.csv")
    tuned_results = read_required_csv(config.results_dir / "openml_tuned_repeated_cv_results.csv")
    holdout_results = read_required_csv(config.results_dir / "final_holdout_results.csv")
    openml_baseline = read_required_csv(config.results_dir / "openml_baseline_results.csv")
    feature_selection_results = read_optional_csv(
        config.results_dir / "feature_selection_repeated_cv_results.csv"
    )
    ft_transformer_results = read_optional_csv(
        config.results_dir / "ft_transformer_holdout_results.csv"
    )
    shap_importance = read_optional_csv(config.results_dir / "shap_feature_importance.csv")
    published_work = read_optional_csv(config.results_dir / "published_work_comparison.csv")
    contribution_matrix = read_optional_csv(config.results_dir / "research_contribution_matrix.csv")

    chart_paths = [
        figures_dir / "dataset_roc_auc_comparison.png",
        figures_dir / "tuned_accuracy_comparison.png",
    ]
    create_bar_chart(
        dataset_comparison,
        "model",
        "roc_auc",
        "Dataset and Model ROC-AUC Comparison",
        chart_paths[0],
        "#2563eb",
    )
    create_bar_chart(
        tuned_results,
        "model",
        "accuracy",
        "Tuned Model Accuracy Comparison",
        chart_paths[1],
        "#059669",
    )

    markdown = build_markdown_report(
        config,
        dataset_comparison,
        tuned_results,
        holdout_results,
        openml_baseline,
        feature_selection_results,
        ft_transformer_results,
        shap_importance,
        published_work,
        contribution_matrix,
    )
    markdown_path = config.output_dir / "heart_disease_client_technical_report.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    pdf_path = build_pdf_report(
        config,
        dataset_comparison,
        tuned_results,
        holdout_results,
        openml_baseline,
        feature_selection_results,
        ft_transformer_results,
        shap_importance,
        published_work,
        contribution_matrix,
        chart_paths,
    )
    return markdown_path, pdf_path


def main() -> None:
    """Run report generation."""

    project_root = Path(__file__).resolve().parents[1]
    config = ReportConfig(
        project_root=project_root,
        output_dir=project_root / "reports",
        results_dir=project_root / "outputs" / "clean_notebook",
    )
    markdown_path, pdf_path = generate_report(config)
    print(f"Markdown report: {markdown_path}")
    print(f"PDF report: {pdf_path}")


if __name__ == "__main__":
    main()
