"""Generate an Arabic PDF for the multi-dataset AF report."""

from __future__ import annotations

from pathlib import Path

import arabic_reshaper
from bidi.algorithm import get_display
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
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


ROOT = Path(__file__).resolve().parents[1]
REPORT_MD = ROOT / "reports" / "af_multidataset_client_report_ar.md"
REPORT_PDF = ROOT / "reports" / "af_multidataset_client_report_ar.pdf"


def rtl(text: object) -> str:
    """Prepare text for Arabic RTL rendering in ReportLab."""

    value = str(text).replace("**", "")
    return get_display(arabic_reshaper.reshape(value))


def register_font() -> str:
    """Register an Arabic-capable font and return its ReportLab name."""

    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/tahoma.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
    ]
    for path in candidates:
        if path.exists():
            pdfmetrics.registerFont(TTFont("ArabicFont", str(path)))
            return "ArabicFont"
    return "Helvetica"


def is_table_line(line: str) -> bool:
    """Return whether a markdown line is a table row."""

    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|")


def parse_table(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    """Parse a markdown table starting at index."""

    rows: list[list[str]] = []
    index = start
    while index < len(lines) and is_table_line(lines[index]):
        raw_cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
        if not all(set(cell) <= {"-", ":", " "} for cell in raw_cells):
            rows.append(raw_cells)
        index += 1
    return rows, index


def build_pdf() -> None:
    """Build the PDF from the markdown report."""

    font_name = register_font()
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ArabicTitle",
        parent=styles["Title"],
        fontName=font_name,
        fontSize=18,
        leading=24,
        alignment=TA_CENTER,
        spaceAfter=18,
    )
    heading_style = ParagraphStyle(
        "ArabicHeading",
        parent=styles["Heading2"],
        fontName=font_name,
        fontSize=14,
        leading=20,
        alignment=TA_RIGHT,
        textColor=colors.HexColor("#1f4e79"),
        spaceBefore=12,
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "ArabicBody",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=10.5,
        leading=16,
        alignment=TA_RIGHT,
        spaceAfter=6,
    )
    bullet_style = ParagraphStyle(
        "ArabicBullet",
        parent=body_style,
        leftIndent=12,
        firstLineIndent=0,
    )

    doc = SimpleDocTemplate(
        str(REPORT_PDF),
        pagesize=A4,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title="AF Multidataset Client Report",
    )

    lines = REPORT_MD.read_text(encoding="utf-8").splitlines()
    story = []
    index = 0
    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 4))
            index += 1
            continue
        if stripped.startswith("# "):
            story.append(Paragraph(rtl(stripped[2:]), title_style))
            index += 1
            continue
        if stripped.startswith("## "):
            if story:
                story.append(Spacer(1, 4))
            story.append(Paragraph(rtl(stripped[3:]), heading_style))
            index += 1
            continue
        if stripped.startswith("### "):
            story.append(Paragraph(rtl(stripped[4:]), heading_style))
            index += 1
            continue
        if is_table_line(stripped):
            rows, index = parse_table(lines, index)
            if rows:
                table_data = [[Paragraph(rtl(cell), body_style) for cell in row] for row in rows]
                table = Table(table_data, hAlign="RIGHT", repeatRows=1)
                table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9eaf7")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0b2f4f")),
                            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9aa6b2")),
                            ("FONTNAME", (0, 0), (-1, -1), font_name),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                            ("ALIGN", (0, 0), (-1, -1), "RIGHT"),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                            ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ]
                    )
                )
                story.append(table)
                story.append(Spacer(1, 8))
            continue
        if stripped.startswith("- "):
            story.append(Paragraph(rtl("• " + stripped[2:]), bullet_style))
            index += 1
            continue
        if stripped.startswith("```"):
            code_lines = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index])
                index += 1
            index += 1
            for code_line in code_lines:
                story.append(Paragraph(code_line.replace("<", "&lt;").replace(">", "&gt;"), body_style))
            continue
        story.append(Paragraph(rtl(stripped), body_style))
        index += 1

    doc.build(story)
    print(REPORT_PDF)


if __name__ == "__main__":
    build_pdf()
