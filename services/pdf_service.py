"""
PDF generation for the Autonomous Cognitive Engine.

Turns a finished research report into a formatted, downloadable PDF with a
cover page, styled sections, real tables, clickable links, and page numbers —
built in memory with ReportLab.
"""

import re
from datetime import datetime
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _clean(text: str) -> str:
    """Make a line safe and displayable for ReportLab's paragraph parser."""
    replacements = {
        "\u2011": "-", "\u2013": "-", "\u2014": "-",
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2026": "...", "\u00a0": " ", "\u2022": "-", "\u2248": "~",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # Normalize all <br> variants to a proper self-closing line break.
    text = re.sub(r"<br\s*/?>", "<br/>", text, flags=re.IGNORECASE)
    # Remove other HTML tags, but KEEP <br/>.
    text = re.sub(r"</?(?!br\s*/?>)[a-zA-Z][^>]*>", "", text)
    # Escape ampersands so ReportLab doesn't read them as entity starts.
    text = text.replace("&", "&amp;")
    # Drop Markdown emphasis markers the base font won't render.
    text = text.replace("**", "").replace("__", "").replace("*", "")
    # Drop any remaining non-ASCII characters the base font can't show.
    text = text.encode("ascii", "ignore").decode("ascii")
    return text


def _link(text: str) -> str:
    """Wrap bare URLs in clickable link tags (run AFTER _clean)."""
    return re.sub(
        r"(https?://[^\s)<]+)",
        r'<a href="\1" color="blue">\1</a>',
        text,
    )


def _is_table_row(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") and s.count("|") >= 2


def _is_separator_row(line: str) -> bool:
    s = line.strip()
    return bool(s) and set(s) <= set("|-: ")


def _split_row(line: str) -> list:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _build_table(block: list, styles):
    """Turn a Markdown table block (lines of |...|) into a ReportLab Table."""
    rows = [_split_row(l) for l in block if not _is_separator_row(l)]
    if not rows:
        return Spacer(1, 0.1 * cm)

    n_cols = max(len(r) for r in rows)
    for r in rows:
        r += [""] * (n_cols - len(r))  # pad short rows to equal width

    head_style = ParagraphStyle(
        "THead", parent=styles["BodyText"], fontSize=8, leading=11,
        textColor=colors.white, wordWrap="CJK",
    )
    cell_style = ParagraphStyle(
        "TCell", parent=styles["BodyText"], fontSize=8, leading=11,
        wordWrap="CJK",
    )

    data = []
    for idx, row in enumerate(rows):
        style = head_style if idx == 0 else cell_style
        data.append([Paragraph(_link(_clean(c)), style) for c in row])

    usable_width = A4[0] - 4 * cm  # page width minus 2cm margins each side
    col_width = usable_width / n_cols

    table = Table(data, colWidths=[col_width] * n_cols, repeatRows=1)
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4b5563")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#9ca3af")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    return table


def _add_page_number(canvas, doc) -> None:
    page = canvas.getPageNumber()
    if page > 1:  # skip the cover page
        canvas.setFont("Helvetica", 9)
        canvas.setFillGray(0.5)
        canvas.drawCentredString(A4[0] / 2, 1.2 * cm, f"Page {page - 1}")


def _markdown_to_flowables(report: str, styles) -> list:
    """Convert the report's light Markdown into ReportLab flowables."""
    flowables = []
    lines = report.splitlines()
    i, n = 0, len(lines)

    while i < n:
        line = lines[i].rstrip()

        if not line.strip():
            flowables.append(Spacer(1, 0.2 * cm))
            i += 1
            continue

        # Table block: collect all consecutive table rows, build one table.
        if _is_table_row(line):
            block = []
            while i < n and _is_table_row(lines[i]):
                block.append(lines[i])
                i += 1
            flowables.append(_build_table(block, styles))
            flowables.append(Spacer(1, 0.3 * cm))
            continue

        # Headings.
        if line.startswith("### "):
            flowables.append(Paragraph(_clean(line[4:]), styles["Heading3"]))
        elif line.startswith("## "):
            flowables.append(Paragraph(_clean(line[3:]), styles["Heading2"]))
        elif line.startswith("# "):
            flowables.append(Paragraph(_clean(line[2:]), styles["Heading1"]))
        else:
            text = re.sub(r"^[-*]\s+", "- ", line)  # normalize bullets
            flowables.append(Paragraph(_link(_clean(text)), styles["BodyText"]))
        i += 1

    return flowables


def generate_report_pdf(query: str, report: str) -> bytes:
    """Build the report PDF and return it as bytes."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=2 * cm, bottomMargin=2 * cm,
        leftMargin=2 * cm, rightMargin=2 * cm,
        title="Research Report",
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="CoverTitle", parent=styles["Title"], fontSize=28,
        leading=34, alignment=TA_CENTER, spaceAfter=20,
    ))
    styles.add(ParagraphStyle(
        name="CoverSub", parent=styles["Normal"], fontSize=13,
        leading=18, alignment=TA_CENTER, textColor="#555555",
    ))

    story = [Spacer(1, 6 * cm)]
    story.append(Paragraph("Autonomous Cognitive Engine", styles["CoverTitle"]))
    story.append(Paragraph("Research Report", styles["CoverSub"]))
    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph(f"<b>Topic:</b> {_clean(query)}", styles["CoverSub"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(f"Generated on {datetime.now():%B %d, %Y}", styles["CoverSub"]))
    story.append(PageBreak())

    story.extend(_markdown_to_flowables(report, styles))

    doc.build(story, onFirstPage=_add_page_number, onLaterPages=_add_page_number)
    return buffer.getvalue()