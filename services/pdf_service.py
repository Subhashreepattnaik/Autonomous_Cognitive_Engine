"""
PDF generation for the Autonomous Cognitive Engine.

Turns a finished research report into a formatted, downloadable PDF with a
cover page, styled sections, and page numbers — built in memory with ReportLab.
"""

import re
from datetime import datetime
from io import BytesIO

from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)


def _add_page_number(canvas, doc) -> None:
    """Draw the page number at the bottom of every page (except the cover)."""
    page = canvas.getPageNumber()
    if page > 1:  # skip the cover page
        canvas.setFont("Helvetica", 9)
        canvas.setFillGray(0.5)
        canvas.drawCentredString(A4[0] / 2, 1.2 * cm, f"Page {page - 1}")

def _clean(text: str) -> str:
    """Make a line safe for ReportLab's mini-HTML paragraph parser."""
    # Normalize all <br> variants to a space.
    text = re.sub(r"<br\s*/?>", " ", text)
    # Remove any other stray HTML/Markdown-table tags.
    text = re.sub(r"</?[a-zA-Z][^>]*>", "", text)
    # Escape & so ReportLab doesn't treat it as an entity start.
    text = text.replace("&", "&amp;")
    # Strip Markdown bold/italic markers ReportLab won't render anyway.
    text = text.replace("**", "").replace("__", "")
    return text

def _markdown_to_flowables(report: str, styles) -> list:
    """Convert the report's light Markdown into ReportLab paragraphs."""
    flowables = []
    for raw_line in report.splitlines():
        line = raw_line.rstrip()
        if not line:
            flowables.append(Spacer(1, 0.2 * cm))
            continue

        # Headings: ## Heading  or  ### Heading
        if line.startswith("### "):
            flowables.append(Paragraph(_clean(line[4:]), styles["Heading3"]))
        elif line.startswith("## "):
            flowables.append(Paragraph(_clean(line[3:]), styles["Heading2"]))
        elif line.startswith("# "):
            flowables.append(Paragraph(_clean(line[2:]), styles["Heading1"]))
        else:
            text = re.sub(r"^[-*]\s+", "• ", line)  # bullets -> dot
            text = _clean(text)
            # Turn bare URLs into clickable links (after cleaning).
            text = re.sub(
                r"(https?://[^\s)<]+)",
                r'<a href="\1" color="blue">\1</a>',
                text,
            )
            flowables.append(Paragraph(text, styles["BodyText"]))
    return flowables


def generate_report_pdf(query: str, report: str) -> bytes:
    """Build the report PDF and return it as bytes.

    Args:
        query: The original research request (shown on the cover).
        report: The final report text (light Markdown).

    Returns:
        The complete PDF as a bytes object, ready to download.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        title="Research Report",
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="CoverTitle", parent=styles["Title"], fontSize=28,
            leading=34, alignment=TA_CENTER, spaceAfter=20,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CoverSub", parent=styles["Normal"], fontSize=13,
            leading=18, alignment=TA_CENTER, textColor="#555555",
        )
    )

    story = []

    # ---- Cover page ----
    story.append(Spacer(1, 6 * cm))
    story.append(Paragraph("Autonomous Cognitive Engine", styles["CoverTitle"]))
    story.append(Paragraph("Research Report", styles["CoverSub"]))
    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph(f"<b>Topic:</b> {query}", styles["CoverSub"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(
        Paragraph(
            f"Generated on {datetime.now():%B %d, %Y}", styles["CoverSub"]
        )
    )
    story.append(PageBreak())

    # ---- Report body ----
    story.extend(_markdown_to_flowables(report, styles))

    doc.build(
        story,
        onFirstPage=_add_page_number,
        onLaterPages=_add_page_number,
    )
    return buffer.getvalue()