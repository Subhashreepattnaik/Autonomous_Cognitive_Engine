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


def _add_page_number(canvas, doc) -> None:
    """Draw the page number at the bottom of every page except the cover."""
    page = canvas.getPageNumber()
    if page > 1:
        canvas.setFont("Helvetica", 9)
        canvas.setFillGray(0.5)
        canvas.drawCentredString(A4[0] / 2, 1.2 * cm, f"Page {page - 1}")


def _clean(text: str) -> str:
    """Make a line safe and displayable for ReportLab's paragraph parser."""
    replacements = {
        "\u2011": "-", "\u2013": "-", "\u2014": "-",
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2026": "...", "\u00a0": " ", "\u2022": "-", "\u2248": "~",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    text = re.sub(r"<br\s*/?>", " ", text)           # br tags -> space
    text = re.sub(r"</?[a-zA-Z][^>]*>", "", text)     # strip other tags
    text = text.replace("&", "&amp;")                 # escape ampersand
    text = text.replace("**", "").replace("__", "").replace("*", "")
    text = text.encode("ascii", "ignore").decode("ascii")  # drop leftovers
    return text


def _linkify_full(text: str) -> str:
    """Wrap bare URLs as clickable links showing the full URL (body text)."""
    return re.sub(
        r"(https?://[^\s)<]+)",
        r'<a href="\1" color="blue">\1</a>',
        text,
    )


def _linkify_short(text: str) -> str:
    """Wrap bare URLs as clickable links showing just the domain (table cells)."""
    def repl(match):
        url = match.group(1)
        domain = re.sub(r"^https?://(www\.)?", "", url).split("/")[0]
        return f'<a href="{url}" color="blue">{domain}</a>'

    return re.sub(r"(https?://[^\s)<]+)", repl, text)


def _is_table_row(line: str) -> bool:
    """A Markdown table row starts with '|' and has at least two pipes."""
    s = line.strip()
    return s.startswith("|") and s.count("|") >= 2


def _is_separator_row(line: str) -> bool:
    """A separator row looks like |---|---| (only dashes, colons, spaces)."""
    cells = line.strip().strip("|").split("|")
    return all(c.strip() and set(c.strip()) <= set("-: ") for c in cells)


def _parse_row(line: str) -> list:
    """Split a '| a | b | c |' row into ['a', 'b', 'c']."""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _table_from_block(block: list, styles) -> Table | None:
    """Build a ReportLab Table from a Markdown table block, or None if empty."""
    rows = [_parse_row(ln) for ln in block if not _is_separator_row(ln)]
    if not rows:
        return None

    ncols = max(len(r) for r in rows)

    cell_style = ParagraphStyle(
        "tcell", parent=styles["BodyText"], fontSize=8, leading=10,
        wordWrap="CJK",
    )
    head_style = ParagraphStyle(
        "thead", parent=styles["BodyText"], fontSize=8, leading=10,
        wordWrap="CJK", textColor=colors.white, fontName="Helvetica-Bold",
    )

    data = []
    for r_idx, row in enumerate(rows):
        style = head_style if r_idx == 0 else cell_style
        cells = [
            Paragraph(
                _linkify_short(_clean(row[c] if c < len(row) else "")), style
            )
            for c in range(ncols)
        ]
        data.append(cells)

    usable_width = A4[0] - 4 * cm  # page width minus left+right margins
    col_w = usable_width / ncols
    table = Table(data, colWidths=[col_w] * ncols, repeatRows=1)
    table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4f46e5")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ])
    )
    return table


def _markdown_to_flowables(report: str, styles) -> list:
    """Convert the report's light Markdown into ReportLab flowables."""
    flowables = []
    lines = report.splitlines()
    i, n = 0, len(lines)

    while i < n:
        line = lines[i].rstrip()

        if not line:
            flowables.append(Spacer(1, 0.2 * cm))
            i += 1
            continue

        # --- Table block: collect consecutive table rows ---
        if _is_table_row(line):
            block = []
            while i < n and _is_table_row(lines[i].rstrip()):
                block.append(lines[i].rstrip())
                i += 1
            try:
                table = _table_from_block(block, styles)
            except Exception:
                table = None
            if table is not None:
                flowables.append(Spacer(1, 0.15 * cm))
                flowables.append(table)
                flowables.append(Spacer(1, 0.15 * cm))
            else:  # fallback: render the rows as plain text
                for bl in block:
                    flowables.append(Paragraph(_clean(bl), styles["BodyText"]))
            continue

        # --- Headings ---
        if line.startswith("### "):
            flowables.append(Paragraph(_clean(line[4:]), styles["Heading3"]))
        elif line.startswith("## "):
            flowables.append(Paragraph(_clean(line[3:]), styles["Heading2"]))
        elif line.startswith("# "):
            flowables.append(Paragraph(_clean(line[2:]), styles["Heading1"]))
        else:
            text = re.sub(r"^[-*]\s+", "- ", line)  # bullets -> dash
            text = _linkify_full(_clean(text))
            flowables.append(Paragraph(text, styles["BodyText"]))
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

    story = [
        Spacer(1, 6 * cm),
        Paragraph("Autonomous Cognitive Engine", styles["CoverTitle"]),
        Paragraph("Research Report", styles["CoverSub"]),
        Spacer(1, 1 * cm),
        Paragraph(f"<b>Topic:</b> {_clean(query)}", styles["CoverSub"]),
        Spacer(1, 0.4 * cm),
        Paragraph(f"Generated on {datetime.now():%B %d, %Y}", styles["CoverSub"]),
        PageBreak(),
    ]
    story.extend(_markdown_to_flowables(report, styles))

    doc.build(story, onFirstPage=_add_page_number, onLaterPages=_add_page_number)
    return buffer.getvalue()