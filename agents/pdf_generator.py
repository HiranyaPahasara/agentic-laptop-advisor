"""
Agent 4 — PDF Report Generator (reportlab)

Converts the final audited Markdown report into a downloadable PDF.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"


def _styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="MainTitle",
            parent=styles["Heading1"],
            fontSize=18,
            spaceAfter=12,
            textColor=colors.HexColor("#0F172A"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            parent=styles["Heading2"],
            fontSize=13,
            spaceBefore=12,
            spaceAfter=6,
            textColor=colors.HexColor("#1E293B"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyTextCustom",
            parent=styles["BodyText"],
            fontSize=10,
            leading=14,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallMeta",
            parent=styles["Normal"],
            fontSize=9,
            textColor=colors.HexColor("#64748B"),
            spaceAfter=10,
        )
    )
    return styles


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _parse_markdown_table(lines: list[str], start_idx: int) -> tuple[list[list[str]], int]:
    """Parse a GitHub-style Markdown table starting at start_idx."""
    rows: list[list[str]] = []
    i = start_idx
    while i < len(lines) and lines[i].strip().startswith("|"):
        line = lines[i].strip()
        # Skip separator rows like | --- | --- |
        if re.match(r"^\|\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?$", line):
            i += 1
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cells)
        i += 1
    return rows, i


def _build_table(rows: list[list[str]]) -> Table:
    """Create a ReportLab table from parsed Markdown rows."""
    if not rows:
        rows = [["No table data found"]]

    # Normalize column count
    col_count = max(len(r) for r in rows)
    normalized = [r + [""] * (col_count - len(r)) for r in rows]

    wrapped = [
        [Paragraph(_escape(cell), ParagraphStyle("cell", fontSize=8, leading=10)) for cell in row]
        for row in normalized
    ]

    table = Table(wrapped, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def markdown_report_to_pdf(report_markdown: str, output_path: Path | None = None) -> Path:
    """
    Agent 4 entry point.
    Convert final Markdown report into a PDF file and return its path.
    """
    if not report_markdown or not str(report_markdown).strip():
        raise ValueError("Final report is empty; cannot generate PDF.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = REPORTS_DIR / f"smart_specs_report_{stamp}.pdf"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = _styles()
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=0.7 * inch,
        leftMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
    )

    story = [
        Paragraph("Smart Specs — Laptop Recommendation Report", styles["MainTitle"]),
        Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            styles["SmallMeta"],
        ),
        Spacer(1, 6),
    ]

    lines = report_markdown.replace("\r\n", "\n").split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        if not line.strip():
            story.append(Spacer(1, 6))
            i += 1
            continue

        # Markdown table
        if line.strip().startswith("|"):
            rows, i = _parse_markdown_table(lines, i)
            story.append(_build_table(rows))
            story.append(Spacer(1, 8))
            continue

        # Headings
        if line.startswith("### "):
            story.append(Paragraph(_escape(line[4:].strip()), styles["Section"]))
            i += 1
            continue
        if line.startswith("## "):
            story.append(Paragraph(_escape(line[3:].strip()), styles["Section"]))
            i += 1
            continue
        if line.startswith("# "):
            story.append(Paragraph(_escape(line[2:].strip()), styles["MainTitle"]))
            i += 1
            continue

        # Bullets
        if line.lstrip().startswith(("- ", "* ")):
            text = line.lstrip()[2:].strip()
            story.append(Paragraph(f"• {_escape(text)}", styles["BodyTextCustom"]))
            i += 1
            continue

        # Bold-ish markdown leftover cleanup
        clean = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
        clean = re.sub(r"`(.+?)`", r"\1", clean)
        story.append(Paragraph(_escape(clean), styles["BodyTextCustom"]))
        i += 1

    doc.build(story)
    return output_path


def generate_pdf_bytes(report_markdown: str) -> bytes:
    """Build PDF and return raw bytes (useful for Streamlit download button)."""
    path = markdown_report_to_pdf(report_markdown)
    return path.read_bytes()


def save_pdf_for_download(report_markdown: str, filename: str | None = None) -> dict:
    """
    Save a PDF for UI download and return path + bytes.

    Returns:
      {
        "path": Path,
        "bytes": bytes,
        "filename": str
      }
    """
    if filename:
        output_path = REPORTS_DIR / filename
        path = markdown_report_to_pdf(report_markdown, output_path=output_path)
    else:
        path = markdown_report_to_pdf(report_markdown)

    data = path.read_bytes()
    return {
        "path": path,
        "bytes": data,
        "filename": path.name,
    }


if __name__ == "__main__":
    sample = """
## Final Recommendations
| Model | Key Specs | Est. Price (LKR) | Best For | Notes |
| --- | --- | --- | --- | --- |
| ASUS Vivobook 15 | i5, 16GB, 512GB | 180000-260000 | Coding | Check upgrades |
| Dell Inspiron 14 | i5, 16GB, 512GB | 220000-310000 | Portability | Weak for gaming |

## Feasibility Audit
- Both options can fit a 250000 LKR budget depending on config

## Buyer Warnings
- Prefer 16GB RAM for coding multitasking
- Confirm battery life in real reviews

## Critic Verdict
Approve with warnings.
"""
    out = markdown_report_to_pdf(sample)
    print(f"PDF created: {out}")
