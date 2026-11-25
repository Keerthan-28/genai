"""Prompt templates, deterministic fallback builder, and PDF/DOCX/TXT exporters."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import docx
import pymupdf

from backend.schemas import DatasetMetadata, DatasetProfile, GeneratedReport, ReportSection

REQUIRED_REPORT_SECTIONS: tuple[tuple[int, str], ...] = (
    (1, "Executive Summary"),
    (2, "Dataset Overview"),
    (3, "Overall Sales Performance"),
    (4, "Monthly Sales Analysis"),
    (5, "Product Performance"),
    (6, "Country-wise Sales Analysis"),
    (7, "Revenue Growth"),
    (8, "Key Business Insights"),
    (9, "Trends and Observations"),
    (10, "Data Limitations"),
    (11, "Conclusion"),
)


def split_sentences(text: str) -> list[str]:
    """Split prose into clean sentences without breaking decimal numbers."""
    cleaned = re.sub(r"\s+", " ", text.strip())
    if not cleaned:
        return []
    # Split on sentence-ending punctuation followed by whitespace + uppercase/digit/bullet
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9£$€₹\-\*])", cleaned)
    return [p.strip() for p in parts if p.strip()]


def build_report_system_instruction() -> str:
    return (
        "You are a senior retail financial analyst drafting an audit-grade sales report. "
        "Strict rules: Use ONLY the numbers, periods, products, countries, and limitations "
        "provided in the structured JSON payload. Never invent figures, never estimate missing "
        "metrics (such as profit, ROI, inventory, or customer satisfaction), preserve exact "
        "reporting periods and currency units, and explicitly note any partial periods."
    )


def build_report_prompt(
    metadata: DatasetMetadata,
    profile: DatasetProfile,
    analytics: dict[str, Any],
) -> str:
    """Build a compact structured prompt containing only Python-calculated metrics."""
    compact_analytics = {
        "currency_symbol": analytics.get("currency_symbol"),
        "currency_code": analytics.get("currency_code"),
        "policy_mode": analytics.get("policy", {}).get("revenue_mode"),
        "summary": analytics.get("summary"),
        "monthly_revenue": analytics.get("monthly_revenue", [])[:12],
        "latest_revenue_growth": analytics.get("latest_revenue_growth"),
        "monthly_trends": analytics.get("monthly_trends"),
        "top_products": analytics.get("top_products", [])[:5],
        "country_revenue": analytics.get("country_revenue", [])[:5],
        "cancellation_analysis": analytics.get("cancellation_analysis"),
    }
    payload = {
        "dataset": {
            "filename": metadata.filename,
            "dataset_id": metadata.dataset_id,
            "raw_rows": profile.total_raw_rows,
            "processed_rows": profile.total_processed_rows,
            "date_min": profile.date_min,
            "date_max": profile.date_max,
            "partial_periods": profile.partial_periods,
            "data_quality_score": profile.data_quality_score,
            "quality_notes": profile.quality_notes[:5],
        },
        "analytics": compact_analytics,
    }
    section_headings = "\n".join(f"## {num}. {title}" for num, title in REQUIRED_REPORT_SECTIONS)
    return (
        "Write a concise, audit-grade retail sales report with these exact 11 markdown headings "
        "(1-2 factual sentences per section):\n"
        f"{section_headings}\n\n"
        "Rules:\n"
        "- Cite only values present in the JSON below.\n"
        "- Include exact numbers with currency units and YYYY-MM periods.\n"
        "- Explicitly mention any partial periods listed under `partial_periods`.\n\n"
        "Structured Analytics JSON:\n"
        f"{json.dumps(payload, separators=(',', ':'), default=str)}"
    )


def build_deterministic_sections(
    metadata: DatasetMetadata,
    profile: DatasetProfile,
    analytics: dict[str, Any],
) -> list[ReportSection]:
    """Deterministic structured 11-section report grounded 100% in Python analytics."""
    sym = analytics.get("currency_symbol", "£")
    s = analytics["summary"]
    monthly = analytics.get("monthly_revenue", [])
    products = analytics.get("top_products", [])
    countries = analytics.get("country_revenue", [])
    cancels = analytics.get("cancellation_analysis", {})
    growth = analytics.get("latest_revenue_growth")
    partial_str = ", ".join(profile.partial_periods) if profile.partial_periods else "None"

    top_prod_text = (
        f"{products[0]['product']} ranked #1 with {sym}{products[0]['revenue']:,.2f} "
        f"({products[0]['revenue_contribution_pct']:.2f}% of total revenue)"
        if products
        else "No product records were available."
    )
    top_country_text = (
        f"{countries[0]['country']} led country revenue at {sym}{countries[0]['revenue']:,.2f} "
        f"({countries[0]['revenue_contribution_pct']:.2f}% share)"
        if countries
        else "No country records were available."
    )

    monthly_lines = [
        f"In {m['period']}, revenue reached {sym}{m['revenue']:,.2f} across {m['orders']} order(s)"
        + (
            f" (growth of {m['mom_growth_pct']:.2f}% vs {m['previous_period']})"
            if m.get("mom_growth_pct") is not None
            else ""
        )
        + (" [partial period coverage]." if not m.get("is_complete_period") else ".")
        for m in monthly
    ]
    product_lines = [
        f"Rank #{p['rank']} product {p['product']} generated {sym}{p['revenue']:,.2f} "
        f"from {p['quantity']:,.0f} units ({p['revenue_contribution_pct']:.2f}% contribution)."
        for p in products[:5]
    ]
    country_lines = [
        f"{c['country']} recorded {sym}{c['revenue']:,.2f} "
        f"({c['revenue_contribution_pct']:.2f}% of total revenue) across {c['orders']} order(s)."
        for c in countries[:5]
    ]

    growth_sentence = (
        f"Latest month-over-month revenue growth in {growth['period']} compared with "
        f"{growth['comparison_period']} was {growth['growth_pct']:.2f}%."
        if growth
        else "Insufficient sequential months with non-zero revenue to compute month-over-month growth."
    )

    raw_sections: list[tuple[int, str, str]] = [
        (
            1,
            "Executive Summary",
            f"Analysis of {metadata.filename} ({profile.date_min} to {profile.date_max}) recorded "
            f"total revenue of {sym}{s['total_revenue']:,.2f} across {s['distinct_orders']} distinct orders. "
            f"Gross revenue was {sym}{s['gross_revenue']:,.2f}, while {top_prod_text}.",
        ),
        (
            2,
            "Dataset Overview",
            f"The dataset contains {profile.total_raw_rows} raw rows and {profile.total_processed_rows} processed rows "
            f"spanning {profile.date_min} to {profile.date_max} across {s['distinct_products']} products and "
            f"{s['distinct_countries']} countries. Data quality score is {profile.data_quality_score:.2f}.",
        ),
        (
            3,
            "Overall Sales Performance",
            f"Total revenue under the active policy was {sym}{s['total_revenue']:,.2f}, net revenue was "
            f"{sym}{s['net_revenue']:,.2f}, gross revenue was {sym}{s['gross_revenue']:,.2f}, "
            f"quantity sold totaled {s['quantity_sold']:,.0f} units, and average order value was "
            f"{sym}{s['average_order_value']:,.2f}.",
        ),
        (
            4,
            "Monthly Sales Analysis",
            " ".join(monthly_lines) if monthly_lines else "No valid monthly records available.",
        ),
        (
            5,
            "Product Performance",
            " ".join(product_lines) if product_lines else "No product performance data available.",
        ),
        (
            6,
            "Country-wise Sales Analysis",
            f"{top_country_text}. " + (" ".join(country_lines[1:]) if len(country_lines) > 1 else ""),
        ),
        (
            7,
            "Revenue Growth",
            growth_sentence,
        ),
        (
            8,
            "Key Business Insights",
            f"Returns amounted to {sym}{cancels.get('return_amount', 0.0):,.2f} across "
            f"{cancels.get('return_transactions', 0)} transaction(s), and cancellations totaled "
            f"{sym}{cancels.get('cancellation_amount', 0.0):,.2f} across "
            f"{cancels.get('cancellation_transactions', 0)} transaction(s) "
            f"({cancels.get('cancellation_rate_pct', 0.0):.2f}% cancellation rate).",
        ),
        (
            9,
            "Trends and Observations",
            f"Observed monthly revenue trend direction across {analytics['monthly_trends']['months_observed']} "
            f"month(s) is {analytics['monthly_trends']['direction']}.",
        ),
        (
            10,
            "Data Limitations",
            f"Partial period coverage applies to: {partial_str}. "
            + (" ".join(profile.quality_notes) if profile.quality_notes else "No additional quality issues logged."),
        ),
        (
            11,
            "Conclusion",
            f"All figures remain traceable to dataset {metadata.dataset_id} under policy mode "
            f"{analytics['policy']['revenue_mode']}, subject to documented data limitations.",
        ),
    ]

    return [
        ReportSection(
            section_number=num,
            title=title,
            content=content.strip(),
            sentences=split_sentences(content),
        )
        for num, title, content in raw_sections
    ]


def parse_markdown_sections(markdown_text: str) -> list[ReportSection]:
    """Parse LLM markdown into the canonical 11 ReportSection objects."""
    headings_pattern = re.compile(r"(?m)^#{1,3}\s*(?:\d+\s*[\.\)]\s*)?(.+?)\s*$")
    matches = list(headings_pattern.finditer(markdown_text))
    extracted_by_title: dict[str, str] = {}

    for i, match in enumerate(matches):
        heading_text = re.sub(r"^\d+\s*[\.\)]\s*", "", match.group(1)).strip().lower()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_text)
        body = markdown_text[start:end].strip()
        if body:
            extracted_by_title[heading_text] = body

    sections: list[ReportSection] = []
    for num, canonical_title in REQUIRED_REPORT_SECTIONS:
        content = extracted_by_title.get(canonical_title.lower(), "")
        if not content:
            # Partial title match fallback
            for k, v in extracted_by_title.items():
                if canonical_title.lower() in k or k in canonical_title.lower():
                    content = v
                    break
        sections.append(
            ReportSection(
                section_number=num,
                title=canonical_title,
                content=content,
                sentences=split_sentences(content),
            )
        )
    return sections


def export_report_txt(report: GeneratedReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.full_text, encoding="utf-8")
    return path


def export_report_docx(report: GeneratedReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = docx.Document()
    doc.add_heading(report.title, level=0)
    doc.add_paragraph(
        f"Report ID: {report.report_id} | Dataset ID: {report.dataset_id} | "
        f"Model: {report.model_used} ({report.generation_mode}) | Generated: {report.generated_at}"
    )
    for sec in report.sections:
        doc.add_heading(f"{sec.section_number}. {sec.title}", level=1)
        doc.add_paragraph(sec.content or "No content generated for this section.")
    doc.save(str(path))
    return path


def export_report_pdf(report: GeneratedReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf_doc = pymupdf.open()
    page = pdf_doc.new_page(width=595, height=842)
    margin = 45
    y = 50.0
    line_height = 14.0

    def _write_wrapped(text: str, font_size: float = 10.5, bold: bool = False) -> None:
        nonlocal page, y
        fontname = "helv" if not bold else "hebo"
        max_chars = int((595 - 2 * margin) / (font_size * 0.52))
        words = text.split()
        lines: list[str] = []
        current = ""
        for w in words:
            candidate = f"{current} {w}".strip()
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = w
        if current:
            lines.append(current)

        for ln in lines:
            if y > 790:
                page = pdf_doc.new_page(width=595, height=842)
                y = 50.0
            page.insert_text((margin, y), ln, fontsize=font_size, fontname=fontname)
            y += line_height

    _write_wrapped(report.title, font_size=15.0, bold=True)
    y += 4
    _write_wrapped(
        f"Report ID: {report.report_id} | Dataset: {report.dataset_id} | "
        f"Generator: {report.model_used} ({report.generation_mode})",
        font_size=9.0,
    )
    y += 10

    for sec in report.sections:
        _write_wrapped(f"{sec.section_number}. {sec.title}", font_size=12.0, bold=True)
        y += 2
        _write_wrapped(sec.content or "N/A", font_size=10.0, bold=False)
        y += 8

    pdf_doc.save(str(path))
    pdf_doc.close()
    return path
