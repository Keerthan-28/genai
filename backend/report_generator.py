"""Sales report generator using local Ollama with deterministic fallback and exports."""

from __future__ import annotations

import hashlib
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.llm_client import OllamaClientWrapper
from backend.report_templates import (
    build_deterministic_sections,
    build_report_prompt,
    build_report_system_instruction,
    export_report_docx,
    export_report_pdf,
    export_report_txt,
    parse_markdown_sections,
)
from backend.schemas import DatasetMetadata, DatasetProfile, GeneratedReport, ReportSection
from config.settings import Settings, get_settings


class ReportGenerator:
    """Generates structured 11-section sales reports grounded in Python analytics."""

    def __init__(
        self,
        llm_client: OllamaClientWrapper | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm_client = llm_client or OllamaClientWrapper(self.settings)

    @staticmethod
    def _assemble_full_text(title: str, sections: list[ReportSection]) -> str:
        blocks = [f"# {title}", ""]
        for sec in sections:
            blocks.append(f"## {sec.section_number}. {sec.title}")
            blocks.append(sec.content)
            blocks.append("")
        return "\n".join(blocks).strip()

    def generate_report(
        self,
        metadata: DatasetMetadata,
        profile: DatasetProfile,
        analytics: dict[str, Any],
        use_llm: bool = True,
        save_files: bool = True,
        output_dir: Path | None = None,
    ) -> GeneratedReport:
        """Generate a structured 11-section sales report and persist TXT/DOCX/PDF files."""
        start = time.perf_counter()
        report_id = f"rep_{uuid.uuid4().hex[:10]}"
        title = f"VeriReport AI Retail Sales Report — {metadata.filename}"

        sections: list[ReportSection] = []
        generation_mode = "deterministic_fallback"
        model_used = "deterministic-python"

        if use_llm:
            try:
                prompt = build_report_prompt(metadata, profile, analytics)
                sys_inst = build_report_system_instruction()
                raw_text = self.llm_client.generate_text(
                    prompt=prompt,
                    system_instruction=sys_inst,
                    temperature=0.1,
                )
                parsed = parse_markdown_sections(raw_text)
                non_empty_count = sum(1 for s in parsed if s.content.strip())
                if non_empty_count >= 8:
                    # Fill any missing minor section deterministically so all 11 exist
                    fallback_sections = {
                        s.section_number: s
                        for s in build_deterministic_sections(metadata, profile, analytics)
                    }
                    sections = [
                        s if s.content.strip() else fallback_sections[s.section_number]
                        for s in parsed
                    ]
                    generation_mode = "ollama"
                    model_used = self.llm_client.resolve_model()
            except Exception:
                sections = []

        if not sections:
            sections = build_deterministic_sections(metadata, profile, analytics)
            generation_mode = "deterministic_fallback"
            model_used = "deterministic-python"

        full_text = self._assemble_full_text(title, sections)
        report_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
        elapsed = round(time.perf_counter() - start, 4)

        report = GeneratedReport(
            report_id=report_id,
            dataset_id=metadata.dataset_id,
            report_hash=report_hash,
            model_used=model_used,
            generation_mode=generation_mode,
            generated_at=datetime.utcnow().isoformat(),
            generation_time_seconds=elapsed,
            title=title,
            currency_code=analytics.get("currency_code", "GBP"),
            currency_symbol=analytics.get("currency_symbol", "£"),
            sections=sections,
            full_text=full_text,
        )

        if save_files:
            target_dir = output_dir or self.settings.reports_dir
            txt_path = export_report_txt(report, target_dir / f"{report_id}.txt")
            docx_path = export_report_docx(report, target_dir / f"{report_id}.docx")
            pdf_path = export_report_pdf(report, target_dir / f"{report_id}.pdf")
            report.txt_path = str(txt_path)
            report.docx_path = str(docx_path)
            report.pdf_path = str(pdf_path)

        return report
