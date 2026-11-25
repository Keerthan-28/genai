"""Phase 4 tests: LLM report generation, fallback, section/sentence preservation, and exports."""

from __future__ import annotations

import docx
import pymupdf

from backend.analytics_engine import AnalyticsEngine
from backend.data_loader import DataLoader
from backend.data_profiler import profile_dataset
from backend.llm_client import OllamaClientWrapper
from backend.preprocessing import preprocess_transactions
from backend.report_generator import ReportGenerator
from backend.report_templates import REQUIRED_REPORT_SECTIONS
from backend.schemas import RevenueMode, TransactionPolicy
from config.settings import PROJECT_ROOT, Settings


def _load_sample_context():
    sample_path = PROJECT_ROOT / "data" / "samples" / "sample_retail_transactions.csv"
    raw_df, metadata, _ = DataLoader().load_from_path(sample_path, persist_raw_copy=False)
    policy = TransactionPolicy(revenue_mode=RevenueMode.NET_REVENUE, deduplicate_exact_rows=True)
    annotated, processed, steps = preprocess_transactions(raw_df, policy)
    profile = profile_dataset(annotated, processed, steps)
    analytics = AnalyticsEngine().compute_full_analytics(annotated, processed, profile, policy)
    return metadata, profile, analytics


def test_deterministic_fallback_report_has_all_11_sections_and_exports(tmp_path) -> None:
    metadata, profile, analytics = _load_sample_context()
    unreachable_client = OllamaClientWrapper(
        Settings(ollama_base_url="http://127.0.0.1:59999", ollama_max_retries=1)
    )
    generator = ReportGenerator(llm_client=unreachable_client)
    report = generator.generate_report(
        metadata, profile, analytics, use_llm=True, save_files=True, output_dir=tmp_path
    )

    assert report.generation_mode == "deterministic_fallback"
    assert len(report.sections) == 11
    for (expected_num, expected_title), sec in zip(REQUIRED_REPORT_SECTIONS, report.sections):
        assert sec.section_number == expected_num
        assert sec.title == expected_title
        assert len(sec.sentences) >= 1
    assert len(report.report_hash) == 64

    # Verify TXT, DOCX, and PDF outputs exist and are readable
    assert report.txt_path and (tmp_path / f"{report.report_id}.txt").is_file()
    assert report.docx_path and (tmp_path / f"{report.report_id}.docx").is_file()
    assert report.pdf_path and (tmp_path / f"{report.report_id}.pdf").is_file()

    doc = docx.Document(report.docx_path)
    assert any("Executive Summary" in p.text for p in doc.paragraphs)

    pdf = pymupdf.open(report.pdf_path)
    pdf_text = "\n".join(page.get_text() for page in pdf)
    pdf.close()
    assert "Executive Summary" in pdf_text
    assert "42,500.00" in pdf_text


def test_mocked_ollama_report_generation(monkeypatch, tmp_path) -> None:
    metadata, profile, analytics = _load_sample_context()
    client = OllamaClientWrapper(Settings(ollama_model="llama3.2:latest"))
    monkeypatch.setattr(client, "resolve_model", lambda: "llama3.2:latest")

    mock_markdown = "\n\n".join(
        f"## {num}. {title}\nIn 2024-12, net revenue reached £14,800.00 for {title}."
        for num, title in REQUIRED_REPORT_SECTIONS
    )
    monkeypatch.setattr(client, "generate_text", lambda **kwargs: mock_markdown)

    generator = ReportGenerator(llm_client=client)
    report = generator.generate_report(
        metadata, profile, analytics, use_llm=True, save_files=False, output_dir=tmp_path
    )

    assert report.generation_mode == "ollama"
    assert report.model_used == "llama3.2:latest"
    assert len(report.sections) == 11
    assert report.sections[0].sentences[0].startswith("In 2024-12")
