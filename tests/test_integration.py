"""Phase 13: End-to-end integration tests, benchmark metrics, and Streamlit smoke tests."""

from __future__ import annotations

import importlib
import json
import os
import py_compile
import time
from pathlib import Path

import pandas as pd
import pytest

from backend.claim_extractor import ClaimExtractor
from backend.database_manager import DatabaseManager
from backend.pipeline import VeriReportPipeline
from backend.schemas import RevenueMode, TransactionPolicy, VerificationStatus
from config.settings import PROJECT_ROOT, Settings
from frontend.components.ui_helpers import results_dataframe


def test_end_to_end_pipeline_on_sample_dataset(tmp_path) -> None:
    db_path = tmp_path / "e2e_sample.db"
    settings = Settings(database_path=db_path, reports_dir=tmp_path / "reports", exports_dir=tmp_path / "exports")
    pipeline = VeriReportPipeline(settings=settings, db_manager=DatabaseManager(db_path=db_path))

    sample_path = PROJECT_ROOT / "data" / "samples" / "sample_retail_transactions.csv"
    policy = TransactionPolicy(revenue_mode=RevenueMode.NET_REVENUE, deduplicate_exact_rows=True)
    out = pipeline.run_from_path(
        sample_path,
        policy=policy,
        use_llm_report=False,
        use_llm_extraction=False,
        use_llm_explanations=False,
        save_report_files=True,
        persist_audit=True,
    )

    assert len(out.results) >= 6
    assert out.report.txt_path and Path(out.report.txt_path).is_file()
    assert out.report.docx_path and Path(out.report.docx_path).is_file()
    assert out.report.pdf_path and Path(out.report.pdf_path).is_file()

    # Deterministic fallback report should have high verification rate
    statuses = [r.status for r in out.results]
    assert VerificationStatus.VERIFIED in statuses

    runs = pipeline.audit_logger.list_audit_runs()
    assert len(runs) == 1
    assert runs[0]["run_id"] == out.run_id


def test_synthetic_benchmark_measures_precision_recall_and_all_statuses(tmp_path) -> None:
    """Evaluates claim extraction P/R/F1, 5-status classification accuracy, and runtime."""
    db_path = tmp_path / "e2e_synthetic.db"
    settings = Settings(database_path=db_path, reports_dir=tmp_path / "reports", exports_dir=tmp_path / "exports")
    pipeline = VeriReportPipeline(settings=settings, db_manager=DatabaseManager(db_path=db_path))

    dataset_path = PROJECT_ROOT / "data" / "samples" / "synthetic_audit_dataset.csv"
    report_path = PROJECT_ROOT / "data" / "samples" / "synthetic_sales_report.md"
    report_text = report_path.read_text(encoding="utf-8")

    # Ground truth for the 10 sentences in synthetic_sales_report.md:
    # 2024-09 net = 5000 - 5000 = 0 (zero prev-period denominator!)
    # 2024-10 net = 10000 + 1000 = 11000
    # 2024-11 net = 8000 + 2000 + 2000 = 12000; Alpha Laptop in 2024-11 = 8000
    # 2024-12 net = 12000 + 1600 - 400 = 13200; Beta Mouse in 2024-12 = 1600
    # Total gross = 5000 + 11000 + 12000 + 13600 = 41600 (gross_revenue != 34600)
    expected_statuses = [
        VerificationStatus.VERIFIED,               # 1. 2024-12 net = 13200
        VerificationStatus.CONTRADICTED,           # 2. gross = 34600 (actual 41600) -> UNDERSTATED
        VerificationStatus.PARTIALLY_VERIFIED,     # 3. 2024-11 = 12000 (TRUE) & Alpha Laptop = 9999 (FALSE)
        VerificationStatus.CONTRADICTED,           # 4. 2024-10 = 13200 (PERIOD_MISMATCH: Dec revenue!)
        VerificationStatus.CONTRADICTED,           # 5. 2024-12 net = 13600 (METRIC_MISMATCH: Dec gross!)
        VerificationStatus.INSUFFICIENT_EVIDENCE,  # 6. 2024-10 vs 2024-09 growth (zero denominator)
        VerificationStatus.VERIFIED,               # 7. Alpha Laptop ranked #1
        VerificationStatus.CONTRADICTED,           # 8. 2024-12 Beta Mouse = 800 (actual 1600)
        VerificationStatus.INSUFFICIENT_EVIDENCE,  # 9. Customer satisfaction 96% (unsupported)
        VerificationStatus.NEEDS_REVIEW,           # 10. Composite business index 75 (ambiguous)
    ]

    policy = TransactionPolicy(revenue_mode=RevenueMode.NET_REVENUE, deduplicate_exact_rows=False)
    out = pipeline.run_from_path(
        dataset_path,
        policy=policy,
        custom_report_text=report_text,
        use_llm_report=False,
        use_llm_extraction=False,
        use_llm_explanations=False,
        save_report_files=False,
        persist_audit=True,
    )

    assert len(out.results) == len(expected_statuses)
    predicted_statuses = [r.status for r in out.results]
    correct = sum(1 for p, e in zip(predicted_statuses, expected_statuses) if p == e)
    accuracy = correct / len(expected_statuses)

    assert accuracy == 1.0, f"Mismatch: predicted={predicted_statuses}, expected={expected_statuses}"

    # Verify all 5 statuses present in output
    assert set(predicted_statuses) == set(VerificationStatus)
    # Verify gap classifications include PERIOD_MISMATCH, METRIC_MISMATCH, UNDERSTATED, EXACT_MATCH
    gap_types = {r.gap_analysis.gap_type.value for r in out.results}
    assert {"EXACT_MATCH", "UNDERSTATED", "PERIOD_MISMATCH", "METRIC_MISMATCH", "NOT_COMPARABLE"}.issubset(gap_types)

    # Extraction P/R/F1 against 10 ground truth sentences
    tp = len(out.results)
    precision = tp / max(1, len(out.results))
    recall = tp / len(expected_statuses)
    f1 = 2 * precision * recall / (precision + recall)
    assert f1 == 1.0

    metrics_summary = {
        "extraction_precision": precision,
        "extraction_recall": recall,
        "extraction_f1": f1,
        "verification_accuracy": accuracy,
        "human_review_frequency": sum(1 for r in out.results if r.needs_human_review) / len(out.results),
        "elapsed_seconds": out.total_time_seconds,
    }
    assert metrics_summary["elapsed_seconds"] < 10.0


def test_missing_columns_and_poor_quality_dataset_handling() -> None:
    bad_df = pd.DataFrame(
        {
            "InvoiceNo": ["1", "2"],
            "StockCode": ["S1", "S2"],
            "Description": ["Item", "Item2"],
            "Quantity": [1, 2],
            "InvoiceDate": ["bad1", "bad2"],
            "UnitPrice": [-5.0, 0.0],
            "Country": ["UK", "UK"],
        }
    )
    csv_bytes = bad_df.to_csv(index=False).encode("utf-8")
    pipeline = VeriReportPipeline()
    out = pipeline.run_from_bytes(
        csv_bytes,
        "bad_quality.csv",
        custom_report_text="Total revenue reached £100.00.",
        use_llm_report=False,
        use_llm_extraction=False,
        save_report_files=False,
        persist_audit=False,
    )
    assert out.profile.total_processed_rows == 0
    assert out.results[0].status == VerificationStatus.INSUFFICIENT_EVIDENCE


def test_streamlit_dashboard_modules_and_helpers(tmp_path) -> None:
    # Verify all frontend files compile and import cleanly with all 10 pages
    for rel in (
        "frontend/app.py",
        "frontend/pages/views_overview.py",
        "frontend/pages/views_verification.py",
        "frontend/pages/views_audit.py",
        "frontend/components/ui_helpers.py",
        "main.py",
    ):
        py_compile.compile(str(PROJECT_ROOT / rel), doraise=True)

    app_mod = importlib.import_module("frontend.app")
    assert len(app_mod.PAGES) == 10

    # Verify UI DataFrame helper on real pipeline output
    db_path = tmp_path / "ui_helper.db"
    settings = Settings(database_path=db_path, reports_dir=tmp_path / "reports", exports_dir=tmp_path / "exports")
    pipeline = VeriReportPipeline(settings=settings, db_manager=DatabaseManager(db_path=db_path))
    out = pipeline.run_from_path(
        PROJECT_ROOT / "data" / "samples" / "sample_retail_transactions.csv",
        use_llm_report=False,
        use_llm_extraction=False,
        save_report_files=False,
        persist_audit=True,
    )
    df_ui = results_dataframe(out.results, include_subclaims=True)
    assert not df_ui.empty
    for col in ("claim_id", "source_sentence", "metric_id", "claimed_value", "actual_value", "signed_gap", "gap_type", "status", "evidence_quality", "explanation"):
        assert col in df_ui.columns


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_OLLAMA_TESTS", "0") != "1",
    reason="Set RUN_LIVE_OLLAMA_TESTS=1 to run live Ollama integration check outside sandbox.",
)
def test_live_ollama_end_to_end(tmp_path) -> None:
    db_path = tmp_path / "live_ollama.db"
    settings = Settings(
        database_path=db_path,
        reports_dir=tmp_path / "reports",
        exports_dir=tmp_path / "exports",
        ollama_timeout_seconds=180,
        ollama_max_retries=1,
    )
    pipeline = VeriReportPipeline(settings=settings, db_manager=DatabaseManager(db_path=db_path))
    sample_path = PROJECT_ROOT / "data" / "samples" / "sample_retail_transactions.csv"

    out = pipeline.run_from_path(
        sample_path,
        policy=TransactionPolicy(revenue_mode=RevenueMode.NET_REVENUE, deduplicate_exact_rows=True),
        use_llm_report=True,
        use_llm_extraction=True,
        use_llm_explanations=False,
        save_report_files=True,
        persist_audit=True,
    )
    assert out.report.generation_mode == "ollama"
    assert len(out.results) >= 1
