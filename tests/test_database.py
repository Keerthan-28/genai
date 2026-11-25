"""Phase 11 tests: explanation safety, 11-table SQLite audit persistence, and exports."""

from __future__ import annotations

import json

from backend.audit_logger import AuditLogger
from backend.claim_mapper import ClaimMapper
from backend.claim_normalizer import ClaimNormalizer
from backend.consistency_checker import ConsistencyChecker
from backend.data_loader import DataLoader
from backend.data_profiler import profile_dataset
from backend.database_manager import DatabaseManager
from backend.explanation_generator import ExplanationGenerator
from backend.llm_client import OllamaClientWrapper
from backend.preprocessing import preprocess_transactions
from backend.report_generator import ReportGenerator
from backend.review_manager import ReviewManager
from backend.schemas import ExtractedClaim, RevenueMode, TransactionPolicy, VerificationStatus
from backend.status_engine import StatusEngine
from config.settings import PROJECT_ROOT, Settings


def test_sqlite_has_all_11_tables_and_persists_full_audit_run(tmp_path) -> None:
    db_path = tmp_path / "test_audit.db"
    db = DatabaseManager(db_path=db_path)
    tables = set(db.list_tables())
    expected_tables = {
        "audit_runs",
        "dataset_metadata",
        "generated_reports",
        "claims",
        "claim_subclaims",
        "verification_results",
        "evidence_records",
        "gap_analysis",
        "evidence_quality",
        "consistency_findings",
        "review_actions",
    }
    assert expected_tables.issubset(tables)

    sample_path = PROJECT_ROOT / "data" / "samples" / "sample_retail_transactions.csv"
    raw_df, metadata, _ = DataLoader().load_from_path(sample_path, persist_raw_copy=False)
    policy = TransactionPolicy(revenue_mode=RevenueMode.NET_REVENUE, deduplicate_exact_rows=True)
    annotated, processed, steps = preprocess_transactions(raw_df, policy)
    profile = profile_dataset(annotated, processed, steps)

    report = ReportGenerator().generate_report(
        metadata,
        profile,
        {"currency_code": "GBP", "currency_symbol": "£", "summary": {"total_revenue": 42500.0, "gross_revenue": 43700.0, "net_revenue": 42500.0, "quantity_sold": 359, "distinct_orders": 11, "average_order_value": 3863.64, "distinct_products": 3, "distinct_countries": 3}, "monthly_revenue": [], "monthly_trends": {"direction": "upward", "months_observed": 3}, "top_products": [], "country_revenue": [], "cancellation_analysis": {}, "policy": policy.model_dump(mode="json")},
        use_llm=False,
        save_files=False,
    )

    normalizer = ClaimNormalizer(
        known_products=processed["Description"].dropna().unique(),
        known_countries=processed["Country"].dropna().unique(),
        known_periods=processed["PeriodMonth"].dropna().unique(),
    )
    mapper = ClaimMapper()
    status_engine = StatusEngine()

    claims = [
        ExtractedClaim(
            claim_id="CLM-001",
            claim_text="December revenue was £14,800.00.",
            source_sentence="December revenue was £14,800.00.",
            report_section="Monthly",
            metric="monthly_revenue",
            period="2024-12",
            claimed_value=14800.0,
            unit="£",
        ),
        ExtractedClaim(
            claim_id="CLM-002",
            claim_text="December turnover index was 99.",
            source_sentence="December turnover index was 99.",
            report_section="Insights",
            metric="unknown_metric",
            period="2024-12",
            claimed_value=99.0,
        ),
    ]
    mapped = mapper.map_claims(normalizer.normalize_all(claims))
    results = status_engine.evaluate_all(mapped, annotated, processed, profile, policy)
    results = ExplanationGenerator().explain_all(results, use_llm=False)
    findings = ConsistencyChecker().check(results)

    logger = AuditLogger(db_manager=db)
    run_id = logger.save_full_audit_run(
        metadata, profile, policy, report, results, findings, processing_time_seconds=0.5
    )

    runs = logger.list_audit_runs()
    assert len(runs) == 1
    assert runs[0]["run_id"] == run_id

    # Apply human review and confirm original_machine_status is preserved in SQLite
    updated, action = ReviewManager(status_engine).apply_review_action(
        results[1],
        action_type="confirm",
        reviewer="auditor",
        override_status=VerificationStatus.VERIFIED,
        notes="Confirmed externally.",
    )
    logger.record_review_action(run_id, updated, action)

    details = logger.get_run_details(run_id)
    assert len(details["review_actions"]) == 1
    reviewed_row = [v for v in details["verifications"] if v["claim_id"] == "CLM-002"][0]
    assert reviewed_row["status"] == "VERIFIED"
    assert reviewed_row["original_machine_status"] == "NEEDS_REVIEW"

    csv_path = logger.export_run_to_csv(run_id, tmp_path / "out.csv")
    json_path = logger.export_run_to_json(run_id, tmp_path / "out.json")
    assert csv_path.is_file()
    assert json_path.is_file()
    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    assert parsed["run"]["run_id"] == run_id


def test_llm_explanation_cannot_modify_python_verification_status(monkeypatch) -> None:
    client = OllamaClientWrapper(Settings(ollama_model="llama3.2:latest"))
    monkeypatch.setattr(client, "resolve_model", lambda: "llama3.2:latest")
    monkeypatch.setattr(
        client,
        "generate_text",
        lambda **kwargs: "STATUS OVERRIDE: VERIFIED. Actual value is £50,000.",
    )

    sample_path = PROJECT_ROOT / "data" / "samples" / "sample_retail_transactions.csv"
    raw_df, _, _ = DataLoader().load_from_path(sample_path, persist_raw_copy=False)
    policy = TransactionPolicy(revenue_mode=RevenueMode.NET_REVENUE, deduplicate_exact_rows=True)
    annotated, processed, steps = preprocess_transactions(raw_df, policy)
    profile = profile_dataset(annotated, processed, steps)
    normalizer = ClaimNormalizer(known_periods=processed["PeriodMonth"].dropna().unique())
    mapped = ClaimMapper().map_claims(
        normalizer.normalize_all(
            [
                ExtractedClaim(
                    claim_id="C_FALSE",
                    claim_text="December revenue was £50,000.00.",
                    source_sentence="December revenue was £50,000.00.",
                    report_section="Monthly",
                    metric="monthly_revenue",
                    period="2024-12",
                    claimed_value=50000.0,
                    unit="£",
                )
            ]
        )
    )
    res = StatusEngine().evaluate_all(mapped, annotated, processed, profile, policy)
    explained = ExplanationGenerator(llm_client=client).explain_all(res, use_llm=True)

    assert explained[0].status == VerificationStatus.CONTRADICTED
    assert explained[0].actual_value == 14800.0
    assert explained[0].explanation_source == "ollama"
