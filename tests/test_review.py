"""Phase 10 tests: human review workflow and original decision preservation."""

from __future__ import annotations

import pytest

from backend.claim_mapper import ClaimMapper
from backend.claim_normalizer import ClaimNormalizer
from backend.data_loader import DataLoader
from backend.data_profiler import profile_dataset
from backend.preprocessing import preprocess_transactions
from backend.review_manager import ReviewManager
from backend.schemas import ExtractedClaim, RevenueMode, TransactionPolicy, VerificationStatus
from backend.status_engine import StatusEngine
from config.settings import PROJECT_ROOT


def test_human_review_remapping_preserves_original_machine_status() -> None:
    sample_path = PROJECT_ROOT / "data" / "samples" / "sample_retail_transactions.csv"
    raw_df, _, _ = DataLoader().load_from_path(sample_path, persist_raw_copy=False)
    policy = TransactionPolicy(revenue_mode=RevenueMode.NET_REVENUE, deduplicate_exact_rows=True)
    annotated, processed, steps = preprocess_transactions(raw_df, policy)
    profile = profile_dataset(annotated, processed, steps)

    normalizer = ClaimNormalizer(
        known_products=processed["Description"].dropna().unique(),
        known_countries=processed["Country"].dropna().unique(),
        known_periods=processed["PeriodMonth"].dropna().unique(),
    )
    mapper = ClaimMapper()
    status_engine = StatusEngine()
    review_mgr = ReviewManager(status_engine=status_engine)

    ambiguous = ExtractedClaim(
        claim_id="REV-1",
        claim_text="December turnover reached £14,800.00.",
        source_sentence="December turnover reached £14,800.00.",
        report_section="Monthly",
        metric="unknown_metric",
        period="2024-12",
        claimed_value=14800.0,
        unit="£",
    )
    mapped = mapper.map_claims(normalizer.normalize_all([ambiguous]))
    initial = status_engine.evaluate_all(mapped, annotated, processed, profile, policy)

    assert len(review_mgr.get_review_queue(initial)) == 1
    assert initial[0].status == VerificationStatus.NEEDS_REVIEW
    assert initial[0].original_machine_status == VerificationStatus.NEEDS_REVIEW

    updated, action = review_mgr.apply_review_action(
        initial[0],
        action_type="remap",
        reviewer="senior_auditor",
        corrected_metric_id="monthly_revenue",
        notes="Turnover refers to monthly net revenue.",
        annotated_df=annotated,
        processed_df=processed,
        profile=profile,
        policy=policy,
    )

    assert updated.status == VerificationStatus.VERIFIED
    assert updated.original_machine_status == VerificationStatus.NEEDS_REVIEW
    assert updated.actual_value == pytest.approx(14800.0)
    assert action.previous_status == VerificationStatus.NEEDS_REVIEW
    assert action.new_status == VerificationStatus.VERIFIED
    assert action.corrected_metric_id == "monthly_revenue"
