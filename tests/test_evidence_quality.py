"""Phase 9 tests: independent 7-dimension evidence quality assessment."""

from __future__ import annotations

from backend.evidence_quality import EvidenceQualityAssessor
from backend.schemas import (
    DatasetProfile,
    EvidenceGateResult,
    EvidenceQualityLevel,
    EvidenceReason,
    MetricComputationResult,
    NormalizedClaim,
)


def test_high_quality_can_coexist_with_contradiction_and_low_with_missing_evidence() -> None:
    assessor = EvidenceQualityAssessor()
    profile = DatasetProfile(
        total_raw_rows=100,
        total_processed_rows=100,
        total_columns=8,
        missing_rates={"InvoiceNo": 0.0, "Quantity": 0.0, "UnitPrice": 0.0},
        data_quality_score=0.98,
    )
    claim = NormalizedClaim(
        claim_id="Q1",
        claim_text="Total revenue was £99,999.",
        source_sentence="Total revenue was £99,999.",
        report_section="Summary",
        metric_id="total_revenue",
        mapping_confidence=0.95,
    )
    good_gate = EvidenceGateResult(
        claim_id="Q1",
        has_sufficient_evidence=True,
        reason_code=EvidenceReason.SUFFICIENT_EVIDENCE,
        reason_detail="Complete evidence.",
        matching_records_count=100,
        is_partial_period=False,
    )
    calc = MetricComputationResult(metric_id="total_revenue", operation="value", value=42500.0)

    res_high = assessor.assess(claim, good_gate, calc, profile)
    assert res_high.quality_level == EvidenceQualityLevel.HIGH
    assert res_high.score >= 0.9

    partial_gate = EvidenceGateResult(
        claim_id="Q2",
        has_sufficient_evidence=True,
        reason_code=EvidenceReason.SUFFICIENT_EVIDENCE,
        reason_detail="Partial period.",
        matching_records_count=10,
        is_partial_period=True,
    )
    res_med = assessor.assess(claim, partial_gate, calc, profile)
    assert res_med.quality_level == EvidenceQualityLevel.MEDIUM

    missing_gate = EvidenceGateResult(
        claim_id="Q3",
        has_sufficient_evidence=False,
        reason_code=EvidenceReason.MISSING_REPORTING_PERIOD,
        reason_detail="Missing period.",
        period_covered=False,
        matching_records_count=0,
    )
    res_low = assessor.assess(claim, missing_gate, None, profile)
    assert res_low.quality_level == EvidenceQualityLevel.LOW
