"""Phase 10 tests: cross-claim consistency checking."""

from __future__ import annotations

from backend.consistency_checker import ConsistencyChecker
from backend.schemas import (
    ClaimVerificationResult,
    EvidenceGateResult,
    EvidenceQualityLevel,
    EvidenceQualityResult,
    EvidenceReason,
    GapAnalysisResult,
    NormalizedClaim,
    VerificationStatus,
)


def _make_result(
    cid: str,
    metric_id: str,
    val,
    period: str | None = None,
    comp_period: str | None = None,
    entity: str | None = None,
    entity_type: str | None = None,
    operation: str = "value",
) -> ClaimVerificationResult:
    nc = NormalizedClaim(
        claim_id=cid,
        claim_text=f"{cid} {val}",
        source_sentence=f"{cid} {val}",
        report_section="Summary",
        metric_id=metric_id,
        normalized_period=period,
        normalized_comparison_period=comp_period,
        normalized_entity=entity,
        entity_type=entity_type,
        operation=operation,
        claimed_value=val,
    )
    return ClaimVerificationResult(
        claim_id=cid,
        status=VerificationStatus.VERIFIED,
        original_machine_status=VerificationStatus.VERIFIED,
        normalized_claim=nc,
        evidence_gate=EvidenceGateResult(
            claim_id=cid,
            has_sufficient_evidence=True,
            reason_code=EvidenceReason.SUFFICIENT_EVIDENCE,
            reason_detail="OK",
        ),
        gap_analysis=GapAnalysisResult(claim_id=cid),
        evidence_quality=EvidenceQualityResult(
            claim_id=cid, quality_level=EvidenceQualityLevel.HIGH, score=1.0
        ),
    )


def test_consistency_checker_detects_sum_share_growth_and_ranking_conflicts() -> None:
    results = [
        _make_result("C_TOT", "total_revenue", 10000.0, period="2024-12", entity_type="overall"),
        _make_result("C_P1", "product_revenue", 7000.0, period="2024-12", entity="Laptop", entity_type="product"),
        _make_result("C_P2", "product_revenue", 5000.0, period="2024-12", entity="Mouse", entity_type="product"),
        _make_result("C_S1", "revenue_contribution", 70.0, period="2024-12", entity="Laptop", entity_type="product", operation="share_pct"),
        _make_result("C_S2", "revenue_contribution", 50.0, period="2024-12", entity="Mouse", entity_type="product", operation="share_pct"),
        _make_result("C_M1", "monthly_revenue", 8000.0, period="2024-11"),
        _make_result("C_M2", "monthly_revenue", 10000.0, period="2024-12"),
        _make_result("C_GR", "revenue_growth", 50.0, period="2024-12", comp_period="2024-11", operation="growth_pct"),
        _make_result("C_RK", "top_products", "Mouse", period="2024-12", operation="ranking"),
    ]

    findings = ConsistencyChecker().check(results)
    rel_types = {f.relationship_type for f in findings}
    assert "product_sum_exceeds_total_revenue" in rel_types
    assert "contribution_percentage_exceeds_100" in rel_types
    assert "monthly_growth_vs_monthly_revenues" in rel_types
    assert "ranking_vs_claimed_product_revenue" in rel_types
