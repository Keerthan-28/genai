"""Phase 8: Independent claim verification engine combining EvidenceGate, AnalyticsEngine, and GapAnalyzer."""

from __future__ import annotations

import pandas as pd

from backend.analytics_engine import AnalyticsEngine
from backend.evidence_gate import EvidenceGate
from backend.gap_analyzer import GapAnalyzer
from backend.schemas import (
    DatasetProfile,
    EvidenceGateResult,
    EvidenceQualityLevel,
    EvidenceQualityResult,
    EvidenceReason,
    GapAnalysisResult,
    GapType,
    MetricComputationResult,
    NormalizedClaim,
    TransactionPolicy,
)
from backend.tolerance_manager import ToleranceManager


class ClaimVerifier:
    """Independently recalculates actual values and prepares raw verification artifacts."""

    def __init__(
        self,
        analytics_engine: AnalyticsEngine | None = None,
        evidence_gate: EvidenceGate | None = None,
        tolerance_manager: ToleranceManager | None = None,
        gap_analyzer: GapAnalyzer | None = None,
    ) -> None:
        self.analytics = analytics_engine or AnalyticsEngine()
        self.evidence_gate = evidence_gate or EvidenceGate(self.analytics.registry)
        self.tolerance_manager = tolerance_manager or ToleranceManager(self.analytics.registry)
        self.gap_analyzer = gap_analyzer or GapAnalyzer()

    def _detect_alternative_period_match(
        self,
        claim: NormalizedClaim,
        claimed_val: float,
        processed_df: pd.DataFrame,
        annotated_df: pd.DataFrame,
        policy: TransactionPolicy,
        profile: DatasetProfile,
    ) -> str | None:
        if not claim.metric_id or claim.operation in ("growth_pct", "ranking"):
            return None
        for cov in profile.period_coverage:
            if cov.period_type != "month" or cov.period == claim.normalized_period:
                continue
            alt = self.analytics.compute_metric(
                metric_id=claim.metric_id,
                processed_df=processed_df,
                annotated_df=annotated_df,
                policy=policy,
                profile=profile,
                period=cov.period,
                entity=claim.normalized_entity,
                entity_type=claim.entity_type,
                operation=claim.operation,
            )
            if isinstance(alt.value, (int, float)):
                tol = self.tolerance_manager.get_tolerance(claim.metric_id, float(alt.value), claimed_val)
                if abs(claimed_val - float(alt.value)) <= tol:
                    return cov.period
        return None

    def _detect_alternative_metric_match(
        self,
        claim: NormalizedClaim,
        claimed_val: float,
        processed_df: pd.DataFrame,
        annotated_df: pd.DataFrame,
        policy: TransactionPolicy,
        profile: DatasetProfile,
    ) -> str | None:
        if claim.metric_id not in ("net_revenue", "gross_revenue", "total_revenue", "monthly_revenue"):
            return None
        for alt_metric in ("gross_revenue", "net_revenue"):
            if alt_metric == claim.metric_id:
                continue
            alt = self.analytics.compute_metric(
                metric_id=alt_metric,
                processed_df=processed_df,
                annotated_df=annotated_df,
                policy=policy,
                profile=profile,
                period=claim.normalized_period,
                entity=claim.normalized_entity,
                entity_type=claim.entity_type,
                operation="value",
            )
            if isinstance(alt.value, (int, float)):
                tol = self.tolerance_manager.get_tolerance(alt_metric, float(alt.value), claimed_val)
                if abs(claimed_val - float(alt.value)) <= tol:
                    return alt_metric
        return None

    def verify_atomic_claim(
        self,
        claim: NormalizedClaim,
        annotated_df: pd.DataFrame,
        processed_df: pd.DataFrame,
        profile: DatasetProfile,
        policy: TransactionPolicy,
    ) -> tuple[EvidenceGateResult, MetricComputationResult | None, GapAnalysisResult]:
        """Run EvidenceGate -> independent calculation -> GapAnalyzer for a single atomic claim."""
        gate = self.evidence_gate.evaluate(claim, annotated_df, processed_df, profile, policy)
        if not gate.has_sufficient_evidence or not claim.metric_id:
            gap = GapAnalysisResult(
                claim_id=claim.claim_id,
                claimed_value=claim.claimed_value,
                actual_value=None,
                gap_type=GapType.NOT_COMPARABLE,
                unit=claim.normalized_unit or "",
                comparison_method="insufficient_evidence",
                notes=gate.reason_detail,
            )
            return gate, None, gap

        calc = self.analytics.compute_metric(
            metric_id=claim.metric_id,
            processed_df=processed_df,
            annotated_df=annotated_df,
            policy=policy,
            profile=profile,
            period=claim.normalized_period,
            comparison_period=claim.normalized_comparison_period,
            entity=claim.normalized_entity,
            entity_type=claim.entity_type,
            operation=claim.operation,
        )

        if calc.error_reason or calc.value is None:
            updated_gate = EvidenceGateResult(
                claim_id=claim.claim_id,
                has_sufficient_evidence=False,
                reason_code=EvidenceReason.NO_MATCHING_RECORDS,
                reason_detail=calc.error_reason or "Calculation returned no value.",
                matching_records_count=calc.matching_rows,
                is_partial_period=calc.is_partial_period,
            )
            gap = GapAnalysisResult(
                claim_id=claim.claim_id,
                claimed_value=claim.claimed_value,
                actual_value=None,
                gap_type=GapType.NOT_COMPARABLE,
                unit=calc.unit,
                comparison_method="calculation_unavailable",
                notes=calc.error_reason or "Calculation unavailable.",
            )
            return updated_gate, calc, gap

        tol: float | None = None
        alt_period = None
        alt_metric = None
        if isinstance(calc.value, (int, float)) and isinstance(claim.claimed_value, (int, float)):
            tol = self.tolerance_manager.get_tolerance(claim.metric_id, float(calc.value), float(claim.claimed_value))
            if abs(float(claim.claimed_value) - float(calc.value)) > tol:
                alt_period = self._detect_alternative_period_match(
                    claim, float(claim.claimed_value), processed_df, annotated_df, policy, profile
                )
                alt_metric = self._detect_alternative_metric_match(
                    claim, float(claim.claimed_value), processed_df, annotated_df, policy, profile
                )

        gap = self.gap_analyzer.analyze(
            claim_id=claim.claim_id,
            claimed_value=claim.claimed_value,
            actual_value=calc.value,
            unit=claim.normalized_unit or calc.unit,
            tolerance=tol,
            operation=claim.operation,
            alternative_period_match=alt_period,
            alternative_metric_match=alt_metric,
        )
        return gate, calc, gap
