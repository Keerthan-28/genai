"""Phase 7: Evidence sufficiency gate before numerical verification.

Ensures missing evidence, missing columns, uncovered periods, or incompatible
units are explicitly recorded and never misclassified as proof of falsity.
"""

from __future__ import annotations

import pandas as pd

from backend.metric_registry import MetricRegistry
from backend.schemas import (
    DatasetProfile,
    EvidenceGateResult,
    EvidenceReason,
    NormalizedClaim,
    TransactionPolicy,
)


def are_units_compatible(
    claim_unit: str | None,
    metric_unit: str,
    policy_currency_code: str,
) -> bool:
    """Check whether the claimed unit matches the metric and active currency policy."""
    if not claim_unit:
        return True
    cu = claim_unit.strip()
    if metric_unit == "currency":
        if cu.upper() in ("GBP", "USD", "INR", "EUR"):
            return cu.upper() == policy_currency_code.upper()
        return False
    if metric_unit == "percent":
        return cu.lower() in ("percent", "%", "pct")
    if metric_unit == "units":
        return cu.lower() in ("units", "unit", "qty")
    if metric_unit == "count":
        return cu.lower() in ("count", "units", "orders", "transactions")
    if metric_unit == "rank":
        return cu.lower() in ("rank", "ranking")
    return True


class EvidenceGate:
    """Evaluates whether sufficient, valid, policy-aligned evidence exists for a claim."""

    def __init__(self, registry: MetricRegistry | None = None) -> None:
        self.registry = registry or MetricRegistry()

    def evaluate(
        self,
        claim: NormalizedClaim,
        annotated_df: pd.DataFrame,
        processed_df: pd.DataFrame,
        profile: DatasetProfile,
        policy: TransactionPolicy,
    ) -> EvidenceGateResult:
        # 1. Data quality & empty processed dataset guard
        if profile.total_processed_rows == 0 or profile.data_quality_score < 0.35:
            return EvidenceGateResult(
                claim_id=claim.claim_id,
                has_sufficient_evidence=False,
                reason_code=EvidenceReason.INSUFFICIENT_DATA_QUALITY,
                reason_detail=(
                    f"Dataset quality score ({profile.data_quality_score:.2f}) or processed rows "
                    f"({profile.total_processed_rows}) is insufficient for reliable verification."
                ),
                data_quality_sufficient=False,
            )

        # 2. Ambiguous metric mapping -> human review
        if claim.mapping_ambiguity:
            return EvidenceGateResult(
                claim_id=claim.claim_id,
                has_sufficient_evidence=False,
                reason_code=EvidenceReason.UNSUPPORTED_METRIC,
                reason_detail=claim.mapping_reason or "Ambiguous metric mapping requires human review.",
                metric_supported=False,
                needs_human_review=True,
            )

        # 3. Unsupported metric check
        metric_def = self.registry.get(claim.metric_id) if claim.metric_id else None
        if metric_def is None:
            return EvidenceGateResult(
                claim_id=claim.claim_id,
                has_sufficient_evidence=False,
                reason_code=EvidenceReason.UNSUPPORTED_METRIC,
                reason_detail=claim.mapping_reason or "Metric is not supported by the transaction dataset.",
                metric_supported=False,
            )

        # 4. Required columns check
        missing_cols = [c for c in metric_def.required_columns if c not in annotated_df.columns or annotated_df[c].isna().all()]
        if missing_cols:
            return EvidenceGateResult(
                claim_id=claim.claim_id,
                has_sufficient_evidence=False,
                reason_code=EvidenceReason.MISSING_REQUIRED_COLUMN,
                reason_detail=f"Missing required column(s) {missing_cols} for metric '{metric_def.metric_id}'.",
                required_columns_present=False,
            )

        # 5. Unit compatibility check
        if not are_units_compatible(claim.normalized_unit, metric_def.unit, policy.currency_code):
            return EvidenceGateResult(
                claim_id=claim.claim_id,
                has_sufficient_evidence=False,
                reason_code=EvidenceReason.INCOMPATIBLE_UNITS,
                reason_detail=(
                    f"Claim unit '{claim.normalized_unit}' is incompatible with metric unit "
                    f"'{metric_def.unit}' and dataset currency '{policy.currency_code}'."
                ),
                units_compatible=False,
                needs_human_review=True,
            )

        # 6. Entity identifiability check
        if claim.entity_ambiguity:
            return EvidenceGateResult(
                claim_id=claim.claim_id,
                has_sufficient_evidence=False,
                reason_code=EvidenceReason.AMBIGUOUS_ENTITY,
                reason_detail=f"Entity '{claim.raw_entity}' matches multiple items or is ambiguous.",
                entity_identifiable=False,
                needs_human_review=True,
            )
        if claim.raw_entity and claim.entity_type is None:
            return EvidenceGateResult(
                claim_id=claim.claim_id,
                has_sufficient_evidence=False,
                reason_code=EvidenceReason.NO_MATCHING_RECORDS,
                reason_detail=f"Entity '{claim.raw_entity}' was not found in the dataset.",
                entity_identifiable=False,
                matching_records_count=0,
            )

        # 7. Reporting period coverage check
        covered_periods = {c.period for c in profile.period_coverage}
        if claim.period_ambiguity:
            return EvidenceGateResult(
                claim_id=claim.claim_id,
                has_sufficient_evidence=False,
                reason_code=EvidenceReason.MISSING_REPORTING_PERIOD,
                reason_detail=f"Reporting period '{claim.raw_period}' is ambiguous across dataset years.",
                period_covered=False,
                needs_human_review=True,
            )

        if claim.normalized_period and claim.normalized_period not in covered_periods:
            return EvidenceGateResult(
                claim_id=claim.claim_id,
                has_sufficient_evidence=False,
                reason_code=EvidenceReason.MISSING_REPORTING_PERIOD,
                reason_detail=f"Reporting period '{claim.normalized_period}' is not present in the dataset.",
                period_covered=False,
            )

        if claim.operation == "growth_pct" or metric_def.metric_id == "revenue_growth":
            if not claim.normalized_comparison_period:
                return EvidenceGateResult(
                    claim_id=claim.claim_id,
                    has_sufficient_evidence=False,
                    reason_code=EvidenceReason.MISSING_COMPARISON_PERIOD,
                    reason_detail="Comparison period is required for growth verification.",
                    comparison_period_covered=False,
                    needs_human_review=True,
                )
            if claim.normalized_comparison_period not in covered_periods:
                return EvidenceGateResult(
                    claim_id=claim.claim_id,
                    has_sufficient_evidence=False,
                    reason_code=EvidenceReason.MISSING_COMPARISON_PERIOD,
                    reason_detail=(
                        f"Comparison period '{claim.normalized_comparison_period}' is not covered by the dataset."
                    ),
                    comparison_period_covered=False,
                )

        # 8. Matching records check
        scoped = processed_df
        if metric_def.metric_id == "returns_amount":
            scoped = annotated_df[annotated_df["is_return"] & annotated_df["is_valid_numeric"] & annotated_df["is_valid_date"]]
        elif metric_def.metric_id == "cancellation_analysis":
            scoped = annotated_df[annotated_df["is_cancellation"] & annotated_df["is_valid_numeric"] & annotated_df["is_valid_date"]]
        elif metric_def.metric_id == "gross_revenue":
            scoped = annotated_df[annotated_df["is_gross_sale"] & annotated_df["is_valid_numeric"] & annotated_df["is_valid_date"]]

        if claim.normalized_period:
            if len(claim.normalized_period) == 7:
                scoped = scoped[scoped["PeriodMonth"] == claim.normalized_period]
            elif len(claim.normalized_period) == 4:
                scoped = scoped[scoped["PeriodYear"] == claim.normalized_period]

        if claim.normalized_entity and claim.entity_type == "product":
            scoped = scoped[
                scoped["Description"].fillna("").str.strip().str.lower() == claim.normalized_entity.lower()
            ]
        elif claim.normalized_entity and claim.entity_type == "country":
            scoped = scoped[
                scoped["Country"].fillna("").str.strip().str.lower() == claim.normalized_entity.lower()
            ]

        match_count = int(len(scoped))
        if match_count == 0 and metric_def.metric_id not in ("returns_amount", "cancellation_analysis"):
            return EvidenceGateResult(
                claim_id=claim.claim_id,
                has_sufficient_evidence=False,
                reason_code=EvidenceReason.NO_MATCHING_RECORDS,
                reason_detail="No matching transaction records found for the specified filters.",
                matching_records_count=0,
            )

        is_partial = bool(claim.normalized_period and claim.normalized_period in profile.partial_periods)

        return EvidenceGateResult(
            claim_id=claim.claim_id,
            has_sufficient_evidence=True,
            reason_code=EvidenceReason.SUFFICIENT_EVIDENCE,
            reason_detail=(
                "Sufficient evidence verified in transaction dataset"
                + (f" (note: partial period coverage for {claim.normalized_period})." if is_partial else ".")
            ),
            matching_records_count=match_count,
            is_partial_period=is_partial,
        )
