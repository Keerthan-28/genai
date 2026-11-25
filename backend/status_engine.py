"""Phase 9: Deterministic five-status verification engine."""

from __future__ import annotations

import pandas as pd

from backend.claim_verifier import ClaimVerifier
from backend.evidence_quality import EvidenceQualityAssessor
from backend.schemas import (
    ClaimVerificationResult,
    DatasetProfile,
    EvidenceGateResult,
    EvidenceQualityLevel,
    EvidenceQualityResult,
    EvidenceReason,
    GapAnalysisResult,
    GapType,
    NormalizedClaim,
    TransactionPolicy,
    VerificationStatus,
)


class StatusEngine:
    """Assigns one of the five canonical verification statuses deterministically."""

    def __init__(
        self,
        verifier: ClaimVerifier | None = None,
        quality_assessor: EvidenceQualityAssessor | None = None,
    ) -> None:
        self.verifier = verifier or ClaimVerifier()
        self.quality_assessor = quality_assessor or EvidenceQualityAssessor()

    @staticmethod
    def decide_atomic_status(
        claim: NormalizedClaim,
        gate: EvidenceGateResult,
        gap: GapAnalysisResult,
    ) -> tuple[VerificationStatus, bool, str | None]:
        """Deterministic rule matrix for atomic claim statuses."""
        # 1. Human review triggers take precedence over contradiction when ambiguous/incompatible
        if (
            gate.needs_human_review
            or claim.mapping_ambiguity
            or claim.entity_ambiguity
            or claim.period_ambiguity
            or claim.extraction_confidence < 0.60
            or (claim.ambiguity_flag and not gate.has_sufficient_evidence)
        ):
            reason = gate.reason_detail if not gate.has_sufficient_evidence else claim.mapping_reason or "Ambiguous claim requires review."
            return VerificationStatus.NEEDS_REVIEW, True, reason

        # 2. Missing/insufficient evidence must NEVER become CONTRADICTED
        if not gate.has_sufficient_evidence:
            return VerificationStatus.INSUFFICIENT_EVIDENCE, False, gate.reason_detail

        # 3. Non-comparable gaps (e.g., missing claimed value) -> NEEDS_REVIEW
        if gap.gap_type == GapType.NOT_COMPARABLE:
            return VerificationStatus.NEEDS_REVIEW, True, gap.notes

        # 4. Exact/within tolerance -> VERIFIED
        if gap.gap_type == GapType.EXACT_MATCH:
            return VerificationStatus.VERIFIED, False, None

        # 5. Sufficient evidence with discrepancy -> CONTRADICTED
        return VerificationStatus.CONTRADICTED, False, None

    @staticmethod
    def aggregate_compound_status(sub_results: list[ClaimVerificationResult]) -> VerificationStatus:
        """Derive parent compound claim status strictly from subclaim outcomes."""
        if not sub_results:
            return VerificationStatus.NEEDS_REVIEW
        statuses = {r.status for r in sub_results}
        if len(statuses) == 1:
            return next(iter(statuses))
        return VerificationStatus.PARTIALLY_VERIFIED

    def evaluate_claim(
        self,
        claim: NormalizedClaim,
        annotated_df: pd.DataFrame,
        processed_df: pd.DataFrame,
        profile: DatasetProfile,
        policy: TransactionPolicy,
    ) -> ClaimVerificationResult:
        # Compound claim -> evaluate every subclaim independently
        if claim.subclaims:
            sub_results = [
                self.evaluate_claim(sub, annotated_df, processed_df, profile, policy)
                for sub in claim.subclaims
            ]
            parent_status = self.aggregate_compound_status(sub_results)
            needs_review = any(r.needs_human_review for r in sub_results)
            review_reasons = [r.review_reason for r in sub_results if r.review_reason]

            # Derive aggregate evidence quality from subclaims
            levels = [r.evidence_quality.quality_level for r in sub_results]
            if EvidenceQualityLevel.LOW in levels:
                agg_level = EvidenceQualityLevel.LOW
            elif EvidenceQualityLevel.MEDIUM in levels:
                agg_level = EvidenceQualityLevel.MEDIUM
            else:
                agg_level = EvidenceQualityLevel.HIGH
            avg_q = round(sum(r.evidence_quality.score for r in sub_results) / len(sub_results), 4)

            parent_gate = EvidenceGateResult(
                claim_id=claim.claim_id,
                has_sufficient_evidence=all(r.evidence_gate.has_sufficient_evidence for r in sub_results),
                reason_code=(
                    EvidenceReason.SUFFICIENT_EVIDENCE
                    if all(r.evidence_gate.has_sufficient_evidence for r in sub_results)
                    else sub_results[0].evidence_gate.reason_code
                ),
                reason_detail=f"Compound claim evaluated across {len(sub_results)} subclaims.",
                matching_records_count=sum(r.evidence_gate.matching_records_count for r in sub_results),
                needs_human_review=needs_review,
            )
            parent_gap = GapAnalysisResult(
                claim_id=claim.claim_id,
                claimed_value=None,
                actual_value=None,
                gap_type=GapType.NOT_COMPARABLE,
                comparison_method="compound_subclaims",
                notes="Compound claim; inspect individual subclaim gaps.",
            )
            parent_quality = EvidenceQualityResult(
                claim_id=claim.claim_id,
                quality_level=agg_level,
                score=avg_q,
                dimension_findings={"subclaims_evaluated": str(len(sub_results))},
                explanation=f"Aggregate evidence quality across {len(sub_results)} subclaims is {agg_level.value}.",
            )
            return ClaimVerificationResult(
                claim_id=claim.claim_id,
                parent_claim_id=claim.parent_claim_id,
                status=parent_status,
                original_machine_status=parent_status,
                normalized_claim=claim,
                evidence_gate=parent_gate,
                actual_value=None,
                formula_used="Compound subclaim evaluation",
                applied_filters=claim.filters,
                gap_analysis=parent_gap,
                evidence_quality=parent_quality,
                needs_human_review=needs_review,
                review_reason="; ".join(review_reasons) if review_reasons else None,
                subclaim_results=sub_results,
            )

        # Atomic claim
        gate, calc, gap = self.verifier.verify_atomic_claim(
            claim, annotated_df, processed_df, profile, policy
        )
        quality = self.quality_assessor.assess(claim, gate, calc, profile)
        status, needs_review, review_reason = self.decide_atomic_status(claim, gate, gap)

        return ClaimVerificationResult(
            claim_id=claim.claim_id,
            parent_claim_id=claim.parent_claim_id,
            status=status,
            original_machine_status=status,
            normalized_claim=claim,
            evidence_gate=gate,
            actual_value=calc.value if calc else None,
            formula_used=calc.formula_used if calc else "",
            applied_filters=calc.filters_applied if calc else {},
            gap_analysis=gap,
            evidence_quality=quality,
            needs_human_review=needs_review,
            review_reason=review_reason,
            subclaim_results=[],
        )

    def evaluate_all(
        self,
        claims: list[NormalizedClaim],
        annotated_df: pd.DataFrame,
        processed_df: pd.DataFrame,
        profile: DatasetProfile,
        policy: TransactionPolicy,
    ) -> list[ClaimVerificationResult]:
        return [
            self.evaluate_claim(c, annotated_df, processed_df, profile, policy)
            for c in claims
        ]
