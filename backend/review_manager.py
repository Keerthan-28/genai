"""Phase 10: Human review queue and non-destructive re-verification workflow."""

from __future__ import annotations

import uuid

import pandas as pd

from backend.schemas import (
    ClaimVerificationResult,
    DatasetProfile,
    ReviewAction,
    TransactionPolicy,
    VerificationStatus,
)
from backend.status_engine import StatusEngine


class ReviewManager:
    """Manages human review queue and audit-preserving verification updates."""

    def __init__(self, status_engine: StatusEngine | None = None) -> None:
        self.status_engine = status_engine or StatusEngine()

    @staticmethod
    def get_review_queue(results: list[ClaimVerificationResult]) -> list[ClaimVerificationResult]:
        """Return claims requiring human review."""
        queue: list[ClaimVerificationResult] = []
        for r in results:
            if r.needs_human_review or r.status == VerificationStatus.NEEDS_REVIEW:
                queue.append(r)
        return queue

    def apply_review_action(
        self,
        result: ClaimVerificationResult,
        action_type: str,
        reviewer: str = "analyst",
        corrected_metric_id: str | None = None,
        corrected_period: str | None = None,
        corrected_entity: str | None = None,
        corrected_entity_type: str | None = None,
        override_status: VerificationStatus | None = None,
        external_evidence_ref: str | None = None,
        notes: str = "",
        annotated_df: pd.DataFrame | None = None,
        processed_df: pd.DataFrame | None = None,
        profile: DatasetProfile | None = None,
        policy: TransactionPolicy | None = None,
    ) -> tuple[ClaimVerificationResult, ReviewAction]:
        """Apply a human review action, preserving `original_machine_status`."""
        previous_status = result.status
        updated = result.model_copy(deep=True)

        # Preserve original automated decision immutably
        updated.original_machine_status = result.original_machine_status

        if corrected_metric_id:
            updated.normalized_claim.metric_id = corrected_metric_id
            updated.normalized_claim.mapping_ambiguity = False
            updated.normalized_claim.mapping_confidence = 1.0
            updated.normalized_claim.mapping_reason = f"Human reviewer mapped metric to '{corrected_metric_id}'."

        if corrected_period:
            updated.normalized_claim.normalized_period = corrected_period
            updated.normalized_claim.period_ambiguity = False

        if corrected_entity:
            updated.normalized_claim.normalized_entity = corrected_entity
            if corrected_entity_type:
                updated.normalized_claim.entity_type = corrected_entity_type
            updated.normalized_claim.entity_ambiguity = False

        updated.normalized_claim.ambiguity_flag = False

        # Re-run deterministic verification when data context is supplied
        if (
            action_type in ("remap", "clarify_period", "clarify_entity", "rerun")
            and annotated_df is not None
            and processed_df is not None
            and profile is not None
            and policy is not None
        ):
            reverified = self.status_engine.evaluate_claim(
                updated.normalized_claim, annotated_df, processed_df, profile, policy
            )
            # Keep the original machine status intact!
            reverified.original_machine_status = result.original_machine_status
            updated = reverified

        if action_type == "confirm":
            updated.status = override_status or VerificationStatus.VERIFIED
            updated.needs_human_review = False
        elif action_type == "reject":
            updated.status = override_status or VerificationStatus.CONTRADICTED
            updated.needs_human_review = False
        elif override_status is not None:
            updated.status = override_status
            updated.needs_human_review = override_status == VerificationStatus.NEEDS_REVIEW

        if notes:
            updated.review_reason = notes

        action = ReviewAction(
            review_id=f"REV-{uuid.uuid4().hex[:10]}",
            claim_id=result.claim_id,
            reviewer=reviewer,
            action_type=action_type,
            previous_status=previous_status,
            new_status=updated.status,
            corrected_metric_id=corrected_metric_id,
            corrected_period=corrected_period,
            corrected_entity=corrected_entity,
            external_evidence_ref=external_evidence_ref,
            notes=notes,
        )
        return updated, action
