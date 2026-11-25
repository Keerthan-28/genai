"""Reusable Streamlit UI helpers, status styling, and flattening utilities."""

from __future__ import annotations

from typing import Any

import pandas as pd

from backend.schemas import ClaimVerificationResult, VerificationStatus

STATUS_COLORS: dict[str, str] = {
    VerificationStatus.VERIFIED.value: "#16a34a",
    VerificationStatus.CONTRADICTED.value: "#dc2626",
    VerificationStatus.INSUFFICIENT_EVIDENCE.value: "#64748b",
    VerificationStatus.NEEDS_REVIEW.value: "#d97706",
    VerificationStatus.PARTIALLY_VERIFIED.value: "#2563eb",
}

STATUS_BADGE_LABELS: dict[str, str] = {
    VerificationStatus.VERIFIED.value: "VERIFIED",
    VerificationStatus.CONTRADICTED.value: "CONTRADICTED",
    VerificationStatus.INSUFFICIENT_EVIDENCE.value: "INSUFFICIENT EVIDENCE",
    VerificationStatus.NEEDS_REVIEW.value: "NEEDS REVIEW",
    VerificationStatus.PARTIALLY_VERIFIED.value: "PARTIALLY VERIFIED",
}


def flatten_claim_rows(results: list[ClaimVerificationResult], include_subclaims: bool = True) -> list[dict[str, Any]]:
    """Convert nested ClaimVerificationResult objects into flat dicts for tables and filters."""
    rows: list[dict[str, Any]] = []

    def _to_row(r: ClaimVerificationResult, is_subclaim: bool = False) -> dict[str, Any]:
        nc = r.normalized_claim
        g = r.gap_analysis
        return {
            "claim_id": r.claim_id,
            "parent_claim_id": r.parent_claim_id or "",
            "is_subclaim": is_subclaim,
            "section": nc.report_section,
            "source_sentence": nc.source_sentence,
            "claim_text": nc.claim_text,
            "metric_id": nc.metric_id or "unmapped",
            "entity": nc.normalized_entity or nc.raw_entity or "overall",
            "period": nc.normalized_period or nc.raw_period or "all",
            "claimed_value": g.claimed_value,
            "actual_value": r.actual_value,
            "signed_gap": g.signed_gap,
            "absolute_gap": g.absolute_gap,
            "relative_gap_pct": g.relative_gap_pct,
            "gap_type": g.gap_type.value,
            "unit": g.unit or nc.normalized_unit or "",
            "status": r.status.value,
            "original_machine_status": r.original_machine_status.value,
            "evidence_quality": r.evidence_quality.quality_level.value,
            "quality_score": r.evidence_quality.score,
            "evidence_reason": r.evidence_gate.reason_code.value,
            "evidence_detail": r.evidence_gate.reason_detail,
            "matching_records": r.evidence_gate.matching_records_count,
            "formula_used": r.formula_used,
            "applied_filters": str(r.applied_filters),
            "needs_human_review": r.needs_human_review,
            "review_reason": r.review_reason or "",
            "explanation": r.explanation,
        }

    for res in results:
        rows.append(_to_row(res, is_subclaim=False))
        if include_subclaims and res.subclaim_results:
            for sub in res.subclaim_results:
                rows.append(_to_row(sub, is_subclaim=True))
    return rows


def results_dataframe(results: list[ClaimVerificationResult], include_subclaims: bool = True) -> pd.DataFrame:
    rows = flatten_claim_rows(results, include_subclaims=include_subclaims)
    return pd.DataFrame(rows)
