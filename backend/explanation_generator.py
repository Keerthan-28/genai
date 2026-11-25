"""Phase 11: Explanation generator grounded strictly in verification engine outputs.

The LLM may translate structured verification findings into plain-English prose
but must NEVER alter statuses, recalculate numbers, or invent evidence.
"""

from __future__ import annotations

import json

from backend.llm_client import OllamaClientWrapper
from backend.schemas import ClaimVerificationResult, VerificationStatus
from config.settings import Settings, get_settings


class ExplanationGenerator:
    """Generates plain-language explanations with deterministic Python fallback."""

    def __init__(
        self,
        llm_client: OllamaClientWrapper | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm_client = llm_client or OllamaClientWrapper(self.settings)

    @staticmethod
    def build_deterministic_explanation(result: ClaimVerificationResult) -> str:
        c = result.normalized_claim
        g = result.gap_analysis
        gate = result.evidence_gate
        q = result.evidence_quality

        if result.subclaim_results:
            counts: dict[str, int] = {}
            for sub in result.subclaim_results:
                counts[sub.status.value] = counts.get(sub.status.value, 0) + 1
            breakdown = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
            return (
                f"Compound claim status is {result.status.value} across {len(result.subclaim_results)} subclaims "
                f"({breakdown}). Evidence quality: {q.quality_level.value}."
            )

        if result.status == VerificationStatus.VERIFIED:
            return (
                f"VERIFIED: Claimed value ({g.claimed_value}) matches independent calculation "
                f"({result.actual_value}) for metric '{c.metric_id}' within tolerance ({g.tolerance}). "
                f"Evidence quality is {q.quality_level.value} ({gate.matching_records_count} matching records)."
            )
        if result.status == VerificationStatus.CONTRADICTED:
            rel_txt = f" ({g.relative_gap_pct:+.2f}%)" if g.relative_gap_pct is not None else ""
            gap_txt = f"{g.signed_gap:+.2f}{rel_txt}" if g.signed_gap is not None else g.notes
            return (
                f"CONTRADICTED ({g.gap_type.value}): Report claimed {g.claimed_value}, whereas independent "
                f"Python analysis calculated {result.actual_value} (gap: {gap_txt}). "
                f"Evidence quality is {q.quality_level.value}."
            )
        if result.status == VerificationStatus.INSUFFICIENT_EVIDENCE:
            return (
                f"INSUFFICIENT_EVIDENCE ({gate.reason_code.value}): {gate.reason_detail} "
                f"Lack of evidence is not proof of falsity. Evidence quality: {q.quality_level.value}."
            )
        if result.status == VerificationStatus.NEEDS_REVIEW:
            return (
                f"NEEDS_REVIEW: {result.review_reason or gate.reason_detail} "
                f"Human verification is required before final sign-off."
            )
        return f"Status {result.status.value}: {gate.reason_detail}"

    def explain_result(
        self,
        result: ClaimVerificationResult,
        use_llm: bool = False,
    ) -> ClaimVerificationResult:
        """Populate explanations recursively without modifying statuses or numbers."""
        updated = result.model_copy(deep=True)
        if updated.subclaim_results:
            updated.subclaim_results = [
                self.explain_result(sub, use_llm=use_llm) for sub in updated.subclaim_results
            ]

        deterministic_text = self.build_deterministic_explanation(updated)
        if not use_llm:
            updated.explanation = deterministic_text
            updated.explanation_source = "deterministic_template"
            return updated

        payload = {
            "claim_id": updated.claim_id,
            "claim_text": updated.normalized_claim.claim_text,
            "status": updated.status.value,
            "metric_id": updated.normalized_claim.metric_id,
            "period": updated.normalized_claim.normalized_period,
            "claimed_value": updated.gap_analysis.claimed_value,
            "actual_value": updated.actual_value,
            "signed_gap": updated.gap_analysis.signed_gap,
            "relative_gap_pct": updated.gap_analysis.relative_gap_pct,
            "gap_type": updated.gap_analysis.gap_type.value,
            "evidence_reason": updated.evidence_gate.reason_code.value,
            "evidence_detail": updated.evidence_gate.reason_detail,
            "evidence_quality": updated.evidence_quality.quality_level.value,
            "review_reason": updated.review_reason,
        }
        try:
            text = self.llm_client.generate_text(
                prompt=(
                    "Write a concise 2-sentence audit explanation based strictly on this JSON. "
                    "Do NOT alter the status, numbers, or invent new evidence.\n"
                    f"{json.dumps(payload, indent=2, default=str)}"
                ),
                system_instruction=(
                    "You are an audit explainer. Summarize only the structured verification facts given. "
                    "Never change status or numbers."
                ),
                temperature=0.1,
            ).strip()
            if text:
                updated.explanation = text
                updated.explanation_source = "ollama"
                return updated
        except Exception:
            pass

        updated.explanation = deterministic_text
        updated.explanation_source = "deterministic_template"
        return updated

    def explain_all(
        self,
        results: list[ClaimVerificationResult],
        use_llm: bool = False,
    ) -> list[ClaimVerificationResult]:
        return [self.explain_result(r, use_llm=use_llm) for r in results]
