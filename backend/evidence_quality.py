"""Phase 9: Independent 7-dimension evidence quality assessment."""

from __future__ import annotations

from backend.schemas import (
    DatasetProfile,
    EvidenceGateResult,
    EvidenceQualityLevel,
    EvidenceQualityResult,
    MetricComputationResult,
    NormalizedClaim,
)


class EvidenceQualityAssessor:
    """Assesses evidence strength independently of whether a claim is true or false."""

    def assess(
        self,
        claim: NormalizedClaim,
        gate: EvidenceGateResult,
        calc: MetricComputationResult | None,
        profile: DatasetProfile,
    ) -> EvidenceQualityResult:
        score = 0.0
        findings: dict[str, str] = {}

        # 1. Data completeness
        req_missing = [
            rate for col, rate in profile.missing_rates.items() if col != "CustomerID"
        ]
        max_missing = max(req_missing, default=0.0)
        if max_missing <= 0.02:
            findings["data_completeness"] = "Complete required fields (<=2% missing)."
            score += 1.0
        elif max_missing <= 0.15:
            findings["data_completeness"] = f"Moderate missingness ({max_missing:.1%})."
            score += 0.6
        else:
            findings["data_completeness"] = f"High missingness ({max_missing:.1%})."
            score += 0.2

        # 2. Data quality
        if profile.data_quality_score >= 0.85:
            findings["data_quality"] = f"High quality score ({profile.data_quality_score:.2f})."
            score += 1.0
        elif profile.data_quality_score >= 0.60:
            findings["data_quality"] = f"Moderate quality score ({profile.data_quality_score:.2f})."
            score += 0.6
        else:
            findings["data_quality"] = f"Low quality score ({profile.data_quality_score:.2f})."
            score += 0.2

        # 3. Reporting period coverage
        if not gate.period_covered or not gate.comparison_period_covered:
            findings["reporting_period_coverage"] = "Target or comparison period not covered."
            score += 0.0
        elif gate.is_partial_period:
            findings["reporting_period_coverage"] = "Partial period coverage (incomplete month/year)."
            score += 0.55
        else:
            findings["reporting_period_coverage"] = "Full period coverage available."
            score += 1.0

        # 4. Record sufficiency
        if gate.matching_records_count >= 5:
            findings["record_sufficiency"] = f"{gate.matching_records_count} matching records."
            score += 1.0
        elif gate.matching_records_count >= 1:
            findings["record_sufficiency"] = f"Limited sample ({gate.matching_records_count} matching record(s))."
            score += 0.7
        else:
            findings["record_sufficiency"] = "No matching records."
            score += 0.0

        # 5. Metric mapping reliability
        if claim.metric_id and not claim.mapping_ambiguity and claim.mapping_confidence >= 0.85:
            findings["metric_mapping_reliability"] = f"High-confidence mapping to '{claim.metric_id}'."
            score += 1.0
        elif claim.metric_id:
            findings["metric_mapping_reliability"] = f"Moderate-confidence mapping ({claim.mapping_confidence:.2f})."
            score += 0.6
        else:
            findings["metric_mapping_reliability"] = "Unmapped or unsupported metric."
            score += 0.1

        # 6. Source traceability
        if claim.source_sentence and claim.report_section:
            findings["source_traceability"] = f"Traceable to section '{claim.report_section}'."
            score += 1.0
        else:
            findings["source_traceability"] = "Partial report traceability."
            score += 0.5

        # 7. Calculation reproducibility
        if gate.has_sufficient_evidence and calc is not None and calc.value is not None and not calc.error_reason:
            findings["calculation_reproducibility"] = "Deterministic Python recalculation succeeded."
            score += 1.0
        else:
            findings["calculation_reproducibility"] = "Deterministic calculation could not complete."
            score += 0.0

        normalized_score = round(score / 7.0, 4)

        if not gate.has_sufficient_evidence or normalized_score < 0.50:
            level = EvidenceQualityLevel.LOW
        elif gate.is_partial_period or normalized_score < 0.80:
            level = EvidenceQualityLevel.MEDIUM
        else:
            level = EvidenceQualityLevel.HIGH

        explanation = (
            f"Evidence quality is {level.value} (score {normalized_score:.2f}/1.00). "
            f"Note: Evidence quality reflects data completeness and reproducibility, not claim truth."
        )
        return EvidenceQualityResult(
            claim_id=claim.claim_id,
            quality_level=level,
            score=normalized_score,
            dimension_findings=findings,
            explanation=explanation,
        )
