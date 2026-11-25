"""Shared Pydantic v2 schemas and enums across all VeriReport AI pipeline phases."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Core Enums
# ---------------------------------------------------------------------------


class RevenueMode(str, Enum):
    NET_REVENUE = "net_revenue"
    GROSS_SALES = "gross_sales"
    RETURNS_ONLY = "returns_only"
    CANCELLATIONS_ONLY = "cancellations_only"


class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"


class EvidenceQualityLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class GapType(str, Enum):
    EXACT_MATCH = "EXACT_MATCH"
    OVERSTATED = "OVERSTATED"
    UNDERSTATED = "UNDERSTATED"
    METRIC_MISMATCH = "METRIC_MISMATCH"
    PERIOD_MISMATCH = "PERIOD_MISMATCH"
    NOT_COMPARABLE = "NOT_COMPARABLE"


class EvidenceReason(str, Enum):
    SUFFICIENT_EVIDENCE = "SUFFICIENT_EVIDENCE"
    MISSING_REQUIRED_COLUMN = "MISSING_REQUIRED_COLUMN"
    UNSUPPORTED_METRIC = "UNSUPPORTED_METRIC"
    MISSING_REPORTING_PERIOD = "MISSING_REPORTING_PERIOD"
    MISSING_COMPARISON_PERIOD = "MISSING_COMPARISON_PERIOD"
    NO_MATCHING_RECORDS = "NO_MATCHING_RECORDS"
    AMBIGUOUS_ENTITY = "AMBIGUOUS_ENTITY"
    INSUFFICIENT_DATA_QUALITY = "INSUFFICIENT_DATA_QUALITY"
    INCOMPATIBLE_UNITS = "INCOMPATIBLE_UNITS"


# ---------------------------------------------------------------------------
# Phase 1: Dataset Ingestion Schemas
# ---------------------------------------------------------------------------


class ColumnMappingResult(BaseModel):
    """Result of mapping raw dataset columns to canonical retail fields."""

    mapping: dict[str, str] = Field(default_factory=dict)
    missing_required_columns: list[str] = Field(default_factory=list)
    unmapped_raw_columns: list[str] = Field(default_factory=list)
    is_valid: bool = True


class DatasetMetadata(BaseModel):
    """Metadata captured upon uploading/loading a retail transaction dataset."""

    dataset_id: str
    filename: str
    safe_filename: str
    file_format: str
    file_size_bytes: int
    dataset_hash: str
    raw_row_count: int
    raw_column_count: int
    raw_columns: list[str] = Field(default_factory=list)
    column_mapping: dict[str, str] = Field(default_factory=dict)
    loaded_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Phase 2: Preprocessing & Profiling Schemas
# ---------------------------------------------------------------------------


class TransactionPolicy(BaseModel):
    """Configurable accounting policy for retail transactions."""

    revenue_mode: RevenueMode = RevenueMode.NET_REVENUE
    include_returns: bool = True
    include_cancellations: bool = True
    deduplicate_exact_rows: bool = False
    drop_invalid_dates: bool = True
    drop_non_positive_unit_price: bool = True
    currency_symbol: str = "£"
    currency_code: str = "GBP"


class TransformationStep(BaseModel):
    step_name: str
    rows_before: int
    rows_after: int
    affected_rows: int
    reason: str


class PeriodCoverageInfo(BaseModel):
    period: str
    period_type: str  # "month" or "year"
    days_present: int
    expected_days: int
    coverage_ratio: float
    is_complete: bool
    first_date: str
    last_date: str


class DatasetProfile(BaseModel):
    """Comprehensive data quality and coverage profile."""

    total_raw_rows: int
    total_processed_rows: int
    total_columns: int
    dtypes: dict[str, str] = Field(default_factory=dict)
    missing_values: dict[str, int] = Field(default_factory=dict)
    missing_rates: dict[str, float] = Field(default_factory=dict)
    duplicate_rows: int = 0
    invalid_date_rows: int = 0
    invalid_numeric_rows: int = 0
    invalid_unit_price_rows: int = 0
    negative_quantity_rows: int = 0
    return_rows: int = 0
    cancellation_rows: int = 0
    date_min: str | None = None
    date_max: str | None = None
    distinct_products: int = 0
    distinct_countries: int = 0
    distinct_invoices: int = 0
    period_coverage: list[PeriodCoverageInfo] = Field(default_factory=list)
    partial_periods: list[str] = Field(default_factory=list)
    data_quality_score: float = 1.0
    quality_notes: list[str] = Field(default_factory=list)
    transformations: list[TransformationStep] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Phase 3: Analytics & Metric Registry Schemas
# ---------------------------------------------------------------------------


class MetricDefinition(BaseModel):
    metric_id: str
    metric_name: str
    description: str
    required_columns: list[str]
    formula: str
    supported_operations: list[str]
    applicable_filters: list[str]
    unit: str
    validation_rules: list[str]
    abs_tolerance: float = 1.0
    rel_tolerance: float = 0.005


class MetricComputationResult(BaseModel):
    metric_id: str
    operation: str
    value: float | str | list[Any] | None = None
    unit: str = ""
    matching_rows: int = 0
    formula_used: str = ""
    filters_applied: dict[str, Any] = Field(default_factory=dict)
    period_covered: bool = True
    is_partial_period: bool = False
    error_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Phase 4: Report Generation Schemas
# ---------------------------------------------------------------------------


class ReportSection(BaseModel):
    section_number: int
    title: str
    content: str
    sentences: list[str] = Field(default_factory=list)


class GeneratedReport(BaseModel):
    report_id: str
    dataset_id: str
    report_hash: str
    model_used: str
    generation_mode: str  # "ollama" or "deterministic_fallback"
    generated_at: str
    generation_time_seconds: float
    title: str
    currency_code: str
    currency_symbol: str
    sections: list[ReportSection] = Field(default_factory=list)
    full_text: str = ""
    txt_path: str | None = None
    docx_path: str | None = None
    pdf_path: str | None = None


# ---------------------------------------------------------------------------
# Phase 5 & 6: Claim Extraction, Validation, Normalization, Mapping Schemas
# ---------------------------------------------------------------------------


class ExtractedSubclaim(BaseModel):
    subclaim_id: str = ""
    claim_text: str
    entity: str | None = None
    metric: str | None = None
    period: str | None = None
    comparison_period: str | None = None
    operation: str = "value"
    claimed_value: float | str | None = None
    unit: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    extraction_confidence: float = 0.8
    ambiguity_flag: bool = False


class ExtractedClaim(BaseModel):
    claim_id: str
    claim_text: str
    source_sentence: str
    report_section: str
    entity: str | None = None
    metric: str | None = None
    period: str | None = None
    comparison_period: str | None = None
    operation: str = "value"
    claimed_value: float | str | None = None
    unit: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    extraction_confidence: float = 0.8
    ambiguity_flag: bool = False
    subclaims: list[ExtractedSubclaim] = Field(default_factory=list)


class NormalizedClaim(BaseModel):
    claim_id: str
    parent_claim_id: str | None = None
    claim_text: str
    source_sentence: str
    report_section: str
    raw_metric: str | None = None
    metric_id: str | None = None
    mapping_confidence: float = 0.0
    mapping_ambiguity: bool = False
    mapping_reason: str = ""
    raw_entity: str | None = None
    normalized_entity: str | None = None
    entity_type: str | None = None  # "overall", "product", "country"
    entity_ambiguity: bool = False
    raw_period: str | None = None
    normalized_period: str | None = None
    period_ambiguity: bool = False
    raw_comparison_period: str | None = None
    normalized_comparison_period: str | None = None
    operation: str = "value"
    claimed_value: float | str | None = None
    normalized_unit: str | None = None
    raw_unit: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    extraction_confidence: float = 0.8
    ambiguity_flag: bool = False
    subclaims: list["NormalizedClaim"] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Phase 7–11: Evidence, Verification, Gap, Quality, Status, Consistency, Review
# ---------------------------------------------------------------------------


class EvidenceGateResult(BaseModel):
    claim_id: str
    has_sufficient_evidence: bool
    reason_code: EvidenceReason
    reason_detail: str
    required_columns_present: bool = True
    metric_supported: bool = True
    entity_identifiable: bool = True
    period_covered: bool = True
    comparison_period_covered: bool = True
    is_partial_period: bool = False
    matching_records_count: int = 0
    units_compatible: bool = True
    data_quality_sufficient: bool = True
    needs_human_review: bool = False


class GapAnalysisResult(BaseModel):
    claim_id: str
    claimed_value: float | str | None = None
    actual_value: float | str | None = None
    signed_gap: float | None = None
    absolute_gap: float | None = None
    relative_gap_pct: float | None = None
    percentage_point_diff: float | None = None
    gap_type: GapType = GapType.NOT_COMPARABLE
    unit: str = ""
    tolerance: float | None = None
    comparison_method: str = "not_comparable"
    notes: str = ""


class EvidenceQualityResult(BaseModel):
    claim_id: str
    quality_level: EvidenceQualityLevel
    score: float
    dimension_findings: dict[str, str] = Field(default_factory=dict)
    explanation: str = ""


class ClaimVerificationResult(BaseModel):
    claim_id: str
    parent_claim_id: str | None = None
    status: VerificationStatus
    original_machine_status: VerificationStatus
    normalized_claim: NormalizedClaim
    evidence_gate: EvidenceGateResult
    actual_value: float | str | None = None
    formula_used: str = ""
    applied_filters: dict[str, Any] = Field(default_factory=dict)
    gap_analysis: GapAnalysisResult
    evidence_quality: EvidenceQualityResult
    explanation: str = ""
    explanation_source: str = "deterministic_template"
    needs_human_review: bool = False
    review_reason: str | None = None
    subclaim_results: list["ClaimVerificationResult"] = Field(default_factory=list)


class ConsistencyFinding(BaseModel):
    finding_id: str
    related_claim_ids: list[str]
    relationship_type: str
    expected_value: float | str | None = None
    observed_value: float | str | None = None
    difference: float | None = None
    is_consistent: bool
    explanation: str


class ReviewAction(BaseModel):
    review_id: str
    claim_id: str
    reviewer: str = "analyst"
    action_type: str  # "confirm", "reject", "remap", "clarify_period", "clarify_entity", "note"
    previous_status: VerificationStatus
    new_status: VerificationStatus
    corrected_metric_id: str | None = None
    corrected_period: str | None = None
    corrected_entity: str | None = None
    external_evidence_ref: str | None = None
    notes: str = ""
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
