"""Phase 7 tests: evidence sufficiency gate."""

from __future__ import annotations

from backend.claim_mapper import ClaimMapper
from backend.claim_normalizer import ClaimNormalizer
from backend.data_loader import DataLoader
from backend.data_profiler import profile_dataset
from backend.evidence_gate import EvidenceGate
from backend.preprocessing import preprocess_transactions
from backend.schemas import EvidenceReason, ExtractedClaim, RevenueMode, TransactionPolicy
from config.settings import PROJECT_ROOT


def _setup_context():
    sample_path = PROJECT_ROOT / "data" / "samples" / "sample_retail_transactions.csv"
    raw_df, _, _ = DataLoader().load_from_path(sample_path, persist_raw_copy=False)
    policy = TransactionPolicy(revenue_mode=RevenueMode.NET_REVENUE, deduplicate_exact_rows=True)
    annotated, processed, steps = preprocess_transactions(raw_df, policy)
    profile = profile_dataset(annotated, processed, steps)
    normalizer = ClaimNormalizer(
        known_products=processed["Description"].dropna().unique(),
        known_countries=processed["Country"].dropna().unique(),
        known_periods=processed["PeriodMonth"].dropna().unique(),
    )
    mapper = ClaimMapper()
    return annotated, processed, profile, policy, normalizer, mapper


def test_evidence_gate_accepts_supported_claim_and_flags_partial_period() -> None:
    annotated, processed, profile, policy, normalizer, mapper = _setup_context()
    gate = EvidenceGate()

    raw = ExtractedClaim(
        claim_id="C1",
        claim_text="December revenue reached £14,800.",
        source_sentence="December revenue reached £14,800.",
        report_section="Monthly Sales",
        metric="monthly_revenue",
        period="2024-12",
        claimed_value=14800.0,
        unit="£",
    )
    claim = mapper.map_claims(normalizer.normalize_all([raw]))[0]
    res = gate.evaluate(claim, annotated, processed, profile, policy)
    assert res.has_sufficient_evidence is True
    assert res.reason_code == EvidenceReason.SUFFICIENT_EVIDENCE
    assert res.is_partial_period is True


def test_evidence_gate_rejects_unsupported_missing_period_and_incompatible_units() -> None:
    annotated, processed, profile, policy, normalizer, mapper = _setup_context()
    gate = EvidenceGate()

    cases = [
        ExtractedClaim(
            claim_id="C_UNSUP",
            claim_text="Profit margin reached 25%.",
            source_sentence="Profit margin reached 25%.",
            report_section="Summary",
            metric="profit_margin",
            claimed_value=25.0,
            unit="%",
        ),
        ExtractedClaim(
            claim_id="C_PERIOD",
            claim_text="January 2025 revenue was £5,000.",
            source_sentence="January 2025 revenue was £5,000.",
            report_section="Monthly",
            metric="monthly_revenue",
            period="2025-01",
            claimed_value=5000.0,
            unit="£",
        ),
        ExtractedClaim(
            claim_id="C_UNIT",
            claim_text="December revenue was $14,800.",
            source_sentence="December revenue was $14,800.",
            report_section="Monthly",
            metric="monthly_revenue",
            period="2024-12",
            claimed_value=14800.0,
            unit="$",
        ),
    ]
    mapped = mapper.map_claims(normalizer.normalize_all(cases))
    r_unsup = gate.evaluate(mapped[0], annotated, processed, profile, policy)
    r_period = gate.evaluate(mapped[1], annotated, processed, profile, policy)
    r_unit = gate.evaluate(mapped[2], annotated, processed, profile, policy)

    assert r_unsup.reason_code == EvidenceReason.UNSUPPORTED_METRIC
    assert r_period.reason_code == EvidenceReason.MISSING_REPORTING_PERIOD
    assert r_unit.reason_code == EvidenceReason.INCOMPATIBLE_UNITS
