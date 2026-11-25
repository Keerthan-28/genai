"""Phase 9 tests: deterministic five-status verification engine."""

from __future__ import annotations

from backend.claim_mapper import ClaimMapper
from backend.claim_normalizer import ClaimNormalizer
from backend.data_loader import DataLoader
from backend.data_profiler import profile_dataset
from backend.preprocessing import preprocess_transactions
from backend.schemas import (
    ExtractedClaim,
    ExtractedSubclaim,
    RevenueMode,
    TransactionPolicy,
    VerificationStatus,
)
from backend.status_engine import StatusEngine
from config.settings import PROJECT_ROOT


def _context():
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
    engine = StatusEngine()
    return annotated, processed, profile, policy, normalizer, mapper, engine


def test_all_five_verification_statuses_independently() -> None:
    annotated, processed, profile, policy, normalizer, mapper, engine = _context()

    claims = [
        # 1. VERIFIED
        ExtractedClaim(
            claim_id="S_VER",
            claim_text="December revenue reached £14,800.00.",
            source_sentence="December revenue reached £14,800.00.",
            report_section="Monthly",
            metric="monthly_revenue",
            period="2024-12",
            claimed_value=14800.0,
            unit="£",
        ),
        # 2. CONTRADICTED
        ExtractedClaim(
            claim_id="S_CON",
            claim_text="December revenue reached £25,000.00.",
            source_sentence="December revenue reached £25,000.00.",
            report_section="Monthly",
            metric="monthly_revenue",
            period="2024-12",
            claimed_value=25000.0,
            unit="£",
        ),
        # 3. INSUFFICIENT_EVIDENCE (unsupported metric)
        ExtractedClaim(
            claim_id="S_INS",
            claim_text="Customer satisfaction reached 94%.",
            source_sentence="Customer satisfaction reached 94%.",
            report_section="Insights",
            metric="customer_satisfaction",
            claimed_value=94.0,
            unit="%",
        ),
        # 4. NEEDS_REVIEW (ambiguous metric mapping)
        ExtractedClaim(
            claim_id="S_REV",
            claim_text="Composite score was 88.",
            source_sentence="Composite score was 88.",
            report_section="Insights",
            metric="unknown_metric",
            claimed_value=88.0,
        ),
        # 5. PARTIALLY_VERIFIED (compound: 1 true + 1 false)
        ExtractedClaim(
            claim_id="S_PAR",
            claim_text="December revenue was £14,800 and Laptop Pro 15 revenue in December was £99,000.",
            source_sentence="December revenue was £14,800 and Laptop Pro 15 revenue in December was £99,000.",
            report_section="Monthly",
            metric="compound_claim",
            operation="compound",
            subclaims=[
                ExtractedSubclaim(
                    subclaim_id="S_PAR.1",
                    claim_text="December revenue was £14,800.",
                    metric="monthly_revenue",
                    period="2024-12",
                    claimed_value=14800.0,
                    unit="£",
                ),
                ExtractedSubclaim(
                    subclaim_id="S_PAR.2",
                    claim_text="Laptop Pro 15 revenue in December was £99,000.",
                    entity="Laptop Pro 15",
                    metric="product_revenue",
                    period="2024-12",
                    claimed_value=99000.0,
                    unit="£",
                ),
            ],
        ),
    ]

    mapped = mapper.map_claims(normalizer.normalize_all(claims))
    results = engine.evaluate_all(mapped, annotated, processed, profile, policy)

    assert results[0].status == VerificationStatus.VERIFIED
    assert results[1].status == VerificationStatus.CONTRADICTED
    assert results[2].status == VerificationStatus.INSUFFICIENT_EVIDENCE
    assert results[3].status == VerificationStatus.NEEDS_REVIEW
    assert results[4].status == VerificationStatus.PARTIALLY_VERIFIED
    assert results[4].subclaim_results[0].status == VerificationStatus.VERIFIED
    assert results[4].subclaim_results[1].status == VerificationStatus.CONTRADICTED
