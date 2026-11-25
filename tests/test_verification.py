"""Phase 8 tests: independent numerical verification and tolerance checks."""

from __future__ import annotations

import pytest

from backend.claim_mapper import ClaimMapper
from backend.claim_normalizer import ClaimNormalizer
from backend.claim_verifier import ClaimVerifier
from backend.data_loader import DataLoader
from backend.data_profiler import profile_dataset
from backend.preprocessing import preprocess_transactions
from backend.schemas import ExtractedClaim, GapType, RevenueMode, TransactionPolicy
from config.settings import PROJECT_ROOT


def test_independent_verification_recalculates_exact_values() -> None:
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
    verifier = ClaimVerifier()

    claims = [
        ExtractedClaim(
            claim_id="V1",
            claim_text="December revenue reached £14,800.00.",
            source_sentence="December revenue reached £14,800.00.",
            report_section="Monthly",
            metric="monthly_revenue",
            period="2024-12",
            claimed_value=14800.0,
            unit="£",
        ),
        ExtractedClaim(
            claim_id="V2",
            claim_text="Laptop Pro 15 ranked #1.",
            source_sentence="Laptop Pro 15 ranked #1.",
            report_section="Products",
            entity="Laptop Pro 15",
            metric="top_products",
            operation="ranking",
            claimed_value="Laptop Pro 15",
            unit="rank",
        ),
        ExtractedClaim(
            claim_id="V3",
            claim_text="December revenue was £14,200.00.",
            source_sentence="December revenue was £14,200.00.",
            report_section="Monthly",
            metric="monthly_revenue",
            period="2024-12",
            claimed_value=14200.0,  # Actually October's revenue -> PERIOD_MISMATCH
            unit="£",
        ),
    ]
    mapped = mapper.map_claims(normalizer.normalize_all(claims))

    g1, c1, gap1 = verifier.verify_atomic_claim(mapped[0], annotated, processed, profile, policy)
    assert g1.has_sufficient_evidence is True
    assert c1 is not None and c1.value == pytest.approx(14800.0)
    assert gap1.gap_type == GapType.EXACT_MATCH

    g2, c2, gap2 = verifier.verify_atomic_claim(mapped[1], annotated, processed, profile, policy)
    assert c2 is not None and c2.value == "Laptop Pro 15"
    assert gap2.gap_type == GapType.EXACT_MATCH

    g3, c3, gap3 = verifier.verify_atomic_claim(mapped[2], annotated, processed, profile, policy)
    assert c3 is not None and c3.value == pytest.approx(14800.0)
    assert gap3.gap_type == GapType.PERIOD_MISMATCH
