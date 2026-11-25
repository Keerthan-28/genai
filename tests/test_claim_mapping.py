"""Phase 6 tests: claim normalization and deterministic metric mapping."""

from __future__ import annotations

from backend.claim_mapper import ClaimMapper
from backend.claim_normalizer import ClaimNormalizer
from backend.schemas import ExtractedClaim, ExtractedSubclaim


def test_normalization_and_mapping_of_compound_claim() -> None:
    normalizer = ClaimNormalizer(
        known_products=["Laptop Pro 15", "Wireless Mouse"],
        known_countries=["United Kingdom", "Germany"],
        known_periods=["2024-10", "2024-11", "2024-12"],
    )
    mapper = ClaimMapper()

    raw = ExtractedClaim(
        claim_id="CLM-001",
        claim_text="December revenue reached £14,800, while Laptop Pro 15 revenue was £12,000 and growth was 9.63% compared with November.",
        source_sentence="December revenue reached £14,800, while Laptop Pro 15 revenue was £12,000 and growth was 9.63% compared with November.",
        report_section="Monthly Sales Analysis",
        metric="compound_claim",
        operation="compound",
        subclaims=[
            ExtractedSubclaim(
                subclaim_id="CLM-001.1",
                claim_text="December revenue reached £14,800.",
                metric="revenue",
                period="December",
                claimed_value=14800.0,
                unit="£",
            ),
            ExtractedSubclaim(
                subclaim_id="CLM-001.2",
                claim_text="Laptop Pro 15 revenue was £12,000.",
                entity="Laptop Pro 15",
                metric="sales",
                period="December",
                claimed_value=12000.0,
                unit="GBP",
            ),
            ExtractedSubclaim(
                subclaim_id="CLM-001.3",
                claim_text="Growth was 9.63% compared with November.",
                metric="growth",
                period="December",
                comparison_period="November",
                operation="growth",
                claimed_value=9.63,
                unit="%",
            ),
        ],
    )

    norm = normalizer.normalize_all([raw])
    mapped = mapper.map_claims(norm)

    subs = mapped[0].subclaims
    assert subs[0].normalized_period == "2024-12"
    assert subs[0].normalized_unit == "GBP"
    assert subs[0].metric_id == "monthly_revenue"

    assert subs[1].normalized_entity == "Laptop Pro 15"
    assert subs[1].entity_type == "product"
    assert subs[1].metric_id == "product_revenue"

    assert subs[2].normalized_period == "2024-12"
    assert subs[2].normalized_comparison_period == "2024-11"
    assert subs[2].metric_id == "revenue_growth"


def test_unsupported_and_ambiguous_claims_handled_cleanly() -> None:
    normalizer = ClaimNormalizer(known_periods=["2023-12", "2024-12"])
    mapper = ClaimMapper()

    unsupported = ExtractedClaim(
        claim_id="CLM-002",
        claim_text="Customer satisfaction reached 95%.",
        source_sentence="Customer satisfaction reached 95%.",
        report_section="Insights",
        metric="customer_satisfaction",
        claimed_value=95.0,
        unit="%",
    )
    ambiguous = ExtractedClaim(
        claim_id="CLM-003",
        claim_text="December performance index hit 42.",
        source_sentence="December performance index hit 42.",
        report_section="Insights",
        metric="unknown_metric",
        period="December",
        claimed_value=42.0,
    )

    mapped = mapper.map_claims(normalizer.normalize_all([unsupported, ambiguous]))
    assert mapped[0].metric_id is None
    assert mapped[0].mapping_ambiguity is False  # explicitly unsupported, not ambiguous

    assert mapped[1].metric_id is None
    assert mapped[1].mapping_ambiguity is True
    assert mapped[1].period_ambiguity is True  # "December" ambiguous across 2023 & 2024
