"""Maps normalized claims to predefined MetricRegistry entries deterministically."""

from __future__ import annotations

from backend.metric_registry import MetricRegistry
from backend.schemas import NormalizedClaim

METRIC_SYNONYMS: dict[str, str] = {
    "total_revenue": "total_revenue",
    "revenue": "total_revenue",
    "sales": "total_revenue",
    "total_sales": "total_revenue",
    "gross_revenue": "gross_revenue",
    "gross_sales": "gross_revenue",
    "net_revenue": "net_revenue",
    "net_sales": "net_revenue",
    "returns": "returns_amount",
    "returns_amount": "returns_amount",
    "return_amount": "returns_amount",
    "monthly_revenue": "monthly_revenue",
    "month_revenue": "monthly_revenue",
    "revenue_growth": "revenue_growth",
    "sales_growth": "revenue_growth",
    "mom_growth": "revenue_growth",
    "growth": "revenue_growth",
    "quantity": "quantity_sold",
    "quantity_sold": "quantity_sold",
    "units_sold": "quantity_sold",
    "product_revenue": "product_revenue",
    "product_sales": "product_revenue",
    "top_products": "top_products",
    "top_product": "top_products",
    "product_ranking": "top_products",
    "country_revenue": "country_revenue",
    "country_sales": "country_revenue",
    "average_order_value": "average_order_value",
    "aov": "average_order_value",
    "revenue_contribution": "revenue_contribution",
    "contribution": "revenue_contribution",
    "revenue_share": "revenue_contribution",
    "cancellation_analysis": "cancellation_analysis",
    "cancellations": "cancellation_analysis",
    "cancellation_amount": "cancellation_analysis",
    "cancellation_rate": "cancellation_analysis",
    "order_count": "order_count",
    "orders": "order_count",
    "distinct_orders": "order_count",
}


class ClaimMapper:
    """Deterministically maps normalized claims to MetricRegistry IDs."""

    def __init__(self, registry: MetricRegistry | None = None) -> None:
        self.registry = registry or MetricRegistry()

    def _map_single(self, claim: NormalizedClaim) -> NormalizedClaim:
        if claim.subclaims:
            claim.subclaims = [self._map_single(sub) for sub in claim.subclaims]
            claim.metric_id = "compound_claim"
            claim.mapping_confidence = min(
                (sub.mapping_confidence for sub in claim.subclaims),
                default=0.0,
            )
            claim.mapping_ambiguity = any(sub.mapping_ambiguity for sub in claim.subclaims)
            claim.mapping_reason = "Compound claim evaluated via subclaim mappings."
            return claim

        raw_m = (claim.raw_metric or "").strip().lower()
        text_low = claim.claim_text.lower()

        # Explicit unsupported metrics check (profit, ROI, CSAT, inventory, etc.)
        if self.registry.is_explicitly_unsupported_phrase(raw_m) or self.registry.is_explicitly_unsupported_phrase(text_low):
            claim.metric_id = None
            claim.mapping_confidence = 0.0
            claim.mapping_ambiguity = False
            claim.mapping_reason = f"Unsupported retail metric referenced ('{claim.raw_metric or claim.claim_text}')."
            return claim

        # Ambiguous/unknown extraction
        if raw_m in ("", "unknown", "unknown_metric"):
            claim.metric_id = None
            claim.mapping_confidence = 0.2
            claim.mapping_ambiguity = True
            claim.mapping_reason = "Claim metric could not be confidently mapped and requires human review."
            return claim

        mapped_id = METRIC_SYNONYMS.get(raw_m)

        # Contextual disambiguation using operation & entity_type
        if mapped_id == "total_revenue":
            if claim.operation == "growth_pct":
                mapped_id = "revenue_growth"
            elif claim.operation == "share_pct":
                mapped_id = "revenue_contribution"
            elif claim.entity_type == "product":
                mapped_id = "product_revenue"
            elif claim.entity_type == "country":
                mapped_id = "country_revenue"
            elif claim.normalized_period and len(claim.normalized_period) == 7:
                mapped_id = "monthly_revenue"
        elif mapped_id == "monthly_revenue" and claim.entity_type == "product":
            mapped_id = "product_revenue"
        elif mapped_id == "monthly_revenue" and claim.entity_type == "country":
            mapped_id = "country_revenue"
        elif mapped_id in ("product_revenue", "country_revenue") and claim.operation == "share_pct":
            mapped_id = "revenue_contribution"

        if mapped_id and self.registry.is_supported(mapped_id):
            claim.metric_id = mapped_id
            claim.mapping_confidence = 0.95 if not claim.ambiguity_flag else 0.70
            claim.mapping_ambiguity = False
            claim.mapping_reason = f"Mapped '{claim.raw_metric}' to metric '{mapped_id}'."
        else:
            claim.metric_id = None
            claim.mapping_confidence = 0.25
            claim.mapping_ambiguity = True
            claim.mapping_reason = f"No confident MetricRegistry match for '{claim.raw_metric}'."

        return claim

    def map_claims(self, claims: list[NormalizedClaim]) -> list[NormalizedClaim]:
        return [self._map_single(c) for c in claims]
