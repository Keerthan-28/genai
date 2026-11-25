"""Claim normalization for metrics, entities, periods, units, and operations."""

from __future__ import annotations

import re
from typing import Iterable

from backend.schemas import ExtractedClaim, ExtractedSubclaim, NormalizedClaim

MONTH_NAME_MAP: dict[str, str] = {
    "january": "01",
    "jan": "01",
    "february": "02",
    "feb": "02",
    "march": "03",
    "mar": "03",
    "april": "04",
    "apr": "04",
    "may": "05",
    "june": "06",
    "jun": "06",
    "july": "07",
    "jul": "07",
    "august": "08",
    "aug": "08",
    "september": "09",
    "sep": "09",
    "sept": "09",
    "october": "10",
    "oct": "10",
    "november": "11",
    "nov": "11",
    "december": "12",
    "dec": "12",
}

CURRENCY_UNIT_MAP: dict[str, str] = {
    "£": "GBP",
    "gbp": "GBP",
    "pound": "GBP",
    "pounds": "GBP",
    "$": "USD",
    "usd": "USD",
    "dollar": "USD",
    "dollars": "USD",
    "₹": "INR",
    "inr": "INR",
    "rs": "INR",
    "rupee": "INR",
    "rupees": "INR",
    "€": "EUR",
    "eur": "EUR",
    "euro": "EUR",
    "euros": "EUR",
}


class ClaimNormalizer:
    """Normalizes extracted claims against dataset vocabulary and canonical formats."""

    def __init__(
        self,
        known_products: Iterable[str] = (),
        known_countries: Iterable[str] = (),
        known_periods: Iterable[str] = (),
    ) -> None:
        self.known_products = {p.strip().lower(): p.strip() for p in known_products if p and str(p).strip()}
        self.known_countries = {c.strip().lower(): c.strip() for c in known_countries if c and str(c).strip()}
        self.known_periods = sorted({str(p).strip() for p in known_periods if p and str(p).strip()})
        self.known_years = sorted({p[:4] for p in self.known_periods if len(p) >= 4 and p[:4].isdigit()})

    def normalize_unit(self, raw_unit: str | None) -> str | None:
        if not raw_unit:
            return None
        cleaned = str(raw_unit).strip()
        low = cleaned.lower()
        if cleaned in CURRENCY_UNIT_MAP:
            return CURRENCY_UNIT_MAP[cleaned]
        if low in CURRENCY_UNIT_MAP:
            return CURRENCY_UNIT_MAP[low]
        if low in ("%", "percent", "percentage", "pct", "percentage points", "pp"):
            return "percent"
        if low in ("unit", "units", "qty", "quantity", "items"):
            return "units"
        if low in ("count", "orders", "transactions"):
            return "count"
        if low in ("rank", "ranking"):
            return "rank"
        return cleaned

    def normalize_period(self, raw_period: str | None) -> tuple[str | None, bool]:
        """Normalize period string to `YYYY-MM` or `YYYY`. Returns `(period, is_ambiguous)`."""
        if not raw_period:
            return None, False
        text = str(raw_period).strip()
        if re.fullmatch(r"20\d{2}-\d{2}", text):
            return text, False
        if re.fullmatch(r"20\d{2}", text):
            return text, False

        # e.g. "December 2024" or "Dec 2024"
        m = re.fullmatch(r"([A-Za-z]+)\s+(20\d{2})", text)
        if m:
            mm = MONTH_NAME_MAP.get(m.group(1).lower())
            if mm:
                return f"{m.group(2)}-{mm}", False

        # Month name only (e.g. "December") -> infer year only if dataset has a single year
        mm = MONTH_NAME_MAP.get(text.lower())
        if mm:
            if len(self.known_years) == 1:
                return f"{self.known_years[0]}-{mm}", False
            matching = [p for p in self.known_periods if p.endswith(f"-{mm}")]
            if len(matching) == 1:
                return matching[0], False
            # Multi-year ambiguity or no matching year in dataset
            return f"????-{mm}", len(self.known_years) > 1

        return text, True

    def normalize_entity(self, raw_entity: str | None) -> tuple[str | None, str | None, bool]:
        """Resolve an entity name to `(canonical_name, entity_type, is_ambiguous)`."""
        if not raw_entity:
            return None, "overall", False
        cleaned = str(raw_entity).strip()
        low = cleaned.lower()

        if low in ("overall", "all", "total", "company", "store", "dataset"):
            return None, "overall", False

        # Exact match first
        if low in self.known_countries:
            return self.known_countries[low], "country", False
        if low in self.known_products:
            return self.known_products[low], "product", False

        # Partial/substring match
        country_hits = [canon for key, canon in self.known_countries.items() if low in key or key in low]
        if len(country_hits) == 1:
            return country_hits[0], "country", False
        if len(country_hits) > 1:
            return cleaned, "country", True

        product_hits = [canon for key, canon in self.known_products.items() if low in key or key in low]
        if len(product_hits) == 1:
            return product_hits[0], "product", False
        if len(product_hits) > 1:
            return cleaned, "product", True

        # Unknown entity (may be missing from dataset or ambiguous)
        return cleaned, None, False

    @staticmethod
    def normalize_operation(operation: str | None, unit: str | None, raw_metric: str | None) -> str:
        op = (operation or "value").strip().lower()
        metric_low = (raw_metric or "").lower()
        if op in ("growth", "growth_pct", "pct_change", "mom_growth"):
            return "growth_pct"
        if op in ("share", "share_pct", "contribution", "pct_share"):
            return "share_pct"
        if op in ("rank", "ranking", "top"):
            return "ranking"
        if op in ("count", "num", "frequency"):
            return "count"
        if op in ("compound",):
            return "compound"
        if "growth" in metric_low:
            return "growth_pct"
        if "contribution" in metric_low or "share" in metric_low:
            return "share_pct"
        if unit == "percent" and op == "value":
            return "share_pct"
        return "value"

    def normalize_claim_obj(
        self,
        claim: ExtractedClaim | ExtractedSubclaim,
        report_section: str = "",
        source_sentence: str = "",
        parent_claim_id: str | None = None,
    ) -> NormalizedClaim:
        if isinstance(claim, ExtractedClaim):
            claim_id = claim.claim_id
            section = claim.report_section
            sentence = claim.source_sentence
            raw_subs = claim.subclaims
        else:
            claim_id = claim.subclaim_id
            section = report_section
            sentence = source_sentence
            raw_subs = []

        norm_unit = self.normalize_unit(claim.unit)
        norm_period, period_ambig = self.normalize_period(claim.period)
        norm_comp, comp_ambig = self.normalize_period(claim.comparison_period)
        norm_entity, entity_type, entity_ambig = self.normalize_entity(claim.entity)
        norm_op = self.normalize_operation(claim.operation, norm_unit, claim.metric)

        sub_norms = [
            self.normalize_claim_obj(
                sub,
                report_section=section,
                source_sentence=sentence,
                parent_claim_id=claim_id,
            )
            for sub in raw_subs
        ]

        return NormalizedClaim(
            claim_id=claim_id,
            parent_claim_id=parent_claim_id,
            claim_text=claim.claim_text,
            source_sentence=sentence,
            report_section=section,
            raw_metric=claim.metric,
            raw_entity=claim.entity,
            normalized_entity=norm_entity,
            entity_type=entity_type,
            entity_ambiguity=entity_ambig,
            raw_period=claim.period,
            normalized_period=norm_period,
            period_ambiguity=period_ambig or comp_ambig,
            raw_comparison_period=claim.comparison_period,
            normalized_comparison_period=norm_comp,
            operation=norm_op,
            claimed_value=claim.claimed_value,
            normalized_unit=norm_unit,
            raw_unit=claim.unit,
            filters=dict(claim.filters),
            extraction_confidence=claim.extraction_confidence,
            ambiguity_flag=bool(claim.ambiguity_flag or period_ambig or comp_ambig or entity_ambig),
            subclaims=sub_norms,
        )

    def normalize_all(self, claims: list[ExtractedClaim]) -> list[NormalizedClaim]:
        return [self.normalize_claim_obj(c) for c in claims]
