"""Phase 10: Cross-claim consistency checker.

Detects logical/arithmetic tensions across claims without assuming which
individual claim is false solely from cross-claim inconsistency.
"""

from __future__ import annotations

from collections import defaultdict

from backend.schemas import ClaimVerificationResult, ConsistencyFinding


def _flatten_results(results: list[ClaimVerificationResult]) -> list[ClaimVerificationResult]:
    flat: list[ClaimVerificationResult] = []
    for r in results:
        if r.subclaim_results:
            flat.extend(r.subclaim_results)
        else:
            flat.append(r)
    return flat


class ConsistencyChecker:
    """Checks cross-claim arithmetic and ranking relationships across extracted claims."""

    def check(self, results: list[ClaimVerificationResult]) -> list[ConsistencyFinding]:
        flat = _flatten_results(results)
        findings: list[ConsistencyFinding] = []
        counter = 1

        def next_id() -> str:
            nonlocal counter
            fid = f"CON-{counter:03d}"
            counter += 1
            return fid

        # Index numerical claims by (metric_id, period, entity_type)
        overall_rev_claims: dict[str | None, list[ClaimVerificationResult]] = defaultdict(list)
        monthly_rev_claims: dict[str, ClaimVerificationResult] = {}
        product_rev_claims: dict[str | None, list[ClaimVerificationResult]] = defaultdict(list)
        contrib_claims: dict[tuple[str | None, str | None], list[ClaimVerificationResult]] = defaultdict(list)
        growth_claims: list[ClaimVerificationResult] = []
        ranking_claims: list[ClaimVerificationResult] = []

        for r in flat:
            c = r.normalized_claim
            mid = c.metric_id
            val = c.claimed_value
            if mid in ("total_revenue", "net_revenue") and c.entity_type in (None, "overall") and isinstance(val, (int, float)):
                overall_rev_claims[c.normalized_period].append(r)
            if mid == "monthly_revenue" and c.normalized_period and isinstance(val, (int, float)):
                monthly_rev_claims[c.normalized_period] = r
            if mid == "product_revenue" and c.normalized_entity and isinstance(val, (int, float)):
                product_rev_claims[c.normalized_period].append(r)
            if mid == "revenue_contribution" and isinstance(val, (int, float)):
                contrib_claims[(c.entity_type, c.normalized_period)].append(r)
            if mid == "revenue_growth" and isinstance(val, (int, float)):
                growth_claims.append(r)
            if mid == "top_products":
                ranking_claims.append(r)

        # 1. Product revenue totals against overall revenue (same period)
        for period, prod_list in product_rev_claims.items():
            if len(prod_list) >= 2 and period in overall_rev_claims:
                total_claim = overall_rev_claims[period][0]
                total_val = float(total_claim.normalized_claim.claimed_value)  # type: ignore[arg-type]
                prod_sum = sum(float(p.normalized_claim.claimed_value) for p in prod_list)  # type: ignore[arg-type]
                is_ok = prod_sum <= total_val + 1.0
                findings.append(
                    ConsistencyFinding(
                        finding_id=next_id(),
                        related_claim_ids=[total_claim.claim_id] + [p.claim_id for p in prod_list],
                        relationship_type="product_sum_vs_total_revenue" if is_ok else "product_sum_exceeds_total_revenue",
                        expected_value=total_val,
                        observed_value=round(prod_sum, 4),
                        difference=round(prod_sum - total_val, 4),
                        is_consistent=is_ok,
                        explanation=(
                            f"Sum of claimed product revenues ({prod_sum:,.2f}) "
                            + ("does not exceed" if is_ok else "exceeds")
                            + f" claimed total revenue ({total_val:,.2f}) for period '{period or 'overall'}'."
                        ),
                    )
                )

        # 2. Contribution percentages sum check
        for (etype, period), clist in contrib_claims.items():
            if len(clist) >= 2:
                pct_sum = sum(float(x.normalized_claim.claimed_value) for x in clist)  # type: ignore[arg-type]
                is_ok = pct_sum <= 100.5
                findings.append(
                    ConsistencyFinding(
                        finding_id=next_id(),
                        related_claim_ids=[x.claim_id for x in clist],
                        relationship_type="contribution_percentage_sum" if is_ok else "contribution_percentage_exceeds_100",
                        expected_value=100.0,
                        observed_value=round(pct_sum, 4),
                        difference=round(pct_sum - 100.0, 4),
                        is_consistent=is_ok,
                        explanation=(
                            f"Sum of claimed {etype or 'entity'} contribution percentages ({pct_sum:.2f}%) "
                            + ("is within 100%" if is_ok else "exceeds 100%")
                            + f" for period '{period or 'overall'}'."
                        ),
                    )
                )

        # 3. Monthly growth consistency against claimed monthly revenues
        for g in growth_claims:
            gc = g.normalized_claim
            p_curr = gc.normalized_period
            p_prev = gc.normalized_comparison_period
            if p_curr in monthly_rev_claims and p_prev in monthly_rev_claims:
                r_curr = float(monthly_rev_claims[p_curr].normalized_claim.claimed_value)  # type: ignore[arg-type]
                r_prev = float(monthly_rev_claims[p_prev].normalized_claim.claimed_value)  # type: ignore[arg-type]
                if abs(r_prev) > 1e-12:
                    implied_growth = ((r_curr - r_prev) / abs(r_prev)) * 100.0
                    claimed_growth = float(gc.claimed_value)  # type: ignore[arg-type]
                    diff = implied_growth - claimed_growth
                    is_ok = abs(diff) <= 0.5
                    findings.append(
                        ConsistencyFinding(
                            finding_id=next_id(),
                            related_claim_ids=[
                                g.claim_id,
                                monthly_rev_claims[p_curr].claim_id,
                                monthly_rev_claims[p_prev].claim_id,
                            ],
                            relationship_type="monthly_growth_vs_monthly_revenues",
                            expected_value=round(implied_growth, 4),
                            observed_value=round(claimed_growth, 4),
                            difference=round(diff, 4),
                            is_consistent=is_ok,
                            explanation=(
                                f"Claimed monthly revenues ({p_prev}: {r_prev:,.2f} -> {p_curr}: {r_curr:,.2f}) "
                                f"imply {implied_growth:.2f}% growth vs claimed {claimed_growth:.2f}%."
                            ),
                        )
                    )

        # 4. Product ranking vs claimed product revenues
        for rk in ranking_claims:
            claimed_top = str(rk.normalized_claim.claimed_value or "").strip().lower()
            period = rk.normalized_claim.normalized_period
            candidates = product_rev_claims.get(period, []) or product_rev_claims.get(None, [])
            if claimed_top and candidates:
                by_name = {
                    str(p.normalized_claim.normalized_entity).strip().lower(): (
                        float(p.normalized_claim.claimed_value),  # type: ignore[arg-type]
                        p.claim_id,
                    )
                    for p in candidates
                    if p.normalized_claim.normalized_entity
                }
                if claimed_top in by_name:
                    top_val, top_cid = by_name[claimed_top]
                    higher = [(name, val, cid) for name, (val, cid) in by_name.items() if name != claimed_top and val > top_val + 1.0]
                    if higher:
                        best_other = max(higher, key=lambda x: x[1])
                        findings.append(
                            ConsistencyFinding(
                                finding_id=next_id(),
                                related_claim_ids=[rk.claim_id, top_cid, best_other[2]],
                                relationship_type="ranking_vs_claimed_product_revenue",
                                expected_value=top_val,
                                observed_value=best_other[1],
                                difference=round(best_other[1] - top_val, 4),
                                is_consistent=False,
                                explanation=(
                                    f"Ranking claim names '{claimed_top}' (#1, claimed {top_val:,.2f}), "
                                    f"but '{best_other[0]}' has higher claimed revenue ({best_other[1]:,.2f})."
                                ),
                            )
                        )
                    else:
                        findings.append(
                            ConsistencyFinding(
                                finding_id=next_id(),
                                related_claim_ids=[rk.claim_id, top_cid],
                                relationship_type="ranking_vs_claimed_product_revenue",
                                expected_value=top_val,
                                observed_value=top_val,
                                difference=0.0,
                                is_consistent=True,
                                explanation=(
                                    f"Ranking claim '#1 {claimed_top}' is consistent with claimed product revenues."
                                ),
                            )
                        )

        # 5. Subtotals (monthly revenues) vs grand total revenue
        if len(monthly_rev_claims) >= 2 and None in overall_rev_claims:
            grand_claim = overall_rev_claims[None][0]
            grand_val = float(grand_claim.normalized_claim.claimed_value)  # type: ignore[arg-type]
            monthly_sum = sum(float(m.normalized_claim.claimed_value) for m in monthly_rev_claims.values())  # type: ignore[arg-type]
            diff = monthly_sum - grand_val
            is_ok = abs(diff) <= 1.0
            findings.append(
                ConsistencyFinding(
                    finding_id=next_id(),
                    related_claim_ids=[grand_claim.claim_id] + [m.claim_id for m in monthly_rev_claims.values()],
                    relationship_type="monthly_subtotal_vs_grand_total",
                    expected_value=grand_val,
                    observed_value=round(monthly_sum, 4),
                    difference=round(diff, 4),
                    is_consistent=is_ok,
                    explanation=(
                        f"Sum of {len(monthly_rev_claims)} claimed monthly revenues ({monthly_sum:,.2f}) "
                        + ("matches" if is_ok else "differs from")
                        + f" claimed overall total ({grand_val:,.2f}) by {diff:+.2f}."
                    ),
                )
            )

        return findings
