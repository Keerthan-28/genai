"""Streamlit Pages 5-8: Verification Dashboard, Claim Comparison, Evidence & Gap, Consistency."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from backend.schemas import GapType, VerificationStatus
from backend.pipeline import PipelineRunOutput
from frontend.components.ui_helpers import STATUS_COLORS, results_dataframe


def render_claim_verification_dashboard(run: PipelineRunOutput) -> None:
    st.title("Claim Verification Dashboard")
    results = run.results
    counts = {s.value: sum(1 for r in results if r.status == s) for s in VerificationStatus}

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Claims", len(results))
    c2.metric("Verified", counts[VerificationStatus.VERIFIED.value])
    c3.metric("Contradicted", counts[VerificationStatus.CONTRADICTED.value])
    c4.metric("Insufficient Evidence", counts[VerificationStatus.INSUFFICIENT_EVIDENCE.value])
    c5.metric("Needs Review", counts[VerificationStatus.NEEDS_REVIEW.value])
    c6.metric("Partially Verified", counts[VerificationStatus.PARTIALLY_VERIFIED.value])

    status_df = pd.DataFrame(
        [{"Status": k, "Count": v} for k, v in counts.items()]
    )
    fig = px.bar(
        status_df,
        x="Status",
        y="Count",
        color="Status",
        color_discrete_map=STATUS_COLORS,
        title="Verification Status Distribution",
    )
    st.plotly_chart(fig, use_container_width=True)

    df_flat = results_dataframe(results, include_subclaims=True)
    if df_flat.empty:
        st.info("No claims extracted for this run.")
        return

    st.subheader("Verification Summary Table")
    st.dataframe(
        df_flat[
            ["claim_id", "section", "metric_id", "period", "claimed_value", "actual_value", "signed_gap", "gap_type", "status", "evidence_quality"]
        ],
        use_container_width=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Largest Numerical Gaps")
        numeric_gaps = df_flat.dropna(subset=["absolute_gap"]).sort_values("absolute_gap", ascending=False)
        st.dataframe(numeric_gaps.head(10)[["claim_id", "metric_id", "claimed_value", "actual_value", "signed_gap", "gap_type"]], use_container_width=True)
    with col_b:
        st.subheader("Overstated & Understated Claims")
        discrepant = df_flat[df_flat["gap_type"].isin([GapType.OVERSTATED.value, GapType.UNDERSTATED.value])]
        st.dataframe(discrepant[["claim_id", "claim_text", "signed_gap", "relative_gap_pct", "gap_type"]], use_container_width=True)

    review_df = df_flat[df_flat["needs_human_review"] == True]  # noqa: E712
    st.subheader("Claims Requiring Human Review")
    if review_df.empty:
        st.success("No claims currently require human review.")
    else:
        st.dataframe(review_df[["claim_id", "claim_text", "status", "review_reason"]], use_container_width=True)


def render_claim_level_comparison(run: PipelineRunOutput) -> None:
    st.title("Claim-Level Comparison")
    df_flat = results_dataframe(run.results, include_subclaims=True)
    if df_flat.empty:
        st.info("No claims available.")
        return

    f1, f2, f3, f4 = st.columns(4)
    sel_status = f1.multiselect("Filter Status", sorted(df_flat["status"].unique()))
    sel_metric = f2.multiselect("Filter Metric", sorted(df_flat["metric_id"].unique()))
    sel_period = f3.multiselect("Filter Period", sorted(df_flat["period"].unique()))
    sel_gap = f4.multiselect("Filter Gap Type", sorted(df_flat["gap_type"].unique()))

    filtered = df_flat.copy()
    if sel_status:
        filtered = filtered[filtered["status"].isin(sel_status)]
    if sel_metric:
        filtered = filtered[filtered["metric_id"].isin(sel_metric)]
    if sel_period:
        filtered = filtered[filtered["period"].isin(sel_period)]
    if sel_gap:
        filtered = filtered[filtered["gap_type"].isin(sel_gap)]

    st.caption(f"Showing {len(filtered)} of {len(df_flat)} claim/subclaim record(s).")

    for _, row in filtered.iterrows():
        header = f"{row['claim_id']} | {row['status']} | {row['metric_id']} ({row['period']})"
        with st.expander(header):
            st.markdown(f"**Original Report Sentence:** {row['source_sentence']}")
            st.markdown(f"**Extracted Claim:** {row['claim_text']}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Claimed Value", str(row["claimed_value"]))
            m2.metric("Actual Value", str(row["actual_value"]))
            m3.metric("Signed Gap", f"{row['signed_gap']:.2f}" if pd.notna(row["signed_gap"]) else "N/A")
            m4.metric(
                "Relative Gap (%)",
                f"{row['relative_gap_pct']:.2f}%" if pd.notna(row["relative_gap_pct"]) else "N/A",
            )
            st.write(
                {
                    "Gap Classification": row["gap_type"],
                    "Verification Status": row["status"],
                    "Original Machine Status": row["original_machine_status"],
                    "Evidence Quality": f"{row['evidence_quality']} ({row['quality_score']:.2f})",
                    "Formula": row["formula_used"],
                    "Applied Filters": row["applied_filters"],
                    "Evidence Gate": f"{row['evidence_reason']} — {row['evidence_detail']}",
                }
            )
            st.info(row["explanation"])


def render_evidence_and_gap_analysis(run: PipelineRunOutput) -> None:
    st.title("Evidence Quality & Gap Analysis")
    df_flat = results_dataframe(run.results, include_subclaims=True)
    if df_flat.empty:
        st.info("No claims available.")
        return

    c1, c2 = st.columns(2)
    with c1:
        q_counts = df_flat["evidence_quality"].value_counts().reset_index()
        q_counts.columns = ["Evidence Quality", "Count"]
        st.plotly_chart(px.pie(q_counts, values="Count", names="Evidence Quality", title="Evidence Quality Distribution"), use_container_width=True)
    with c2:
        g_counts = df_flat["gap_type"].value_counts().reset_index()
        g_counts.columns = ["Gap Classification", "Count"]
        st.plotly_chart(px.bar(g_counts, x="Gap Classification", y="Count", title="Gap Classification Breakdown"), use_container_width=True)

    st.subheader("Missing / Insufficient Evidence Reasons")
    missing_df = df_flat[df_flat["evidence_reason"] != "SUFFICIENT_EVIDENCE"]
    if missing_df.empty:
        st.success("All evaluated claims had sufficient evidence.")
    else:
        st.dataframe(missing_df[["claim_id", "claim_text", "evidence_reason", "evidence_detail", "evidence_quality"]], use_container_width=True)

    st.subheader("Overstated vs Understated & Mismatches")
    mismatch_df = df_flat[df_flat["gap_type"] != GapType.EXACT_MATCH.value]
    st.dataframe(
        mismatch_df[["claim_id", "metric_id", "claimed_value", "actual_value", "signed_gap", "absolute_gap", "relative_gap_pct", "gap_type"]],
        use_container_width=True,
    )


def render_consistency_analysis(run: PipelineRunOutput) -> None:
    st.title("Cross-Claim Consistency Analysis")
    st.caption(
        "Cross-checks arithmetic and ranking relationships across extracted claims "
        "(product totals vs overall revenue, contribution percentages, MoM growth vs monthly revenues, "
        "rankings vs product revenues, and monthly subtotals vs grand total)."
    )
    findings = run.consistency_findings
    if not findings:
        st.info(
            "No eligible multi-claim pairs were found in the extracted report to run cross-claim checks "
            "(for example, pairs of monthly revenues + growth rate, or multiple product revenues + overall total)."
        )
        return

    inconsistent = [f for f in findings if not f.is_consistent]
    consistent = [f for f in findings if f.is_consistent]

    c1, c2, c3 = st.columns(3)
    c1.metric("Cross-Claim Checks Evaluated", len(findings))
    c2.metric("Consistent Checks", len(consistent))
    c3.metric("Inconsistencies Detected", len(inconsistent))

    if not inconsistent:
        st.success(f"All {len(consistent)} evaluated cross-claim check(s) are internally consistent.")
    else:
        st.warning(f"Detected {len(inconsistent)} cross-claim inconsistency/inconsistencies.")

    con_df = pd.DataFrame([f.model_dump() for f in findings])
    st.dataframe(con_df, use_container_width=True)
