"""Streamlit Pages 9-10: Human Review Queue and Audit History & Exports."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from backend.audit_logger import AuditLogger
from backend.pipeline import PipelineRunOutput, VeriReportPipeline
from backend.schemas import VerificationStatus


def render_human_review_queue(run: PipelineRunOutput, pipeline: VeriReportPipeline) -> None:
    st.title("Human Review Queue")
    queue = pipeline.review_manager.get_review_queue(run.results)
    details = pipeline.audit_logger.get_run_details(run.run_id)

    if not queue:
        st.success("No pending claims require human review in this audit run.")
    else:
        claim_ids = [r.claim_id for r in queue]
        selected_id = st.selectbox("Select Claim for Review", claim_ids)
        target = next(r for r in queue if r.claim_id == selected_id)

        st.markdown(f"**Original Sentence:** {target.normalized_claim.source_sentence}")
        st.markdown(f"**Claim Text:** {target.normalized_claim.claim_text}")
        st.write(
            {
                "Current Status": target.status.value,
                "Original Machine Status": target.original_machine_status.value,
                "Current Metric Mapping": target.normalized_claim.metric_id,
                "Current Period": target.normalized_claim.normalized_period,
                "Current Entity": target.normalized_claim.normalized_entity,
                "Review Reason": target.review_reason or target.evidence_gate.reason_detail,
            }
        )

        with st.form("human_review_form"):
            metric_options = [""] + [m.metric_id for m in pipeline.registry.list_all()]
            corrected_metric = st.selectbox("Correct Metric Mapping (optional)", metric_options)
            corrected_period = st.text_input(
                "Clarify Reporting Period (YYYY-MM, optional)",
                value=target.normalized_claim.normalized_period or "",
            )
            corrected_entity = st.text_input(
                "Clarify Entity Name (optional)",
                value=target.normalized_claim.normalized_entity or "",
            )
            action_type = st.selectbox(
                "Review Action",
                ["remap", "clarify_period", "clarify_entity", "rerun", "confirm", "reject"],
            )
            reviewer = st.text_input("Reviewer Name", value="analyst")
            ext_ref = st.text_input("External Evidence Reference (optional)")
            notes = st.text_area("Review Notes")
            submit_rev = st.form_submit_button("Apply Review & Re-run Verification", type="primary")

        if submit_rev:
            updated, action = pipeline.review_manager.apply_review_action(
                target,
                action_type=action_type,
                reviewer=reviewer,
                corrected_metric_id=corrected_metric or None,
                corrected_period=corrected_period.strip() or None,
                corrected_entity=corrected_entity.strip() or None,
                external_evidence_ref=ext_ref.strip() or None,
                notes=notes,
                annotated_df=run.annotated_df,
                processed_df=run.processed_df,
                profile=run.profile,
                policy=run.policy,
            )
            updated = pipeline.explanation_generator.explain_result(updated, use_llm=False)
            run.results = [updated if r.claim_id == updated.claim_id else r for r in run.results]
            pipeline.audit_logger.record_review_action(run.run_id, updated, action)
            st.success(
                f"Review recorded (`{action.review_id}`). Status: `{updated.status.value}` "
                f"(Original machine status preserved: `{updated.original_machine_status.value}`)."
            )
            details = pipeline.audit_logger.get_run_details(run.run_id)

    st.subheader("Audit Review History")
    rev_history = details.get("review_actions", [])
    if rev_history:
        st.dataframe(pd.DataFrame(rev_history), use_container_width=True)
    else:
        st.info("No human review actions recorded for this run yet.")


def render_audit_history_and_exports(audit_logger: AuditLogger) -> None:
    st.title("Audit History & Verification Exports")
    runs = audit_logger.list_audit_runs()
    if not runs:
        st.info("No audit runs saved in SQLite yet. Run an analysis from the Home Dashboard first.")
        return

    runs_df = pd.DataFrame(runs)
    st.dataframe(
        runs_df[
            [
                "run_id",
                "filename",
                "model_used",
                "total_claims",
                "verified_count",
                "contradicted_count",
                "insufficient_count",
                "needs_review_count",
                "partially_verified_count",
                "processing_time_seconds",
                "created_at",
            ]
        ],
        use_container_width=True,
    )

    selected_run = st.selectbox("Inspect Historical Audit Run", [r["run_id"] for r in runs])
    details = audit_logger.get_run_details(selected_run)

    ver_df = pd.DataFrame(details.get("verifications", []))
    st.subheader(f"Verification Records for `{selected_run}`")
    if not ver_df.empty:
        st.dataframe(
            ver_df[
                ["claim_id", "status", "original_machine_status", "metric_id", "claimed_value", "actual_value", "gap_type", "quality_level"]
            ],
            use_container_width=True,
        )

    c1, c2, c3 = st.columns(3)
    csv_path = audit_logger.export_run_to_csv(selected_run)
    json_path = audit_logger.export_run_to_json(selected_run)
    summary_md = (
        f"# VeriReport AI Verification Summary — {selected_run}\n\n"
        f"- Total Claims: {details['run']['total_claims']}\n"
        f"- Verified: {details['run']['verified_count']}\n"
        f"- Contradicted: {details['run']['contradicted_count']}\n"
        f"- Insufficient Evidence: {details['run']['insufficient_count']}\n"
        f"- Needs Review: {details['run']['needs_review_count']}\n"
        f"- Partially Verified: {details['run']['partially_verified_count']}\n"
    )
    c1.download_button(
        "Download Verification CSV",
        data=Path(csv_path).read_bytes(),
        file_name=Path(csv_path).name,
        mime="text/csv",
    )
    c2.download_button(
        "Download Verification JSON",
        data=Path(json_path).read_bytes(),
        file_name=Path(json_path).name,
        mime="application/json",
    )
    c3.download_button(
        "Download Verification Summary (MD)",
        data=summary_md.encode("utf-8"),
        file_name=f"{selected_run}_summary.md",
        mime="text/markdown",
    )
