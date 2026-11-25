"""VeriReport AI — Main Streamlit 10-page interactive dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable when invoked via `streamlit run frontend/app.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from backend.pipeline import VeriReportPipeline
from frontend.pages.views_audit import (
    render_audit_history_and_exports,
    render_human_review_queue,
)
from frontend.pages.views_overview import (
    render_dataset_overview,
    render_generated_report,
    render_home_dashboard,
    render_sales_analytics,
)
from frontend.pages.views_verification import (
    render_claim_level_comparison,
    render_claim_verification_dashboard,
    render_consistency_analysis,
    render_evidence_and_gap_analysis,
)

st.set_page_config(
    page_title="VeriReport AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = (
    "1. Home Dashboard",
    "2. Dataset Overview",
    "3. Sales Analytics",
    "4. Generated Report",
    "5. Claim Verification Dashboard",
    "6. Claim-Level Comparison",
    "7. Evidence and Gap Analysis",
    "8. Consistency Analysis",
    "9. Human Review Queue",
    "10. Audit History and Exports",
)


@st.cache_resource
def get_pipeline() -> VeriReportPipeline:
    return VeriReportPipeline()


def main() -> None:
    pipeline = get_pipeline()
    st.sidebar.title("VeriReport AI")
    st.sidebar.caption("Evidence-Based Sales Report Verification")
    selected_page = st.sidebar.radio("Navigation", PAGES)

    active_run = st.session_state.get("active_run")
    if active_run is not None:
        st.sidebar.success(f"Active Run: `{active_run.run_id}`")

    if selected_page == PAGES[0]:
        render_home_dashboard(pipeline, pipeline.audit_logger)
        return
    if selected_page == PAGES[9]:
        render_audit_history_and_exports(pipeline.audit_logger)
        return

    if active_run is None:
        st.warning("No active dataset/report run in this session. Upload or run the sample dataset on the **Home Dashboard** first.")
        return

    if selected_page == PAGES[1]:
        render_dataset_overview(active_run)
    elif selected_page == PAGES[2]:
        render_sales_analytics(active_run)
    elif selected_page == PAGES[3]:
        render_generated_report(active_run)
    elif selected_page == PAGES[4]:
        render_claim_verification_dashboard(active_run)
    elif selected_page == PAGES[5]:
        render_claim_level_comparison(active_run)
    elif selected_page == PAGES[6]:
        render_evidence_and_gap_analysis(active_run)
    elif selected_page == PAGES[7]:
        render_consistency_analysis(active_run)
    elif selected_page == PAGES[8]:
        render_human_review_queue(active_run, pipeline)


if __name__ == "__main__":
    main()
