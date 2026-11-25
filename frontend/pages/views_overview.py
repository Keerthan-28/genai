"""Streamlit Pages 1-4: Home, Dataset Overview, Sales Analytics, and Generated Report."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from backend.audit_logger import AuditLogger
from backend.llm_client import OllamaClientWrapper
from backend.pipeline import PipelineRunOutput, VeriReportPipeline
from backend.schemas import RevenueMode, TransactionPolicy
from config.settings import PROJECT_ROOT


def render_home_dashboard(pipeline: VeriReportPipeline, audit_logger: AuditLogger) -> None:
    st.title("VeriReport AI — Evidence-Based Retail Sales Report Verifier")
    st.markdown(
        "Upload a retail transaction dataset to generate an AI-authored sales report, "
        "independently recalculate every metric in Python, quantify numerical gaps, and audit every claim."
    )

    llm_check = OllamaClientWrapper(pipeline.settings).verify_model_availability(quick=True)
    if llm_check.model_verified:
        st.success(f"Local Ollama connected: `{llm_check.selected_model}` at `{pipeline.settings.ollama_base_url}`")
    else:
        st.info(
            f"Ollama status: {llm_check.message}. Deterministic Python fallback remains available for report & claim workflows."
        )

    runs = audit_logger.list_audit_runs()
    latest = runs[0] if runs else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Audit Runs", len(runs))
    c2.metric("Last Dataset", latest["filename"] if latest else "None yet")
    c3.metric("Last Report Model", latest["model_used"] if latest else "None yet")
    if latest:
        c4.metric(
            "Latest Verified / Total",
            f"{latest['verified_count']} / {latest['total_claims']}",
        )
    else:
        c4.metric("Latest Verified / Total", "0 / 0")

    st.subheader("Run New Audit Pipeline")
    with st.form("upload_and_run_form"):
        uploaded = st.file_uploader("Upload Retail Transaction Dataset (CSV or Excel)", type=["csv", "xlsx", "xls"])
        use_sample = st.checkbox("Or run using bundled sample dataset (`sample_retail_transactions.csv`)", value=uploaded is None)

        col_a, col_b, col_c = st.columns(3)
        rev_mode = col_a.selectbox(
            "Transaction Revenue Policy",
            [m.value for m in RevenueMode],
            index=0,
        )
        include_returns = col_b.checkbox("Include Returns", value=True)
        include_cancellations = col_c.checkbox("Include Cancellations", value=True)

        col_d, col_e, col_f = st.columns(3)
        dedup = col_d.checkbox("Deduplicate Exact Rows", value=True)
        currency_sym = col_e.selectbox("Currency Symbol", ["£", "$", "₹", "€"], index=0)
        currency_map = {"£": "GBP", "$": "USD", "₹": "INR", "€": "EUR"}
        use_llm = col_f.checkbox(
            "Use Local Ollama for Report & Claims",
            value=False,
            help="When checked, uses local Ollama (llama3.2) for report & claim extraction (~1–2 min on CPU). Uncheck for fast (<5s) deterministic mode.",
        )

        use_llm_explanations = st.checkbox(
            "Also use Ollama for per-claim explanations (runs 1 LLM call per claim; slower)",
            value=False,
        )

        submitted = st.form_submit_button("Run VeriReport AI Audit", type="primary")

    if submitted:
        policy = TransactionPolicy(
            revenue_mode=RevenueMode(rev_mode),
            include_returns=include_returns,
            include_cancellations=include_cancellations,
            deduplicate_exact_rows=dedup,
            currency_symbol=currency_sym,
            currency_code=currency_map[currency_sym],
        )
        with st.status("Running VeriReport AI Audit...", expanded=True) as status_box:
            def _update(msg: str) -> None:
                status_box.write(msg)

            if uploaded is not None:
                output = pipeline.run_from_bytes(
                    content=uploaded.getvalue(),
                    filename=uploaded.name,
                    policy=policy,
                    use_llm_report=use_llm,
                    use_llm_extraction=use_llm,
                    use_llm_explanations=use_llm_explanations,
                    progress_callback=_update,
                )
            elif use_sample:
                sample_path = PROJECT_ROOT / "data" / "samples" / "sample_retail_transactions.csv"
                output = pipeline.run_from_path(
                    sample_path,
                    policy=policy,
                    use_llm_report=use_llm,
                    use_llm_extraction=use_llm,
                    use_llm_explanations=use_llm_explanations,
                    progress_callback=_update,
                )
            else:
                status_box.update(label="Missing dataset", state="error")
                st.error("Please upload a CSV/Excel dataset or select the bundled sample dataset.")
                return
            status_box.update(
                label=f"Audit `{output.run_id}` completed in {output.total_time_seconds:.2f}s",
                state="complete",
            )

        st.session_state["active_run"] = output
        st.success(f"Audit run `{output.run_id}` completed in {output.total_time_seconds:.2f}s.")


def render_dataset_overview(run: PipelineRunOutput) -> None:
    st.title("Dataset Overview & Data Quality Profile")
    prof = run.profile
    meta = run.metadata

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Raw Rows", prof.total_raw_rows)
    c2.metric("Processed Rows", prof.total_processed_rows)
    c3.metric("Duplicates", prof.duplicate_rows)
    c4.metric("Returns / Cancels", f"{prof.return_rows} / {prof.cancellation_rows}")
    c5.metric("Quality Score", f"{prof.data_quality_score:.2f}")

    st.caption(
        f"File: `{meta.filename}` | Format: `{meta.file_format}` | "
        f"Date Coverage: `{prof.date_min}` to `{prof.date_max}` | SHA-256: `{meta.dataset_hash[:16]}...`"
    )

    if prof.partial_periods:
        st.warning(f"Partial period coverage detected for: {', '.join(prof.partial_periods)}.")
    for note in prof.quality_notes:
        st.info(note)

    tab1, tab2, tab3, tab4 = st.tabs(["Data Preview", "Columns & Missing Values", "Period Coverage", "Transformations"])
    with tab1:
        st.dataframe(run.raw_df.head(50), use_container_width=True)
    with tab2:
        col_df = pd.DataFrame(
            [
                {
                    "Column": col,
                    "Data Type": prof.dtypes.get(col, ""),
                    "Missing Count": prof.missing_values.get(col, 0),
                    "Missing Rate": f"{prof.missing_rates.get(col, 0.0):.2%}",
                }
                for col in prof.dtypes
            ]
        )
        st.dataframe(col_df, use_container_width=True)
    with tab3:
        cov_df = pd.DataFrame([c.model_dump() for c in prof.period_coverage])
        st.dataframe(cov_df, use_container_width=True)
    with tab4:
        tr_df = pd.DataFrame([t.model_dump() for t in prof.transformations])
        st.dataframe(tr_df, use_container_width=True)


def render_sales_analytics(run: PipelineRunOutput) -> None:
    st.title("Deterministic Python Sales Analytics")
    a = run.analytics
    s = a["summary"]
    sym = a["currency_symbol"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue (Policy)", f"{sym}{s['total_revenue']:,.2f}")
    c2.metric("Gross Revenue", f"{sym}{s['gross_revenue']:,.2f}")
    c3.metric("Net Revenue", f"{sym}{s['net_revenue']:,.2f}")
    c4.metric("Average Order Value", f"{sym}{s['average_order_value']:,.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Quantity Sold", f"{s['quantity_sold']:,.0f}")
    c6.metric("Distinct Orders", s["distinct_orders"])
    c7.metric("Returns Impact", f"{sym}{s['returns_amount']:,.2f}")
    c8.metric("Cancellations Impact", f"{sym}{s['cancellations_amount']:,.2f}")

    monthly_df = pd.DataFrame(a.get("monthly_revenue", []))
    if not monthly_df.empty:
        col_l, col_r = st.columns(2)
        with col_l:
            fig_m = px.bar(monthly_df, x="period", y="revenue", title="Monthly Revenue", text_auto=".2s")
            st.plotly_chart(fig_m, use_container_width=True)
        with col_r:
            growth_df = monthly_df.dropna(subset=["mom_growth_pct"])
            if not growth_df.empty:
                fig_g = px.line(growth_df, x="period", y="mom_growth_pct", markers=True, title="Month-over-Month Growth (%)")
                st.plotly_chart(fig_g, use_container_width=True)
            else:
                st.info("Not enough sequential non-zero months to plot MoM growth.")

    prod_df = pd.DataFrame(a.get("product_performance", []))
    country_df = pd.DataFrame(a.get("country_revenue", []))
    col_p, col_c = st.columns(2)
    with col_p:
        st.subheader("Top Products by Revenue")
        if not prod_df.empty:
            fig_p = px.bar(prod_df.head(10), x="revenue", y="product", orientation="h", title="Top 10 Products")
            st.plotly_chart(fig_p, use_container_width=True)
            st.dataframe(prod_df, use_container_width=True)
    with col_c:
        st.subheader("Country-wise Revenue")
        if not country_df.empty:
            fig_co = px.pie(country_df.head(10), values="revenue", names="country", title="Revenue Share by Country")
            st.plotly_chart(fig_co, use_container_width=True)
            st.dataframe(country_df, use_container_width=True)


def render_generated_report(run: PipelineRunOutput) -> None:
    st.title("Generated Retail Sales Report")
    rep = run.report
    st.caption(
        f"Report ID: `{rep.report_id}` | Dataset: `{rep.dataset_id}` | "
        f"Generator: `{rep.model_used}` (`{rep.generation_mode}`) | Generated: `{rep.generated_at}`"
    )

    dl1, dl2, dl3 = st.columns(3)
    if rep.pdf_path and Path(rep.pdf_path).is_file():
        dl1.download_button(
            "Download PDF",
            data=Path(rep.pdf_path).read_bytes(),
            file_name=Path(rep.pdf_path).name,
            mime="application/pdf",
        )
    if rep.docx_path and Path(rep.docx_path).is_file():
        dl2.download_button(
            "Download DOCX",
            data=Path(rep.docx_path).read_bytes(),
            file_name=Path(rep.docx_path).name,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    dl3.download_button(
        "Download TXT",
        data=rep.full_text.encode("utf-8"),
        file_name=f"{rep.report_id}.txt",
        mime="text/plain",
    )

    for sec in rep.sections:
        with st.expander(f"{sec.section_number}. {sec.title}", expanded=True):
            st.write(sec.content)
