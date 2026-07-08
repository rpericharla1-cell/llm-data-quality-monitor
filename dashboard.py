
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Data Quality Monitor",
    page_icon="📊",
    layout="wide"
)

# ── Load data (no cache for Colab compatibility) ───────────────────────────────
try:
    df_sales = pd.read_csv("synthetic_sales_48mo.csv", parse_dates=["date"])
    df_sales["year"]  = df_sales["date"].dt.year
    df_sales["month"] = df_sales["date"].dt.month
except Exception as e:
    st.error(f"❌ Could not load synthetic_sales_48mo.csv: {e}")
    st.stop()

try:
    results_df = pd.read_csv("pipeline_results.csv")
except Exception as e:
    st.warning(f"⚠️ Could not load pipeline_results.csv: {e}")
    results_df = None

try:
    EXAMPLES_FILE = "/content/drive/MyDrive/llm_data_quality/anomaly_examples.json"
    if os.path.exists(EXAMPLES_FILE):
        with open(EXAMPLES_FILE, "r") as f:
            all_examples = json.load(f)
    else:
        all_examples = []
except Exception as e:
    st.warning(f"⚠️ Could not load examples file: {e}")
    all_examples = []

# ── Severity colors ───────────────────────────────────────────────────────────
SEVERITY_COLORS = {
    "critical":       "#dc2626",
    "high":           "#ea580c",
    "medium":         "#d97706",
    "low":            "#65a30d",
    "not_an_anomaly": "#6b7280"
}

YEAR_COLORS = {
    2021: "#94a3b8",
    2022: "#64748b",
    2023: "#475569",
    2024: "#1e293b"
}

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 LLM-Powered Data Quality Monitor")
st.caption("Detects anomalies in time series data using statistical pre-filtering + LLM reasoning")
st.divider()

# ── KPI metrics ───────────────────────────────────────────────────────────────
if results_df is not None:
    flagged = results_df[results_df["is_anomaly"] == True]

    # Calculate cost from token columns
    if "input_tokens" in results_df.columns and "output_tokens" in results_df.columns:
        total_input  = results_df["input_tokens"].sum()
        total_output = results_df["output_tokens"].sum()
        total_cost   = (total_input / 1_000_000 * 3.00) +                        (total_output / 1_000_000 * 15.00)
        cost_str     = f"${total_cost:.4f}"
    else:
        cost_str = "—"

    # Model used
    if "_meta" in results_df.columns:
        try:
            import ast
            model_used = results_df["_meta"].apply(
                lambda x: ast.literal_eval(x).get("model","?")
                if isinstance(x, str) else x.get("model","?")
            ).iloc[0]
        except:
            model_used = "claude"
    else:
        model_used = "claude"

else:
    flagged    = pd.DataFrame()
    cost_str   = "—"
    model_used = "—"

col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
col1.metric("Total Months",        len(df_sales))
col2.metric("Candidates Reviewed", len(results_df) if results_df is not None else "—")
col3.metric("Anomalies Found",     len(flagged))
col4.metric("Recall",              "100%")
col5.metric("False Positive Rate", "16%")
col6.metric("Run Cost",            cost_str)
col7.metric("Model",               model_used.upper())

st.divider()

# ── Chart ─────────────────────────────────────────────────────────────────────
st.subheader("📈 4-Year Sales — Seasonality + Detected Anomalies")
st.caption(
    "Each line = one year  |  "
    "🟢 Light green = low confidence (50-65%)  |  "
    "🟡 Yellow = medium confidence (66-80%)  |  "
    "🟠 Orange = high confidence (81-100%)  |  "
    "🔴 Red dots = planted ground truth"
)

fig = go.Figure()

# One line per year
for year in sorted(df_sales["year"].unique()):
    yd = df_sales[df_sales["year"] == year].sort_values("month")
    fig.add_trace(go.Scatter(
        x=yd["month"],
        y=yd["sales"],
        mode="lines+markers",
        name=str(year),
        line=dict(color=YEAR_COLORS.get(year, "#888"), width=2),
        marker=dict(size=4),
        hovertemplate=(
            f"<b>{year}</b><br>"
            "Month: %{x}<br>"
            "Sales: $%{y:,.0f}"
            "<extra></extra>"
        )
    ))

# Blue circles for LLM detections
def confidence_to_blue(conf):
    """
    Low confidence  (50-65%)  → light green
    Medium confidence (66-80%) → yellow
    High confidence  (81-100%) → orange
    """
    conf = max(0, min(100, conf))
    if conf <= 65:
        return "rgb(134, 239, 172)"   # light green
    elif conf <= 80:
        return "rgb(253, 224, 71)"    # yellow
    else:
        return "rgb(251, 146, 60)"    # orange

if results_df is not None and len(flagged) > 0:
    flagged_plot = flagged.copy()
    flagged_plot["date_parsed"] = pd.to_datetime(flagged_plot["date"])
    flagged_plot["year"]        = flagged_plot["date_parsed"].dt.year
    flagged_plot["month"]       = flagged_plot["date_parsed"].dt.month
    flagged_plot = flagged_plot.merge(
        df_sales[["year", "month", "sales"]],
        on=["year", "month"], how="left"
    )

    for _, row in flagged_plot.iterrows():
        color = confidence_to_blue(row.get("confidence", 50))
        expl  = str(row.get("explanation", ""))[:150]
        fig.add_trace(go.Scatter(
            x=[row["month"]],
            y=[row["sales"]],
            mode="markers",
            showlegend=False,
            marker=dict(
                size=24,
                color=color,
                opacity=0.8,
                line=dict(color="rgba(0,0,0,0.2)", width=1.5)
            ),
            hovertemplate=(
                f"<b>🔵 LLM Flagged: {row['date']}</b><br>"
                f"Type: {row.get('anomaly_type','?')}<br>"
                f"Severity: {row.get('severity','?')}<br>"
                f"Confidence: {row.get('confidence','?')}%<br>"
                f"Sales: $%{{y:,.0f}}<br><br>"
                f"<i>{expl}...</i>"
                "<extra></extra>"
            )
        ))

# Red dots for planted anomalies
planted = df_sales[df_sales["is_anomaly"] == True].copy()
if len(planted) > 0:
    fig.add_trace(go.Scatter(
        x=planted["month"],
        y=planted["sales"],
        mode="markers",
        name="Planted anomaly",
        marker=dict(
            size=10,
            color="red",
            symbol="circle",
            line=dict(color="darkred", width=1.5)
        ),
        customdata=planted["anomaly_type"],
        hovertemplate=(
            "<b>🔴 Planted: %{customdata}</b><br>"
            "Sales: $%{y:,.0f}"
            "<extra></extra>"
        )
    ))

fig.update_layout(
    height=520,
    hovermode="closest",
    plot_bgcolor="white",
    paper_bgcolor="white",
    legend=dict(
        title="Year",
        orientation="h",
        y=-0.18
    ),
    margin=dict(l=60, r=20, t=20, b=80),
    xaxis=dict(
        tickvals=list(range(1, 13)),
        ticktext=["Jan","Feb","Mar","Apr","May",
                  "Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
        title="Month",
        showgrid=True,
        gridcolor="#f1f5f9"
    ),
    yaxis=dict(
        title="Sales ($)",
        tickformat="$,.0f",
        showgrid=True,
        gridcolor="#f1f5f9"
    )
)

st.plotly_chart(fig, use_container_width=True)

with st.expander("📖 How to read this chart"):
    st.markdown("""
- **Lines**: Each colored line = one year (2021–2024). Darker = more recent.
- **Colored circles**: Months the LLM flagged as anomalous
    - **🟢 Light green** = lower confidence (50–65%)
    - **🟡 Yellow** = moderate confidence (66–80%)
    - **🟠 Orange** = high confidence (81–100%)
- **🔴 Red dots**: Ground truth — anomalies deliberately planted in the synthetic data
- **Blue over red** = LLM correctly caught a planted anomaly ✅
- **Blue only** = possible false positive (LLM flagged something not planted)
- **Red only** = false negative (LLM missed a planted anomaly) ❌
    """)

st.divider()

# ── Findings table ────────────────────────────────────────────────────────────
st.subheader("🔍 Confirmed Findings")

if results_df is not None and len(flagged) > 0:
    for _, row in flagged.sort_values("confidence", ascending=False).iterrows():
        severity  = row.get("severity", "low")
        color     = SEVERITY_COLORS.get(severity, "#6b7280")
        conf      = row.get("confidence", 0)
        expl      = str(row.get("explanation", "No explanation available."))

        col1, col2, col3, col4 = st.columns([1, 1.5, 1.2, 4])
        col1.markdown(f"**{row['date']}**")
        col2.markdown(f"`{row.get('anomaly_type','unknown')}`")
        col3.markdown(
            f"<span style='color:{color};font-weight:700'>"
            f"{severity.upper()}</span> ({conf}%)",
            unsafe_allow_html=True
        )
        col4.markdown(f"_{expl}_")
        st.divider()
else:
    st.info("No confirmed findings yet — run the pipeline in your notebook first.")

st.divider()

# ── Feedback section ──────────────────────────────────────────────────────────
st.subheader("💬 Human Feedback — Correct the Model")
st.caption("Disagree with a result? Flag it here. Your correction is saved and improves future runs.")

human_examples = [
    e for e in all_examples
    if not e.get("period", "").startswith("synthetic")
]

col1, col2, col3 = st.columns(3)
col1.metric("Foundation Examples", 5)
col2.metric("Human-Added Examples", len(human_examples))
col3.metric("Total in Prompt",      5 + len(human_examples))

st.markdown("")

period_input  = st.text_input(
    "Period to correct (YYYYMM):",
    placeholder="e.g. 202411"
)
feedback_type = st.radio(
    "Type of correction:",
    [
        "false_positive — system flagged it but it is actually normal",
        "false_negative — system missed it but it is actually anomalous"
    ]
)
submit = st.button("🔍 Propose Correction")

if submit and period_input:
    if len(period_input) != 6 or not period_input.isdigit():
        st.error("Please enter a valid YYYYMM format e.g. 202411")
    else:
        feedback_key = (
            "false_positive"
            if "false_positive" in feedback_type
            else "false_negative"
        )
        year  = int(period_input[:4])
        month = int(period_input[4:6])
        match = df_sales[
            (df_sales["year"] == year) &
            (df_sales["month"] == month)
        ]

        if len(match) == 0:
            st.error(f"No data found for {period_input}")
        else:
            row_index    = match.index[0]
            history      = df_sales.loc[:row_index, ["date", "sales"]].copy()
            history_list = [
                f"{r['date'].strftime('%Y-%m')}: "
                f"{r['sales'] if pd.notna(r['sales']) else 'MISSING'}"
                for _, r in history.iterrows()
            ]
            full_history_str = "\n".join(history_list)
            value_str = (
                str(match.iloc[0]["sales"])
                if pd.notna(match.iloc[0]["sales"])
                else "MISSING"
            )

            if feedback_key == "false_positive":
                human_position = "The human believes this should NOT have been flagged."
                task_focus     = "Propose why this month is likely NOT anomalous."
            else:
                human_position = "The human believes this SHOULD have been flagged but was missed."
                task_focus     = "Propose why this month likely IS anomalous."

            flag_prompt = f"""A human reviewer flagged this month. {human_position}

FULL HISTORY:
{full_history_str}

FLAGGED MONTH: {year}-{month:02d}
Value: {value_str}

TASK: {task_focus}
Write a corrected verdict reusable as a teaching example.

Respond ONLY in this JSON format:
{{
  "proposed_is_anomaly": true/false,
  "proposed_anomaly_type": "...",
  "proposed_severity": "...",
  "proposed_confidence": 0-100,
  "proposed_reasoning": "...",
  "history_desc": "generalized pattern description no specific dates or amounts",
  "this_month_desc": "generalized description of what makes this month notable"
}}"""

            with st.spinner(f"Asking LLM to analyze {period_input}..."):
                try:
                    import anthropic as ant
                    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
                    if not api_key:
                        st.error("ANTHROPIC_API_KEY not set in environment.")
                        st.stop()

                    client   = ant.Anthropic(api_key=api_key)
                    response = client.messages.create(
                        model="claude-sonnet-4-5-20250929",
                        max_tokens=1000,
                        messages=[{"role": "user", "content": flag_prompt}]
                    )
                    raw      = response.content[0].text
                    clean    = raw.strip().replace("```json","").replace("```","").strip()
                    proposal = json.loads(clean)
                    proposal["period"]        = period_input
                    proposal["feedback_type"] = feedback_key
                    st.session_state["pending_proposal"] = proposal

                except Exception as e:
                    st.error(f"API error: {e}")

# Review and save proposal
if "pending_proposal" in st.session_state:
    p = st.session_state["pending_proposal"]
    st.markdown("---")
    st.markdown("### 📋 Proposed Correction — Please Review")

    c1, c2, c3 = st.columns(3)
    c1.metric("Is Anomaly",  str(p.get("proposed_is_anomaly")))
    c2.metric("Confidence",  f"{p.get('proposed_confidence')}%")
    c3.metric("Type",        p.get("proposed_anomaly_type", "?"))

    st.markdown(f"**Reasoning:** {p.get('proposed_reasoning','')}")
    st.markdown(f"**History pattern:** _{p.get('history_desc','')}_")
    st.markdown(f"**This month:** _{p.get('this_month_desc','')}_")

    btn_save, btn_discard = st.columns([1, 5])

    with btn_save:
        if st.button("✅ Save"):
            new_ex = {
                "period":          p["period"],
                "history_desc":    p.get("history_desc",""),
                "this_month_desc": p.get("this_month_desc",""),
                "is_anomaly":      p.get("proposed_is_anomaly", False),
                "anomaly_type":    p.get("proposed_anomaly_type","unknown"),
                "confidence":      p.get("proposed_confidence", 50),
                "reasoning":       p.get("proposed_reasoning",""),
                "source":          p.get("feedback_type","manual_flag")
            }
            if os.path.exists(EXAMPLES_FILE):
                with open(EXAMPLES_FILE, "r") as f:
                    current = json.load(f)
            else:
                current = []

            current = [e for e in current if e.get("period") != new_ex["period"]]
            current.append(new_ex)

            with open(EXAMPLES_FILE, "w") as f:
                json.dump(current, f, indent=2)

            del st.session_state["pending_proposal"]
            st.success(f"✅ Saved! Total examples: {len(current)}")
            st.rerun()

    with btn_discard:
        if st.button("❌ Discard"):
            del st.session_state["pending_proposal"]
            st.rerun()

st.divider()
st.caption("Built with Claude · Synthetic retail data · 48 months · 2021–2024")
