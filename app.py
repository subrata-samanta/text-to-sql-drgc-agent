"""Streamlit chat interface for the Nielsen Text-to-SQL Agent."""

import streamlit as st
import pandas as pd
from loguru import logger

from graph import stream_agent_steps, stream_nl_response, NODE_LABELS
from agents.interaction import DATA_INTENTS
from agents.filter_resolver import initialize_value_cache
from tools import seed_examples, semantic_cache, few_shot_retriever
from core.database import db_manager
from config import settings

initialize_value_cache()

# ─── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nielsen AI · Ask a Question",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
  --brand:       #0053A5;
  --accent:      #00A3E0;
  --bg:          #F4F6FA;
  --surface:     #FFFFFF;
  --border:      #E2E8F2;
  --text:        #0D1B2A;
  --muted:       #6B7A8D;
  --success:     #10B981;
  --warning-bg:  #FFFBEB;
  --warning-br:  #FCD34D;
  --r:           10px;
  --shadow:      0 2px 12px rgba(0,83,165,.09);
}

html, body, [class*="css"] {
  font-family: 'Inter', system-ui, sans-serif !important;
  background: var(--bg) !important;
  color: var(--text);
}

/* ── Narrow centered column ── */
.main .block-container {
  padding: 2rem 1rem 6rem !important;
  max-width: 820px !important;
}

/* ── Hide chrome ── */
#MainMenu, footer, header, [data-testid="stSidebar"] { display: none !important; }
.stDeployButton { display: none !important; }

/* ── App title bar ── */
.app-bar {
  display: flex;
  align-items: center;
  gap: .75rem;
  padding: 1.1rem 1.4rem;
  background: var(--brand);
  border-radius: var(--r);
  margin-bottom: 1.5rem;
  box-shadow: 0 4px 20px rgba(0,83,165,.22);
}
.app-bar-icon { font-size: 1.7rem; line-height: 1; }
.app-bar-title {
  font-size: 1.1rem;
  font-weight: 700;
  color: #fff;
  letter-spacing: -.01em;
}
.app-bar-sub {
  font-size: .72rem;
  color: rgba(255,255,255,.65);
  margin-top: 1px;
}
.app-bar-status {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: .72rem;
  color: rgba(255,255,255,.8);
}
.dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: #34D399;
  box-shadow: 0 0 6px #34D399;
  animation: blink 2s infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.35} }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
  border-radius: var(--r) !important;
  margin-bottom: .65rem !important;
  border: 1px solid transparent !important;
  padding: .9rem 1.1rem !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background: linear-gradient(135deg,#EEF4FF,#DBEAFE) !important;
  border-color: #C3D9FF !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
  background: var(--surface) !important;
  border-color: var(--border) !important;
  box-shadow: var(--shadow) !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
  border-radius: var(--r) !important;
  border: 2px solid var(--border) !important;
  box-shadow: 0 2px 16px rgba(0,83,165,.10) !important;
  font-size: .93rem !important;
  background: var(--surface) !important;
}
[data-testid="stChatInput"]:focus-within {
  border-color: var(--brand) !important;
  box-shadow: 0 0 0 3px rgba(0,83,165,.10), 0 2px 16px rgba(0,83,165,.10) !important;
}

/* ── Thinking / progress container ── */
.think-box {
  background: var(--surface);
  border: 1.5px solid var(--border);
  border-radius: var(--r);
  padding: .9rem 1.1rem;
  margin-bottom: .65rem;
  box-shadow: var(--shadow);
}
.think-title {
  font-size: .72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .07em;
  color: var(--muted);
  margin-bottom: .6rem;
}
.think-step {
  display: flex;
  align-items: flex-start;
  gap: .55rem;
  padding: .3rem 0;
  font-size: .84rem;
  color: var(--text);
  border-bottom: 1px solid var(--border);
}
.think-step:last-child { border-bottom: none; }
.step-icon { font-size: 1rem; flex-shrink: 0; margin-top: 1px; }
.step-body { flex: 1; line-height: 1.45; }
.step-label { font-weight: 600; }
.step-detail { font-size: .78rem; color: var(--muted); margin-top: 2px; }
.step-done   { color: var(--success); }
.step-active { color: var(--brand); }
.step-warn   { color: #D97706; }

/* ── Spinner dots ── */
.spinner { display: inline-flex; gap: 3px; margin-left: 4px; }
.spinner span {
  width: 5px; height: 5px; border-radius: 50%; background: var(--brand);
  animation: dot-pulse 1.2s infinite ease-in-out;
}
.spinner span:nth-child(2) { animation-delay: .2s; }
.spinner span:nth-child(3) { animation-delay: .4s; }
@keyframes dot-pulse { 0%,80%,100%{transform:scale(.6);opacity:.4} 40%{transform:scale(1);opacity:1} }

/* ── Plan box ── */
.plan-box {
  background: #F8FAFD;
  border-left: 3px solid var(--accent);
  border-radius: 0 6px 6px 0;
  padding: .65rem .9rem;
  font-size: .8rem;
  line-height: 1.65;
  color: #3A4A5C;
  white-space: pre-wrap;
  margin-top: .35rem;
}

/* ── Expander (details) ── */
[data-testid="stExpander"] {
  border: 1.5px solid var(--border) !important;
  border-radius: var(--r) !important;
  background: var(--surface) !important;
  box-shadow: var(--shadow) !important;
  margin-top: .5rem !important;
  overflow: hidden !important;
}
[data-testid="stExpander"] summary {
  background: #F8FAFD !important;
  padding: .55rem .9rem !important;
  font-size: .8rem !important;
  font-weight: 600 !important;
  color: var(--muted) !important;
  border-bottom: 1px solid var(--border) !important;
}
[data-testid="stExpander"] summary:hover {
  background: #EEF4FF !important;
  color: var(--brand) !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-testid="stTab"] {
  font-size: .8rem !important;
  font-weight: 600 !important;
  color: var(--muted) !important;
  border: none !important;
}
[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] {
  color: var(--brand) !important;
  border-bottom: 2.5px solid var(--brand) !important;
}
[data-testid="stTabsContent"] {
  border: 1px solid var(--border) !important;
  border-top: none !important;
  border-radius: 0 0 var(--r) var(--r) !important;
  padding: .85rem !important;
  background: var(--surface) !important;
}

/* ── Data table ── */
[data-testid="stDataFrame"] {
  border-radius: 8px !important;
  border: 1px solid var(--border) !important;
  overflow: hidden !important;
}
[data-testid="stDataFrame"] th {
  background: #EEF4FF !important;
  color: var(--brand) !important;
  font-size: .73rem !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: .03em !important;
}
[data-testid="stDataFrame"] td { font-size: .82rem !important; }

/* ── Metric pill strip ── */
.metric-strip {
  display: flex;
  gap: .5rem;
  flex-wrap: wrap;
  margin-bottom: .6rem;
}
.metric-pill {
  background: #F0F5FF;
  border: 1px solid #C3D9FF;
  border-radius: 20px;
  padding: 3px 12px;
  font-size: .75rem;
  color: var(--brand);
  font-weight: 600;
}
.metric-pill span { color: var(--muted); font-weight: 400; margin-right: 4px; }

/* ── Download button ── */
.stDownloadButton > button {
  background: linear-gradient(135deg, #0053A5, #00A3E0) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 6px !important;
  font-size: .77rem !important;
  font-weight: 600 !important;
  padding: .3rem .8rem !important;
}

/* ── Code ── */
[data-testid="stCode"] {
  border-radius: 8px !important;
  border: 1px solid var(--border) !important;
  font-size: .8rem !important;
}

/* ── Example pill buttons ── */
.stButton > button {
  border-radius: 6px !important;
  font-size: .79rem !important;
  font-weight: 500 !important;
  transition: all .18s !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
  border-radius: 8px !important;
  font-size: .83rem !important;
}

/* ── Status widget ── */
[data-testid="stStatus"] {
  border-radius: var(--r) !important;
  border: 1.5px solid var(--border) !important;
  background: var(--surface) !important;
}

/* ── Divider ── */
.hr { height: 1px; background: var(--border); border: none; margin: 1.2rem 0; }

/* ── Result section header ── */
.result-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: .65rem;
}
.result-header-title {
  font-size: .72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .07em;
  color: var(--muted);
}
.result-badge {
  font-size: .7rem;
  font-weight: 600;
  padding: 2px 10px;
  border-radius: 20px;
  background: #EEF4FF;
  border: 1px solid #C3D9FF;
  color: var(--brand);
}

/* ── KPI cards (single-row result) ── */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: .65rem;
  margin-bottom: .75rem;
}
.kpi-card {
  background: linear-gradient(135deg, #F0F5FF 0%, #EEF4FF 100%);
  border: 1.5px solid #C3D9FF;
  border-radius: var(--r);
  padding: .85rem 1rem;
  text-align: center;
  transition: box-shadow .18s;
}
.kpi-card:hover { box-shadow: 0 4px 16px rgba(0,83,165,.14); }
.kpi-label {
  font-size: .62rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .06em;
  color: var(--muted);
  margin-bottom: .4rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.kpi-value {
  font-size: 1.35rem;
  font-weight: 700;
  color: var(--brand);
  line-height: 1.1;
  word-break: break-word;
}
.kpi-value.kpi-text {
  font-size: .95rem;
  color: var(--text);
  font-weight: 600;
}

/* ── Summary stat strip ── */
.summary-strip {
  display: flex;
  gap: .4rem;
  flex-wrap: wrap;
  padding: .55rem .75rem;
  background: #F8FAFD;
  border: 1px solid var(--border);
  border-radius: 8px;
  margin-top: .55rem;
  margin-bottom: .25rem;
}
.summary-item {
  font-size: .72rem;
  color: var(--muted);
  display: flex;
  align-items: center;
  gap: 3px;
}
.summary-item b { color: var(--brand); font-weight: 600; }
.summary-sep {
  color: var(--border);
  font-size: .8rem;
  align-self: center;
}

/* ── Chart container ── */
.chart-wrap {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r);
  padding: .75rem .75rem .25rem;
  margin-bottom: .55rem;
}

/* ── Footer ── */
.app-footer {
  text-align: center;
  font-size: .68rem;
  color: var(--muted);
  padding: 2rem 0 .5rem;
}
</style>
""", unsafe_allow_html=True)

# ─── session state ─────────────────────────────────────────────────────────────
for _k, _v in [
    ("chat_history", []),
    ("show_correction_input", False),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─── data helpers ──────────────────────────────────────────────────────────────

def _rows_to_df(query_result):
    import decimal, datetime as dt
    if query_result is None:
        return None
    if isinstance(query_result, pd.DataFrame):
        return _normalise_df(query_result)
    if not isinstance(query_result, (list, tuple)) or len(query_result) == 0:
        return None
    try:
        first = query_result[0]
        if isinstance(first, dict):
            df = pd.DataFrame(query_result)
        elif hasattr(first, "_mapping"):
            df = pd.DataFrame([dict(r._mapping) for r in query_result])
        elif hasattr(first, "_asdict"):
            df = pd.DataFrame([r._asdict() for r in query_result])
        elif hasattr(first, "keys"):
            cols = list(first.keys())
            df = pd.DataFrame([dict(zip(cols, r)) for r in query_result])
        else:
            df = pd.DataFrame(query_result)
        return _normalise_df(df)
    except Exception as e:
        logger.warning(f"Could not convert query result to DataFrame: {e}")
        return None


def _normalise_df(df: pd.DataFrame) -> pd.DataFrame:
    import decimal, datetime as dt
    df = df.copy()
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, decimal.Decimal)).any():
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif df[col].apply(lambda x: isinstance(x, (dt.date, dt.datetime))).any():
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _col_config(df: pd.DataFrame) -> dict:
    cfg = {}
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            cfg[col] = st.column_config.NumberColumn(col, format="%.2f")
        elif pd.api.types.is_integer_dtype(df[col]):
            cfg[col] = st.column_config.NumberColumn(col, format="%d")
    return cfg


def _detect_display_mode(df: pd.DataFrame) -> str:
    """kpi=single row, chart=few rows with 1 category + numerics, table=rest."""
    if df is None or df.empty:
        return "empty"
    n_rows = len(df)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    text_cols    = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if n_rows == 1:
        return "kpi"
    if 2 <= n_rows <= 40 and len(text_cols) == 1 and len(numeric_cols) >= 1:
        return "chart"
    return "table"


def _fmt_val(val) -> tuple[str, bool]:
    """Return (formatted_string, is_numeric)."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—", False
    if isinstance(val, float):
        return f"{val:,.2f}", True
    if isinstance(val, int):
        return f"{val:,}", True
    try:
        import pandas as _pd
        if _pd.api.types.is_integer_dtype(type(val)):
            return f"{int(val):,}", True
    except Exception:
        pass
    return str(val), False


def _render_kpi_cards(df: pd.DataFrame):
    """Render a single-row result as KPI metric cards."""
    cols = df.columns.tolist()
    card_groups = [cols[i:i+4] for i in range(0, min(len(cols), 16), 4)]
    for grp in card_groups:
        gcols = st.columns(len(grp))
        for ci, col in enumerate(grp):
            raw = df[col].iloc[0]
            val_str, is_num = _fmt_val(raw)
            cls = "kpi-value" if is_num else "kpi-value kpi-text"
            gcols[ci].markdown(
                f"<div class='kpi-card'>"
                f"<div class='kpi-label' title='{col}'>{col}</div>"
                f"<div class='{cls}'>{val_str}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


def _render_chart_view(df: pd.DataFrame):
    """Render a professional Plotly chart — auto-selects type based on data shape."""
    import plotly.graph_objects as go

    # Nielsen brand palette
    _PALETTE = [
        "#0053A5", "#00A3E0", "#34D399", "#F59E0B",
        "#EF4444", "#8B5CF6", "#EC4899", "#14B8A6",
    ]
    _GRAD_LOW, _GRAD_HIGH = "#A8CBFF", "#0053A5"

    text_cols    = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_col      = text_cols[0]
    n_rows       = len(df)

    # Detect time series
    is_time = (
        pd.api.types.is_datetime64_any_dtype(df[cat_col])
        or bool(re.search(r'date|week|month|year|period|quarter', cat_col, re.IGNORECASE))
    )

    # Detect long labels → use horizontal bars
    long_labels = not is_time and (
        n_rows > 10 or any(len(str(v)) > 14 for v in df[cat_col])
    )

    chart_df = df[[cat_col] + numeric_cols].copy()
    if not is_time:
        chart_df = chart_df.sort_values(numeric_cols[0], ascending=False if not long_labels else True)

    # ── Base layout shared by all chart types ─────────────────────────────────
    _common_layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", size=11, color="#0D1B2A"),
        margin=dict(l=4, r=4, t=36, b=4),
        hoverlabel=dict(
            bgcolor="#0053A5",
            font_color="#fff",
            font_size=12,
            bordercolor="#0053A5",
            namelength=-1,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.04,
            xanchor="left",  x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10),
        ),
    )

    def _axis(grid=True, line=False):
        return dict(
            showgrid=grid,
            gridcolor="#EEF2F8",
            gridwidth=1,
            zeroline=False,
            showline=line,
            linecolor="#D0D9E8",
            tickfont=dict(size=10, color="#6B7A8D"),
        )

    fig = go.Figure()

    # ── Chart type selection ──────────────────────────────────────────────────
    if is_time:
        # Line chart with area fill for single metric, multi-line for several
        for i, nc in enumerate(numeric_cols[:5]):
            fill = "tozeroy" if (i == 0 and len(numeric_cols) == 1) else "none"
            fig.add_trace(go.Scatter(
                x=chart_df[cat_col],
                y=chart_df[nc],
                mode="lines+markers",
                name=nc,
                fill=fill,
                fillcolor="rgba(0,83,165,0.08)",
                line=dict(color=_PALETTE[i % len(_PALETTE)], width=2.5, shape="spline"),
                marker=dict(size=6, symbol="circle", line=dict(width=1.5, color="#fff")),
                hovertemplate=f"<b>%{{x}}</b><br>{nc}: <b>%{{y:,.2f}}</b><extra></extra>",
            ))
        fig.update_layout(
            **_common_layout,
            height=310,
            xaxis=_axis(grid=False, line=True),
            yaxis=_axis(grid=True,  line=False),
        )

    elif len(numeric_cols) > 1:
        # Grouped bar — multiple metrics
        for i, nc in enumerate(numeric_cols[:5]):
            fig.add_trace(go.Bar(
                x=chart_df[cat_col],
                y=chart_df[nc],
                name=nc,
                marker_color=_PALETTE[i % len(_PALETTE)],
                marker_line_width=0,
                hovertemplate=f"<b>%{{x}}</b><br>{nc}: <b>%{{y:,.2f}}</b><extra></extra>",
            ))
        fig.update_layout(
            **_common_layout,
            barmode="group",
            bargap=0.22,
            bargroupgap=0.08,
            height=310,
            xaxis=_axis(grid=False, line=True),
            yaxis=_axis(grid=True,  line=False),
        )

    elif long_labels:
        # Horizontal bar — gradient fill, sorted ascending so top value is at top
        vals = chart_df[numeric_cols[0]].tolist()
        labels = chart_df[cat_col].tolist()
        h = max(260, min(520, 42 * n_rows + 40))
        fig.add_trace(go.Bar(
            x=vals,
            y=labels,
            orientation="h",
            marker=dict(
                color=vals,
                colorscale=[[0, _GRAD_LOW], [1, _GRAD_HIGH]],
                showscale=False,
                line=dict(width=0),
            ),
            hovertemplate=f"<b>%{{y}}</b><br>{numeric_cols[0]}: <b>%{{x:,.2f}}</b><extra></extra>",
        ))
        fig.update_layout(
            **_common_layout,
            height=h,
            xaxis=_axis(grid=True,  line=False),
            yaxis=_axis(grid=False, line=True) | {"tickmode": "array", "tickvals": labels},
        )

    else:
        # Vertical gradient bar — single metric, few categories
        vals   = chart_df[numeric_cols[0]].tolist()
        labels = chart_df[cat_col].tolist()
        fig.add_trace(go.Bar(
            x=labels,
            y=vals,
            marker=dict(
                color=vals,
                colorscale=[[0, _GRAD_LOW], [1, _GRAD_HIGH]],
                showscale=False,
                line=dict(width=0),
            ),
            hovertemplate=f"<b>%{{x}}</b><br>{numeric_cols[0]}: <b>%{{y:,.2f}}</b><extra></extra>",
        ))
        fig.update_layout(
            **_common_layout,
            height=290,
            xaxis=_axis(grid=False, line=True),
            yaxis=_axis(grid=True,  line=False),
        )

    st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": [
                "zoom2d", "pan2d", "select2d", "lasso2d",
                "zoomIn2d", "zoomOut2d", "autoScale2d",
            ],
            "toImageButtonOptions": {"format": "png", "filename": "nielsen_chart", "scale": 2},
        },
    )
    st.markdown("</div>", unsafe_allow_html=True)


def _render_summary_strip(df: pd.DataFrame):
    """Render a row of styled stat pills for numeric columns."""
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return

    _ICONS = ["∑", "⌀", "↑", "↓"]
    pills_html = ""
    for col in numeric_cols[:4]:
        total = df[col].sum()
        avg   = df[col].mean()
        mx    = df[col].max()
        mn    = df[col].min()
        t_s, _ = _fmt_val(float(total))
        a_s, _ = _fmt_val(float(avg))
        x_s, _ = _fmt_val(float(mx))
        n_s, _ = _fmt_val(float(mn))
        pills_html += f"""
        <div class="stat-card">
          <div class="stat-col-label">{col}</div>
          <div class="stat-row">
            <div class="stat-item"><span class="stat-icon">∑</span><span class="stat-lbl">Total</span><b>{t_s}</b></div>
            <div class="stat-item"><span class="stat-icon">⌀</span><span class="stat-lbl">Avg</span><b>{a_s}</b></div>
            <div class="stat-item"><span class="stat-icon">↑</span><span class="stat-lbl">Max</span><b>{x_s}</b></div>
            <div class="stat-item"><span class="stat-icon">↓</span><span class="stat-lbl">Min</span><b>{n_s}</b></div>
          </div>
        </div>"""
    st.markdown(
        f"""<style>
        .stat-card{{background:#F8FAFD;border:1px solid #E2E8F2;border-radius:8px;
          padding:.5rem .8rem .45rem;margin:.45rem 0 .15rem;}}
        .stat-col-label{{font-size:.62rem;font-weight:700;text-transform:uppercase;
          letter-spacing:.06em;color:#0053A5;margin-bottom:.3rem;}}
        .stat-row{{display:flex;gap:.6rem;flex-wrap:wrap;}}
        .stat-item{{display:flex;align-items:center;gap:3px;font-size:.72rem;color:#0D1B2A;
          background:#fff;border:1px solid #E2E8F2;border-radius:6px;padding:2px 8px;}}
        .stat-icon{{font-size:.7rem;color:#0053A5;font-weight:700;margin-right:1px;}}
        .stat-lbl{{color:#6B7A8D;margin-right:3px;}}
        .stat-item b{{font-weight:600;color:#0053A5;}}
        </style>{pills_html}""",
        unsafe_allow_html=True,
    )


def _render_results(df: pd.DataFrame, uid: str):
    """Main smart result renderer — picks KPI / chart / table mode."""
    mode = _detect_display_mode(df)

    n_rows = len(df)
    n_cols = len(df.columns)
    badge  = f"{n_rows:,} row{'s' if n_rows != 1 else ''} · {n_cols} col{'s' if n_cols != 1 else ''}"
    st.markdown(
        f"<div class='result-header'>"
        f"<span class='result-header-title'>Query Result</span>"
        f"<span class='result-badge'>{badge}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if mode == "empty":
        st.info("No data returned.")
        return

    if mode == "kpi":
        _render_kpi_cards(df)

    elif mode == "chart":
        _render_chart_view(df)
        with st.expander(f"📋 Full table ({n_rows} rows)", expanded=False):
            st.dataframe(
                df,
                use_container_width=True,
                height=min(340, max(120, (n_rows + 1) * 36 + 12)),
                column_config=_col_config(df),
            )
        _render_summary_strip(df)

    else:  # table
        st.dataframe(
            df,
            use_container_width=True,
            height=min(420, max(120, (n_rows + 1) * 36 + 12)),
            column_config=_col_config(df),
        )
        _render_summary_strip(df)

    # download
    try:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download CSV",
            data=csv_bytes,
            file_name=f"results_{uid}.csv",
            mime="text/csv",
            key=f"dl_csv_{uid}",
            use_container_width=False,
        )
    except Exception:
        pass


def _history_for_agent():
    return [
        {
            "question":             t["question"],
            "nl_response":          t["nl_response"],
            "sql":                  t.get("sql"),
            "result_preview":       t.get("result_preview"),
            "filter_clarification": t.get("filter_clarification"),
        }
        for t in st.session_state.chat_history
        if t.get("nl_response")
    ]


# ─── node step config ──────────────────────────────────────────────────────────
_STEP_CFG = {
    "interaction":     ("🧠", "Understanding intent"),
    "check_cache":     ("⚡", "Checking cache"),
    "planner":         ("📋", "Planning query"),
    "schema_retriever":("🗂️", "Selecting schema"),
    "generator":       ("⚙️", "Generating SQL"),
    "executor":        ("▶️", "Executing SQL"),
    "reflector":       ("🔄", "Self-correcting"),
    "responder":       ("✍️", "Composing answer"),
}

INTENT_LABELS = {
    "data_query":    "📊 Data question",
    "follow_up":     "🔗 Follow-up",
    "correction":    "🔧 Correction",
    "smalltalk":     "💬 Chat",
    "clarification": "❓ Needs clarification",
    "out_of_scope":  "🚫 Out of scope",
}


# ─── turn detail renderer ─────────────────────────────────────────────────────

def _render_details(turn: dict, uid: str, expanded: bool = False):
    intent = turn.get("intent") or "data_query"
    if intent not in DATA_INTENTS and not turn.get("sql") and turn.get("df") is None:
        return

    df = turn.get("df")
    if df is not None and not isinstance(df, pd.DataFrame):
        df = _rows_to_df(df)

    # ── inline result (always visible, no expander needed for data) ───────────
    if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
        _render_results(df, uid)
    elif turn.get("error"):
        st.error(f"**Query error:** {turn['error']}")

    # ── collapsible SQL + trace ───────────────────────────────────────────────
    with st.expander("🗃️ SQL & trace", expanded=False):
        tabs = st.tabs(["🗃️ SQL", "⚙️ Trace"])

        with tabs[0]:
            sql = turn.get("sql") or "—"
            st.code(sql, language="sql")
            if sql != "—":
                st.download_button(
                    "📋 Download SQL",
                    data=sql,
                    file_name=f"query_{uid}.sql",
                    mime="text/plain",
                    key=f"dl_sql_{uid}",
                )

        with tabs[1]:
            exec_t  = f"{turn.get('exec_time', 0):.0f} ms"  if turn.get("exec_time")  else "—"
            total_t = f"{turn.get('total_time', 0):.0f} ms" if turn.get("total_time") else "—"
            cache   = "✅ Hit" if turn.get("cache_hit") else "❌ Miss"
            iters   = str(turn.get("iterations", 0))
            st.markdown(
                f'<div class="metric-strip">'
                f'<div class="metric-pill"><span>SQL time</span>{exec_t}</div>'
                f'<div class="metric-pill"><span>Total</span>{total_t}</div>'
                f'<div class="metric-pill"><span>Corrections</span>{iters}</div>'
                f'<div class="metric-pill"><span>Cache</span>{cache}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if turn.get("plan"):
                st.caption("Logical plan")
                st.markdown(
                    f"<div class='plan-box'>{turn['plan']}</div>",
                    unsafe_allow_html=True,
                )
            if turn.get("few_shot_count", 0) > 0:
                st.caption(f"📚 {turn['few_shot_count']} few-shot example(s) used")


# ─── app bar ───────────────────────────────────────────────────────────────────
try:
    tables   = db_manager.get_all_table_names()
    db_label = tables[0] if tables else "Connected"
    db_ok    = True
except Exception:
    db_label = "Disconnected"
    db_ok    = False

st.markdown(
    f"""<div class="app-bar">
      <span class="app-bar-icon">📊</span>
      <div>
        <div class="app-bar-title">Nielsen Text-to-SQL Agent</div>
        <div class="app-bar-sub">Ask questions in plain English — Databricks · LangGraph DRGC</div>
      </div>
      <div class="app-bar-status">
        <span class="dot" style="background:{'#34D399' if db_ok else '#F87171'};
          box-shadow:0 0 6px {'#34D399' if db_ok else '#F87171'}"></span>
        {db_label}
      </div>
    </div>""",
    unsafe_allow_html=True,
)

# ─── example questions ─────────────────────────────────────────────────────────
EXAMPLES = [
    "MONDELEZ market share in TOTAL BARS 2024",
    "Compare OREO sales Dec 2023 vs Mar 2024",
    "TDP for MDLZ SINGLES PPG in 2024",
    "Sales YTD vs YTD last year",
]
ex_cols = st.columns(len(EXAMPLES))
for i, ex in enumerate(EXAMPLES):
    if ex_cols[i].button(ex, key=f"ex_{i}", use_container_width=True):
        st.session_state["prefill"] = ex
        st.rerun()

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# ─── chat history ──────────────────────────────────────────────────────────────
for idx, turn in enumerate(st.session_state.chat_history):
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn.get("nl_response") or "*(no response)*")
        _render_details({**turn, "idx": idx}, uid=str(idx))

# ─── chat input ────────────────────────────────────────────────────────────────
prefill  = st.session_state.pop("prefill", None)
prompt   = st.chat_input("Ask about Nielsen POS data…", key="chat_input")
question = prompt or prefill

# ─── agent pipeline ────────────────────────────────────────────────────────────
if question:
    with st.chat_message("user"):
        st.markdown(question)

    conversation_history = _history_for_agent()
    final_state: dict    = {}
    detected_intent: str = "data_query"

    with st.chat_message("assistant"):

        # ── live thinking panel ────────────────────────────────────────────────
        steps_log: list[dict] = []      # each: {icon, label, detail, state}
        thinking_placeholder  = st.empty()

        def _render_thinking(active_label: str = ""):
            """Re-render the thinking panel from steps_log."""
            rows_html = ""
            for s in steps_log:
                cls = f"step-{s.get('state','active')}"
                detail_html = (
                    f"<div class='step-detail'>{s['detail']}</div>"
                    if s.get("detail") else ""
                )
                spinner_html = (
                    "<span class='spinner'><span></span><span></span><span></span></span>"
                    if s.get("state") == "active" else ""
                )
                rows_html += (
                    f"<div class='think-step {cls}'>"
                    f"  <span class='step-icon'>{s['icon']}</span>"
                    f"  <div class='step-body'>"
                    f"    <div class='step-label'>{s['label']}{spinner_html}</div>"
                    f"    {detail_html}"
                    f"  </div>"
                    f"</div>"
                )
            thinking_placeholder.markdown(
                f"<div class='think-box'>"
                f"  <div class='think-title'>🤖 Thinking…</div>"
                f"  {rows_html}"
                f"</div>",
                unsafe_allow_html=True,
            )

        plan_placeholder = st.empty()
        sql_placeholder  = st.empty()

        for event in stream_agent_steps(question, conversation_history):
            if event["type"] == "step":
                node  = event["node"]
                out   = event.get("output", {})
                icon, lbl = _STEP_CFG.get(node, ("•", NODE_LABELS.get(node, node)))

                # mark previous step done
                if steps_log and steps_log[-1].get("state") == "active":
                    steps_log[-1]["state"] = "done"

                if node == "interaction":
                    detected_intent = out.get("intent", "data_query")
                    badge = INTENT_LABELS.get(detected_intent, detected_intent)
                    steps_log.append({"icon": icon, "label": lbl, "detail": badge, "state": "active"})
                    _render_thinking()
                    if detected_intent not in DATA_INTENTS:
                        steps_log[-1]["state"] = "done"
                        _render_thinking()
                    continue

                if detected_intent not in DATA_INTENTS:
                    continue

                if node == "check_cache" and out.get("cache_hit"):
                    steps_log.append({"icon": "⚡", "label": "Cache hit — skipping generation", "state": "done"})
                    _render_thinking()
                    continue

                detail = ""
                if node == "planner" and out.get("plan"):
                    detail = "Plan ready"
                elif node == "schema_retriever" and out.get("schema_metadata"):
                    cols = list(out["schema_metadata"].keys())
                    detail = f"Columns: {', '.join(str(c) for c in cols[:6])}" + (
                        f" +{len(cols)-6}" if len(cols) > 6 else ""
                    )
                elif node == "generator" and out.get("sql_query"):
                    detail = "SQL generated"
                elif node == "executor":
                    detail = "⚠️ Error — retrying" if out.get("error") else "Success"
                elif node == "reflector":
                    detail = "Rewriting SQL…"

                steps_log.append({"icon": icon, "label": lbl, "detail": detail, "state": "active"})
                _render_thinking()

                # live previews
                if node == "planner" and out.get("plan"):
                    plan_placeholder.markdown(
                        f"<div class='plan-box'>{out['plan']}</div>",
                        unsafe_allow_html=True,
                    )
                elif node == "generator" and out.get("sql_query"):
                    sql_placeholder.code(out["sql_query"], language="sql")

            elif event["type"] == "final":
                final_state = event["state"]

        # close last active step
        if steps_log and steps_log[-1].get("state") == "active":
            steps_log[-1]["state"] = "done"

        # finalise thinking panel
        if detected_intent in DATA_INTENTS:
            ok = not final_state.get("error")
            steps_log.append({
                "icon":  "✅" if ok else "⚠️",
                "label": "Done!" if ok else "Completed with errors",
                "state": "done" if ok else "warn",
            })
        _render_thinking()

        # clear live previews (they'll appear in details expander)
        plan_placeholder.empty()
        sql_placeholder.empty()

        # ── streaming response ─────────────────────────────────────────────────
        is_data = detected_intent in DATA_INTENTS

        if is_data:
            nl_answer = st.write_stream(stream_nl_response(final_state, conversation_history))
        else:
            nl_answer = (
                final_state.get("nl_response")
                or final_state.get("direct_response")
                or ""
            )
            st.markdown(nl_answer)

        df = _rows_to_df(final_state.get("query_result")) if is_data else None
        _pending_fc = final_state.get("pending_filter_clarification")

        turn = {
            "question":             question,
            "nl_response":          nl_answer,
            "intent":               detected_intent,
            "sql":                  final_state.get("sql_query") if (is_data or _pending_fc) else None,
            "df":                   df,
            "plan":                 final_state.get("plan"),
            "error":                final_state.get("error"),
            "iterations":           final_state.get("iterations", 0),
            "exec_time":            final_state.get("execution_time_ms"),
            "total_time":           final_state.get("total_latency_ms"),
            "cache_hit":            final_state.get("cache_hit", False),
            "few_shot_count":       len(final_state.get("few_shot_examples") or []),
            "filter_clarification": _pending_fc,
            "result_preview":       final_state.get("result_preview"),
        }
        _uid = str(len(st.session_state.chat_history))
        _render_details({**turn, "idx": int(_uid)}, uid=_uid, expanded=True)

    st.session_state.chat_history.append(turn)
    st.session_state.show_correction_input = False

# ─── feedback / correction ────────────────────────────────────────────────────
if st.session_state.chat_history:
    _last = st.session_state.chat_history[-1]
    if _last.get("intent") in DATA_INTENTS and not _last.get("error"):
        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        st.caption("Was this answer helpful?")
        _c1, _c2, _c3 = st.columns([1, 1, 8])
        if _c1.button("👍", key="fb_up", help="Correct"):
            st.session_state.show_correction_input = False
        if _c2.button("👎", key="fb_down", help="Incorrect"):
            st.session_state.show_correction_input = True

        if st.session_state.show_correction_input:
            _correction = st.text_input(
                "What was wrong?",
                key="correction_input",
                placeholder="e.g. wrong year, wrong brand…",
                label_visibility="collapsed",
            )
            if st.button("Send ➤", key="send_correction", type="primary"):
                msg = _correction.strip() or "That answer was incorrect, please try again."
                st.session_state["prefill"] = msg
                st.session_state.show_correction_input = False
                st.rerun()

# ─── new conversation ─────────────────────────────────────────────────────────
if st.session_state.chat_history:
    if st.button("🗑 New conversation", key="new_conv"):
        st.session_state.chat_history = []
        st.session_state.show_correction_input = False
        st.rerun()

# ─── footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='app-footer'>"
    "Nielsen Text-to-SQL &nbsp;·&nbsp; LangGraph DRGC &nbsp;·&nbsp; "
    "Databricks Gemini &nbsp;·&nbsp; ChromaDB"
    "</div>",
    unsafe_allow_html=True,
)
