"""Streamlit chat interface for the Nielsen Text-to-SQL Agent."""

import streamlit as st
import pandas as pd
from pathlib import Path
from loguru import logger

from graph import stream_agent_steps, stream_nl_response, NODE_LABELS
from agents.interaction import DATA_INTENTS
from tools import seed_examples, semantic_cache, few_shot_retriever
from core.database import db_manager
from core.data_loader import DataLoader
from config import settings

# ─── page config ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nielsen Text-to-SQL Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── session state ──────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "query_count" not in st.session_state:
    st.session_state.query_count = 0
if "show_example_form" not in st.session_state:
    st.session_state.show_example_form = False

# ─── helpers ──────────────────────────────────────────────────────────────────────

def _rows_to_df(query_result):
    """Convert query result (list of dicts or SQLAlchemy Rows) to DataFrame."""
    if not query_result or not isinstance(query_result, (list, tuple)):
        return None
    if len(query_result) == 0:
        return None
    try:
        first = query_result[0]
        if isinstance(first, dict):
            return pd.DataFrame(query_result)
        if hasattr(first, "_mapping"):
            return pd.DataFrame([dict(r._mapping) for r in query_result])
        if hasattr(first, "_asdict"):
            return pd.DataFrame([r._asdict() for r in query_result])
        if hasattr(first, "keys"):
            cols = list(first.keys())
            return pd.DataFrame([dict(zip(cols, r)) for r in query_result])
        return pd.DataFrame(query_result)
    except Exception as e:
        logger.warning(f"Could not convert query result to DataFrame: {e}")
        return None


def _history_for_agent():
    """Return minimal history list for the agent (question + nl_response only)."""
    return [
        {"question": t["question"], "nl_response": t["nl_response"]}
        for t in st.session_state.chat_history
        if t.get("nl_response")
    ]


def _render_turn(turn: dict, expanded: bool = False):
    """Render a single assistant turn.

    Data intents  → Data / SQL / Trace tabs.
    Direct intents (smalltalk, clarification, out_of_scope) → no tabs.
    """
    intent = turn.get("intent") or "data_query"
    # Skip the details expander for non-data turns that carry no SQL/data
    if intent not in DATA_INTENTS and not turn.get("sql") and turn.get("df") is None:
        return

    with st.expander("🔍 Details", expanded=expanded):
        tabs = st.tabs(["📊 Data", "🗃 SQL", "⚙️ Trace"])

        with tabs[0]:
            df = turn.get("df")
            if df is not None and not df.empty:
                st.dataframe(df, use_container_width=True)
                st.caption(f"{len(df)} row(s) returned")
            elif turn.get("error"):
                st.error(turn["error"])
            else:
                st.info("No tabular data returned.")

        with tabs[1]:
            sql = turn.get("sql") or "—"
            st.code(sql, language="sql")
            if sql != "—":
                st.download_button(
                    "📋 Download SQL", data=sql,
                    file_name="query.sql", mime="text/plain",
                    key=f"dl_{turn.get('idx', 0)}",
                )

        with tabs[2]:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("SQL exec",   f"{turn.get('exec_time', 0):.0f} ms"  if turn.get("exec_time")  else "—")
            c2.metric("Total time", f"{turn.get('total_time', 0):.0f} ms" if turn.get("total_time") else "—")
            c3.metric("Corrections", turn.get("iterations", 0))
            c4.metric("Cache hit", "✅" if turn.get("cache_hit") else "❌")
            if turn.get("plan"):
                st.markdown("**Logical Plan**")
                st.text(turn["plan"])
            if turn.get("few_shot_count", 0) > 0:
                st.caption(f"📚 {turn['few_shot_count']} few-shot example(s) used")

# ─── sidebar ─────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("Database")
    try:
        tables = db_manager.get_all_table_names()
        st.success(f"✅ Connected — {', '.join(tables)}")
    except Exception as e:
        st.error(f"DB error: {e}")

    st.markdown("---")

    st.subheader("💬 Conversation")
    st.metric("Turns", len(st.session_state.chat_history))
    if st.button("🗑 New Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")

    st.subheader("🛠️ Tools")
    if st.button("🌱 Seed Examples", use_container_width=True):
        with st.spinner("Seeding…"):
            seed_examples()
        st.success("Examples seeded from nielsen_few_shots.yaml!")

    if st.button("🗑 Clear Cache", use_container_width=True):
        semantic_cache.clear()
        st.success("Cache cleared!")

    if st.button("➕ Add Custom Example", use_container_width=True):
        st.session_state.show_example_form = True

    st.markdown("---")

    st.subheader("📂 Upload Data")
    with st.expander("Upload CSV / Excel"):
        uploaded_files = st.file_uploader(
            "Choose files", type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
        )
        if uploaded_files and st.button("🚀 Load as nielsen_pos"):
            db_path = settings.database_uri.replace("sqlite:///", "")
            loader = DataLoader(db_path=db_path)
            import tempfile
            with st.spinner("Loading…"):
                for uf in uploaded_files:
                    try:
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=Path(uf.name).suffix
                        ) as tmp:
                            tmp.write(uf.read())
                            stats = loader.load_file(
                                tmp.name, table_name="nielsen_pos",
                                if_exists="replace",
                            )
                            st.success(f"✓ {stats['rows']} rows loaded as nielsen_pos")
                            Path(tmp.name).unlink()
                    except Exception as e:
                        st.error(f"Error: {e}")
            db_manager.__class__.__init__(db_manager)
            st.rerun()

# ─── main header ────────────────────────────────────────────────────────────────────
st.title("📊 Nielsen Text-to-SQL Agent")
st.caption(
    "Multi-agent DRGC pipeline · Interaction routing · "
    "CTE SQL · Dynamic few-shot · Streaming responses"
)

# ─── example question pills ────────────────────────────────────────────────────────────
EXAMPLES = [
    "What was MONDELEZ market share in TOTAL BARS in 2024?",
    "Compare OREO brand sales in December 2023 vs March 2024",
    "What is the TDP for MDLZ SINGLES PPG in 2024?",
    "How did sales perform YTD vs YTD last year?",
]

st.markdown("**Try an example:**")
ex_cols = st.columns(len(EXAMPLES))
for i, ex in enumerate(EXAMPLES):
    if ex_cols[i].button(ex, key=f"ex_{i}", use_container_width=True):
        st.session_state["prefill"] = ex
        st.rerun()

st.markdown("---")

# ─── render chat history ────────────────────────────────────────────────────────────
for idx, turn in enumerate(st.session_state.chat_history):
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn.get("nl_response", "*(no response)*"))
        _render_turn({**turn, "idx": idx})

# ─── intent badge map ────────────────────────────────────────────────────────────────
INTENT_BADGES = {
    "data_query":    "🔍 Data question",
    "follow_up":     "🔗 Follow-up question",
    "smalltalk":     "💬 Conversation",
    "clarification": "❓ Needs clarification",
    "out_of_scope":  "🚫 Out of scope",
}

# ─── chat input ──────────────────────────────────────────────────────────────────
prefill  = st.session_state.pop("prefill", None)
prompt   = st.chat_input("Ask a question about Nielsen POS data…", key="chat_input")
question = prompt or prefill

if question:
    # 1. Show user message immediately
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.query_count += 1
    conversation_history = _history_for_agent()
    final_state: dict   = {}
    detected_intent: str = "data_query"   # will be updated when interaction step fires

    # 2. Step-by-step agent thinking
    with st.chat_message("assistant"):
        with st.status("🤖 Agent is thinking…", expanded=True) as status:
            step_placeholder = st.empty()

            for event in stream_agent_steps(question, conversation_history):
                if event["type"] == "step":
                    node  = event["node"]
                    label = event["label"]
                    out   = event.get("output", {})

                    # ---- Interaction node: classify intent ----
                    if node == "interaction":
                        detected_intent = out.get("intent", "data_query")
                        badge = INTENT_BADGES.get(detected_intent, detected_intent)
                        step_placeholder.markdown(f"**{label}** — {badge}")
                        if detected_intent not in DATA_INTENTS:
                            # Non-data path is very fast — collapse immediately
                            status.update(label=badge, state="complete", expanded=False)
                        continue

                    # ---- For non-data intents, skip showing SQL steps ----
                    if detected_intent not in DATA_INTENTS:
                        continue

                    # ---- Data path: show progress ----
                    if node == "check_cache" and out.get("cache_hit"):
                        step_placeholder.success("⚡ Cache hit — returning cached result")
                    else:
                        step_placeholder.markdown(f"**{label}**")

                    if node == "planner" and out.get("plan"):
                        with st.expander("📋 Query Plan", expanded=False):
                            st.text(out["plan"])
                    elif node == "schema_retriever" and out.get("schema_metadata"):
                        cols = list(out["schema_metadata"].keys())
                        st.caption(f"Selected columns: {cols}")
                    elif node == "generator" and out.get("sql_query"):
                        with st.expander("🗃 Generated SQL", expanded=False):
                            st.code(out["sql_query"], language="sql")
                    elif node == "executor":
                        if out.get("error"):
                            st.warning(f"Execution error — will retry: `{out['error']}`")
                        else:
                            st.success("✅ SQL executed successfully")
                    elif node == "reflector":
                        st.warning("🔄 Correcting SQL…")

                elif event["type"] == "final":
                    final_state = event["state"]

            # Update status label for data path
            if detected_intent in DATA_INTENTS:
                if final_state.get("error"):
                    status.update(label="⚠️ Completed with errors", state="error", expanded=False)
                else:
                    status.update(label="✅ Done!", state="complete", expanded=False)

        # 3. Response rendering
        is_data = detected_intent in DATA_INTENTS

        if is_data:
            # Stream the NL answer token-by-token for data results
            nl_answer = st.write_stream(
                stream_nl_response(final_state, conversation_history)
            )
        else:
            # Direct responses are already computed by the interaction/direct_respond nodes
            nl_answer = (
                final_state.get("nl_response")
                or final_state.get("direct_response")
                or ""
            )
            st.markdown(nl_answer)

        # 4. Details tabs (shown only for data intents)
        df = _rows_to_df(final_state.get("query_result")) if is_data else None
        turn = {
            "question":       question,
            "nl_response":    nl_answer,
            "intent":         detected_intent,
            "sql":            final_state.get("sql_query") if is_data else None,
            "df":             df,
            "plan":           final_state.get("plan"),
            "error":          final_state.get("error"),
            "iterations":     final_state.get("iterations", 0),
            "exec_time":      final_state.get("execution_time_ms"),
            "total_time":     final_state.get("total_latency_ms"),
            "cache_hit":      final_state.get("cache_hit", False),
            "few_shot_count": len(final_state.get("few_shot_examples") or []),
        }
        _render_turn({**turn, "idx": len(st.session_state.chat_history)}, expanded=is_data)

    # 5. Persist to history
    st.session_state.chat_history.append(turn)

# ─── custom example form ───────────────────────────────────────────────────────────────────
if st.session_state.show_example_form:
    st.markdown("---")
    st.subheader("➕ Add Custom Few-Shot Example")
    with st.form("example_form"):
        ex_q   = st.text_input("Natural language question")
        ex_sql = st.text_area("SQL (CTE preferred)", height=150)
        c1, c2 = st.columns([1, 4])
        submitted = c1.form_submit_button("Add")
        cancelled = c2.form_submit_button("Cancel")

        if submitted and ex_q and ex_sql:
            few_shot_retriever.add_example(
                question=ex_q, sql=ex_sql, complexity="domain-specific"
            )
            st.success("Example added to vector store!")
            st.session_state.show_example_form = False
            st.rerun()
        if cancelled:
            st.session_state.show_example_form = False
            st.rerun()

# ─── footer ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Nielsen Text-to-SQL · LangGraph DRGC + Interaction Router "
    "· Groq LLM · ChromaDB few-shot · SQLite"
)
