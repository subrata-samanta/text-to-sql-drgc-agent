"""Streamlit chat interface for the Nielsen Text-to-SQL Agent."""

import streamlit as st
import pandas as pd
from pathlib import Path
from loguru import logger

from graph import stream_agent_steps, stream_nl_response, NODE_LABELS
from agents.interaction import DATA_INTENTS
from tools import seed_examples, semantic_cache, few_shot_retriever
from core.database import get_db_manager, invalidate_db_cache
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
if "show_correction_input" not in st.session_state:
    st.session_state.show_correction_input = False
if "last_provider" not in st.session_state:
    st.session_state.last_provider = settings.llm_provider


# ── Apply provider switch mid-session ─────────────────────────────────────────
def _apply_provider_switch(new_provider: str):
    """
    Called when the user changes the LLM provider in the sidebar.
    Updates settings, invalidates the DB cache, and re-initialises the
    few-shot retriever with the right embedding model.
    """
    import importlib
    settings.llm_provider = new_provider
    invalidate_db_cache()           # force a fresh DB manager for new provider

    # Re-initialise the vector store with the right embedding model.
    # The FewShotRetriever picks up settings.llm_provider at __init__ time.
    from tools.vector_store import FewShotRetriever
    import tools.vector_store as _vs_mod
    _vs_mod.few_shot_retriever = FewShotRetriever()

    st.session_state.last_provider = new_provider
    logger.info(f"Provider switched to: {new_provider}")

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
    """Return minimal history list for the agent (question + nl_response + sql)."""
    return [
        {
            "question":              t["question"],
            "nl_response":           t["nl_response"],
            "sql":                   t.get("sql"),
            # Result table text so interaction_node can detect result_lookup intent.
            "result_preview":        t.get("result_preview"),
            # Carry filter clarification context so interaction_node can
            # auto-route the user's answer to the correction pipeline.
            "filter_clarification":  t.get("filter_clarification"),
        }
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

    # ── LLM Provider selector ────────────────────────────────────────────────
    st.subheader("🤖 LLM Provider")
    provider_choice = st.radio(
        "Select provider",
        options=["groq", "dbrx"],
        index=0 if settings.llm_provider == "groq" else 1,
        format_func=lambda p: "🟢 Groq  (local DB + HuggingFace embeddings)"
        if p == "groq"
        else "🔵 DBRX  (Databricks Delta + Databricks embeddings)",
        horizontal=False,
        key="provider_radio",
    )
    if provider_choice != st.session_state.last_provider:
        _apply_provider_switch(provider_choice)
        st.success(f"Switched to **{provider_choice.upper()}** provider!")
        st.rerun()

    if settings.is_dbrx:
        if not settings.databricks_token:
            st.warning("⚠️ DATABRICKS_TOKEN not set in .env")
        if not settings.databricks_host:
            st.warning("⚠️ DATABRICKS_HOST not set in .env")
        if not settings.databricks_http_path:
            st.warning("⚠️ DATABRICKS_HTTP_PATH not set in .env")

    st.markdown("---")

    # ── Database status ─────────────────────────────────────────────────────
    st.subheader("Database")
    try:
        _active_db = get_db_manager()
        tables = _active_db.get_all_table_names()
        if settings.is_dbrx:
            st.success(
                f"✅ Databricks — `{settings.databricks_catalog}.{settings.databricks_schema}`  "
                f"({', '.join(tables) or 'no tables'})"
            )
        else:
            st.success(f"✅ SQLite — {', '.join(tables)}")
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

    _cache_enabled = st.toggle(
        "⚡ Semantic cache",
        value=settings.enable_semantic_cache,
        help="Cache SQL results for similar questions to speed up repeated queries.",
    )
    if _cache_enabled != settings.enable_semantic_cache:
        settings.enable_semantic_cache = _cache_enabled
        if _cache_enabled:
            semantic_cache._init_backend()   # warm up lazily
            st.success("Cache enabled!")
        else:
            st.info("Cache disabled for this session.")

    if st.button("🗑 Clear Cache", use_container_width=True, disabled=not settings.enable_semantic_cache):
        semantic_cache.clear()
        st.success("Cache cleared!")

    if st.button("➕ Add Custom Example", use_container_width=True):
        st.session_state.show_example_form = True

    st.markdown("---")

    if not settings.is_dbrx:
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
                invalidate_db_cache()
                st.rerun()
    else:
        st.info("📡 Using Databricks Delta tables — no local upload needed.")

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
    "correction":    "🔧 Correcting answer",
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
        # For filter-clarification turns the SQL was generated but not yet
        # executed — store it so the next turn's interaction_node can patch it.
        _pending_fc = final_state.get("pending_filter_clarification")
        turn = {
            "question":             question,
            "nl_response":          nl_answer,
            "intent":               detected_intent,
            "sql":                  (
                final_state.get("sql_query")   # draft SQL even on clarification turns
                if (is_data or _pending_fc)
                else None
            ),
            "df":                   df,
            "plan":                 final_state.get("plan"),
            "error":                final_state.get("error"),
            "iterations":           final_state.get("iterations", 0),
            "exec_time":            final_state.get("execution_time_ms"),
            "total_time":           final_state.get("total_latency_ms"),
            "cache_hit":            final_state.get("cache_hit", False),
            "few_shot_count":       len(final_state.get("few_shot_examples") or []),
            # Carry clarification context forward for the next turn
            "filter_clarification": _pending_fc,
            # Carry result table text so next turn can answer from it
            "result_preview":        final_state.get("result_preview"),
        }
        _render_turn({**turn, "idx": len(st.session_state.chat_history)}, expanded=is_data)

    # 5. Persist to history
    st.session_state.chat_history.append(turn)
    # Reset correction UI after every new turn
    st.session_state.show_correction_input = False

# ─── feedback widget (persists between reruns for last data turn) ─────────────
if st.session_state.chat_history:
    _last = st.session_state.chat_history[-1]
    if _last.get("intent") in DATA_INTENTS and not _last.get("error"):
        st.markdown("---")
        _fc1, _fc2, _fc3 = st.columns([1, 1, 8])
        if _fc1.button("👍", key="fb_up", help="Answer looks correct"):
            st.session_state.show_correction_input = False
        if _fc2.button("👎", key="fb_down", help="Answer is wrong — give feedback"):
            st.session_state.show_correction_input = True

        if st.session_state.show_correction_input:
            with st.container():
                _corr_text = st.text_input(
                    "What was wrong? (optional)",
                    key="correction_input",
                    placeholder="e.g. wrong year, wrong brand, try recalculating…",
                )
                if st.button("Send correction ➤", key="send_correction"):
                    _msg = (
                        _corr_text.strip()
                        if _corr_text.strip()
                        else "That answer was incorrect, please try again."
                    )
                    st.session_state["prefill"] = _msg
                    st.session_state.show_correction_input = False
                    st.rerun()

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
_provider_label = (
    "Databricks DBRX · Delta tables · Databricks embeddings"
    if settings.is_dbrx
    else "Groq LLM · SQLite · HuggingFace embeddings"
)
st.caption(
    f"Nielsen Text-to-SQL · LangGraph DRGC + Interaction Router "
    f"· {_provider_label} · ChromaDB few-shot"
)
