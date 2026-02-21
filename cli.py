#!/usr/bin/env python
"""
Nielsen Text-to-SQL Agent — CLI interface.

Usage
-----
Interactive conversation (multi-turn):
    python cli.py

Single question:
    python cli.py "What was MONDELEZ market share in TOTAL BARS in 2024?"

Options:
    --no-stream     Disable token streaming (print full answer at once)
    --no-steps      Hide step-by-step agent thinking
"""

import sys
import argparse
import textwrap
import time
from typing import List, Dict, Optional

import pandas as pd

# ── optional Rich import (graceful fallback to plain print) ──────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.syntax import Syntax
    from rich.rule import Rule
    from rich.text import Text
    from rich import box
    RICH = True
except ImportError:
    RICH = False

# ── agent imports ─────────────────────────────────────────────────────────────
from graph import stream_agent_steps, stream_nl_response, NODE_LABELS

console = Console() if RICH else None


# ─── print helpers ────────────────────────────────────────────────────────────

def _hr(title: str = ""):
    if RICH:
        console.print(Rule(title, style="dim"))
    else:
        print(f"\n{'─'*70}" + (f" {title}" if title else ""))


def _step(label: str, detail: str = ""):
    if RICH:
        txt = Text()
        txt.append("  ▸ ", style="bold cyan")
        txt.append(label, style="cyan")
        if detail:
            txt.append(f"  {detail}", style="dim")
        console.print(txt)
    else:
        print(f"  ▸ {label}" + (f"  {detail}" if detail else ""))


def _warn(msg: str):
    if RICH:
        console.print(f"  ⚠  {msg}", style="yellow")
    else:
        print(f"  ⚠  {msg}")


def _success(msg: str):
    if RICH:
        console.print(f"  ✅ {msg}", style="green")
    else:
        print(f"  ✅ {msg}")


def _error(msg: str):
    if RICH:
        console.print(f"  ✗  {msg}", style="bold red")
    else:
        print(f"  ✗  {msg}")


def _print_sql(sql: str):
    if RICH:
        syntax = Syntax(sql, "sql", theme="monokai", line_numbers=False,
                        word_wrap=True)
        console.print(Panel(syntax, title="Generated SQL", border_style="blue"))
    else:
        print("\n── SQL ──")
        print(sql)
        print("────────")


def _rows_to_df_cli(query_result):
    """Convert query result to DataFrame."""
    import pandas as pd
    if not query_result or not isinstance(query_result, (list, tuple)) or len(query_result) == 0:
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
    except Exception:
        return None


def _print_df(query_result):
    """Convert query result → DataFrame and print as table."""
    df = _rows_to_df_cli(query_result)
    if df is None:
        return

    if df.empty:
        print("  (no rows)")
        return

    if RICH:
        tbl = Table(box=box.SIMPLE_HEAD, show_lines=False)
        for col in df.columns:
            tbl.add_column(str(col), style="cyan", no_wrap=False)
        for _, row in df.iterrows():
            tbl.add_row(*[str(v) for v in row])
        console.print(tbl)
        console.print(f"  [dim]{len(df)} row(s) returned[/dim]")
    else:
        print(df.to_string(index=False))
        print(f"\n{len(df)} row(s) returned")


def _print_plan(plan: str):
    if RICH:
        console.print(Panel(
            plan.strip(), title="📋 Query Plan",
            border_style="dim", expand=False
        ))
    else:
        print("\n── Query Plan ──")
        for line in plan.strip().splitlines():
            print(f"  {line}")


def _metrics(final_state: dict):
    exec_ms  = final_state.get("execution_time_ms")
    total_ms = final_state.get("total_latency_ms")
    iters    = final_state.get("iterations", 0)
    cache    = final_state.get("cache_hit", False)
    parts = []
    if exec_ms:  parts.append(f"SQL exec: {exec_ms:.0f} ms")
    if total_ms: parts.append(f"Total: {total_ms:.0f} ms")
    if iters:    parts.append(f"Corrections: {iters}")
    if cache:    parts.append("⚡ cache hit")
    if parts:
        if RICH:
            console.print("  " + "  ·  ".join(parts), style="dim")
        else:
            print("  " + "  |  ".join(parts))


# ─── core runner ──────────────────────────────────────────────────────────────

def run_question(
    question: str,
    conversation_history: Optional[List[Dict]] = None,
    show_steps: bool = True,
    stream: bool = True,
) -> Dict:
    """
    Run a single question through the agent.

    Returns the final state dict (includes nl_response, sql_query, etc.).
    """
    _hr()
    if RICH:
        console.print(f"\n[bold]You:[/bold] {question}\n")
    else:
        print(f"\nYou: {question}\n")

    final_state: dict = {}

    # ── Phase 1: step-by-step agent execution ─────────────────────────────
    if show_steps:
        if RICH:
            console.print("[bold cyan]Agent thinking:[/bold cyan]")
        else:
            print("Agent thinking:")

    for event in stream_agent_steps(question, conversation_history or []):
        if event["type"] == "step":
            node   = event["node"]
            label  = event["label"]
            output = event.get("output", {}) or {}

            if not show_steps:
                continue  # still iterate to collect final state below

            # Interaction node: classify & print intent badge
            if node == "interaction":
                intent = output.get("intent", "data_query")
                badge  = {
                    "data_query":    "[Data question]",
                    "follow_up":     "[Follow-up question]",
                    "smalltalk":     "[Conversation]",
                    "clarification": "[Needs clarification]",
                    "out_of_scope":  "[Out of scope]",
                }.get(intent, intent)
                _step(label, badge)
                continue

            # For non-data intents, skip SQL-specific steps
            intent_now = output.get("intent") or ""
            # (intent is stored at final_state level; check accumulated from event)
            # We skip if the node is SQL-specific and intent was non-data
            # (final_state may not be set yet, so we rely on the graph routing)

            # Cache hit short-circuits
            if node == "check_cache" and output.get("cache_hit"):
                _success("Cache hit — returning cached result")
                continue

            _step(label)

            # Per-node extras
            if node == "planner" and output.get("plan") and show_steps:
                _print_plan(output["plan"])
            elif node == "schema_retriever" and output.get("schema_metadata"):
                cols = list(output["schema_metadata"].keys())
                _step("Columns selected", ", ".join(cols))
            elif node == "generator" and output.get("sql_query"):
                _print_sql(output["sql_query"])
            elif node == "executor":
                if output.get("error"):
                    _warn(f"Execution error (will retry): {output['error']}")
                else:
                    _success("SQL executed successfully")
            elif node == "reflector":
                _warn("Correcting SQL…")

        elif event["type"] == "final":
            final_state = event["state"]

    # ── Phase 2: stream / print NL answer ─────────────────────────────────
    _hr("Answer")

    if RICH:
        console.print("[bold]Agent:[/bold] ", end="")
    else:
        print("Agent: ", end="", flush=True)

    is_data = final_state.get("intent", "data_query") in ("data_query", "follow_up")

    nl_text = ""
    if stream:
        for token in stream_nl_response(final_state, conversation_history):
            print(token, end="", flush=True)
            nl_text += token
        print()  # newline at end
    else:
        for token in stream_nl_response(final_state, conversation_history):
            nl_text += token
        print(nl_text)

    # ── Phase 3: data table ────────────────────────────────────────────────
    if is_data and final_state.get("query_result") and not final_state.get("error"):
        _hr("Data")
        _print_df(final_state.get("query_result"))

    # ── metrics ───────────────────────────────────────────────────────────
    _metrics(final_state)
    print()

    final_state["nl_response"] = nl_text
    return final_state


# ─── interactive conversation loop ────────────────────────────────────────────

def interactive_loop(show_steps: bool = True, stream: bool = True):
    """Multi-turn conversation loop with history."""
    if RICH:
        console.print(Panel(
            "[bold]Nielsen Text-to-SQL Agent[/bold]\n"
            "Type your question and press Enter.\n"
            "Commands: [cyan]exit[/cyan] · [cyan]quit[/cyan] · [cyan]clear[/cyan] (reset history) · [cyan]history[/cyan]",
            border_style="green",
        ))
    else:
        print("=" * 70)
        print("  Nielsen Text-to-SQL Agent")
        print("  Commands: exit | quit | clear | history")
        print("=" * 70)

    conversation_history: List[Dict] = []

    while True:
        try:
            if RICH:
                question = console.input("\n[bold green]You >[/bold green] ").strip()
            else:
                question = input("\nYou > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not question:
            continue

        cmd = question.lower()
        if cmd in ("exit", "quit"):
            print("Bye!")
            break
        elif cmd == "clear":
            conversation_history = []
            _success("Conversation history cleared.")
            continue
        elif cmd == "history":
            if not conversation_history:
                print("  No history yet.")
            else:
                for i, t in enumerate(conversation_history, 1):
                    print(f"  [{i}] Q: {t['question']}")
                    wrapped = textwrap.shorten(t.get("nl_response", ""), 80)
                    print(f"      A: {wrapped}")
            continue

        try:
            final_state = run_question(
                question,
                conversation_history=conversation_history,
                show_steps=show_steps,
                stream=stream,
            )
            # Append to history for follow-up context
            if final_state.get("nl_response"):
                conversation_history.append({
                    "question":    question,
                    "nl_response": final_state["nl_response"],
                })
                # Keep last 10 turns in memory
                conversation_history = conversation_history[-10:]

        except Exception as e:
            _error(f"Unexpected error: {e}")


# ─── entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Nielsen Text-to-SQL Agent — CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          # Interactive conversation
          python cli.py

          # Single question
          python cli.py "What was MONDELEZ market share in TOTAL BARS in 2024?"

          # Single question, no streaming, no step details
          python cli.py "Compare OREO sales Dec 2023 vs Mar 2024" --no-stream --no-steps
        """),
    )
    parser.add_argument(
        "question", nargs="?", default=None,
        help="Question to ask (omit for interactive mode)"
    )
    parser.add_argument(
        "--no-stream", action="store_true",
        help="Print full answer at once instead of streaming tokens"
    )
    parser.add_argument(
        "--no-steps", action="store_true",
        help="Hide step-by-step agent thinking"
    )
    args = parser.parse_args()

    do_stream = not args.no_stream
    do_steps  = not args.no_steps

    if args.question:
        # Single-question mode
        run_question(args.question, show_steps=do_steps, stream=do_stream)
    else:
        # Interactive loop
        interactive_loop(show_steps=do_steps, stream=do_stream)


if __name__ == "__main__":
    main()
