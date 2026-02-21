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
    --cache         Enable semantic cache (off by default)
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
    from rich.padding import Padding
    from rich import box
    RICH = True
except ImportError:
    RICH = False

# ── agent imports ─────────────────────────────────────────────────────────────
from graph import stream_agent_steps, stream_nl_response, NODE_LABELS
from agents.interaction import DATA_INTENTS

console = Console() if RICH else None

_STEP_W = 6   # width of "Step N" prefix column

# ─── low-level print helpers ──────────────────────────────────────────────────

def _rule(title: str = "", style: str = "dim"):
    if RICH:
        console.print(Rule(title, style=style))
    else:
        pad = f"{'─'*3} {title} " if title else ""
        print(f"\n{pad}{'─' * max(0, 70 - len(pad))}")


def _blank():
    if RICH:
        console.print()
    else:
        print()


def _numbered_step(n: int, label: str, detail: str = "", status: str = ""):
    """Print one pipeline step in the format:

    Rich:
        Step 1  │  🧠  Understanding your message       [Data question]  ✅
    Plain:
        Step 1  |  Understanding your message            [Data question]
    """
    prefix = f"Step {n}"
    if RICH:
        txt = Text()
        txt.append(f"  {prefix:<{_STEP_W}}", style="bold white")
        txt.append("  │  ", style="dim white")
        txt.append(f"{label}", style="bold cyan")
        if detail:
            txt.append(f"   {detail}", style="yellow")
        if status:
            txt.append(f"  {status}", style="green")
        console.print(txt)
    else:
        line = f"  {prefix:<{_STEP_W}}  |  {label}"
        if detail:
            line += f"   {detail}"
        if status:
            line += f"  {status}"
        print(line)


def _sub(lines: List[str], style: str = "dim"):
    """Print indented continuation lines below a numbered step."""
    indent = " " * (_STEP_W + 9)   # align past "  Step N  │  "
    for line in lines:
        if RICH:
            console.print(f"{indent}[{style}]{line}[/{style}]")
        else:
            print(f"{indent}{line}")


def _warn(msg: str):
    if RICH:
        console.print(f"  {'':>{_STEP_W + 5}}⚠  {msg}", style="yellow")
    else:
        print(f"  WARNING: {msg}")


def _success(msg: str):
    if RICH:
        console.print(f"  {'':>{_STEP_W + 5}}✅  {msg}", style="green")
    else:
        print(f"  OK: {msg}")


def _error(msg: str):
    if RICH:
        console.print(f"  ✗  {msg}", style="bold red")
    else:
        print(f"  ERROR: {msg}")


def _print_sql(sql: str):
    indent = " " * (_STEP_W + 9)
    if RICH:
        syntax = Syntax(
            sql, "sql", theme="monokai",
            line_numbers=True, word_wrap=True,
            indent_guides=False,
        )
        console.print(Padding(
            Panel(syntax, title="Generated SQL", border_style="blue",
                  expand=False, padding=(0, 1)),
            pad=(0, 0, 0, len(indent)),
        ))
    else:
        print(f"{indent}┌─ SQL {'─'*48}")
        for line in sql.splitlines():
            print(f"{indent}│  {line}")
        print(f"{indent}└{'─'*54}")


def _print_plan(plan: str):
    indent = " " * (_STEP_W + 9)
    lines = plan.strip().splitlines()
    if RICH:
        console.print(Padding(
            Panel("\n".join(lines), title="Query Plan",
                  border_style="dim", expand=False, padding=(0, 1)),
            pad=(0, 0, 0, len(indent)),
        ))
    else:
        print(f"{indent}┌─ Query Plan {'─'*40}")
        for line in lines:
            print(f"{indent}│  {line}")
        print(f"{indent}└{'─'*54}")


def _rows_to_df_cli(query_result):
    """Convert query result to DataFrame."""
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
        tbl = Table(box=box.SIMPLE_HEAD, show_lines=False, padding=(0, 1))
        for col in df.columns:
            tbl.add_column(str(col), style="cyan", no_wrap=False)
        for _, row in df.iterrows():
            tbl.add_row(*[str(v) for v in row])
        console.print(Padding(tbl, pad=(0, 0, 0, 4)))
        console.print(f"    [dim]{len(df)} row(s) returned[/dim]")
    else:
        print(df.to_string(index=False))
        print(f"\n  {len(df)} row(s) returned")


def _metrics_line(final_state: dict):
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
        line = "  ·  ".join(parts)
        if RICH:
            console.print(f"\n  [dim]{line}[/dim]")
        else:
            print(f"\n  {line}")


def _print_step_box(
    n: int,
    label: str,
    badge: str = "",
    rows: Optional[List[tuple]] = None,
    status_style: str = "dim",
):
    """Print one pipeline step as a tidy panel.

    Rich example:
    ╭─ Step 1 ─ 🧠 Understanding your message ─────────────────────────────────╮
    │  Intent   data_query                                                      │
    │  Routing  SQL pipeline                                                    │
    ╰────────────────────────────────────────────────────────────────────────── ╯
    """
    rows = rows or []
    title_parts = [f"[bold white]Step {n}[/bold white]  [bold cyan]{label}[/bold cyan]"]
    if badge:
        title_parts.append(f"[{status_style}]{badge}[/{status_style}]")
    title = "  ".join(title_parts)

    if RICH:
        body_lines = []
        for key, val in rows:
            if key.startswith(" "):        # indented list item
                body_lines.append(f"  [dim]{key}[/dim] {val}")
            else:
                body_lines.append(f"  [bold]{key:<12}[/bold] {val}")
        body = "\n".join(body_lines) if body_lines else ""
        console.print(Panel(
            body,
            title=title,
            title_align="left",
            border_style="bright_black",
            padding=(0, 1),
            expand=False,
        ))
        console.print()
    else:
        # plain text fallback
        print(f"\n  Step {n}  |  {label}" + (f"  {badge}" if badge else ""))
        for key, val in rows:
            print(f"    {key:<12} {val}")
        print()


# INTENT badge map (plain text — no markdown)
_INTENT_BADGES = {
    "data_query":    "[Data question]",
    "follow_up":     "[Follow-up]",
    "correction":    "[Correcting]",
    "smalltalk":     "[Conversation]",
    "clarification": "[Clarification]",
    "out_of_scope":  "[Out of scope]",
}


# ─── core runner ──────────────────────────────────────────────────────────────

def run_question(
    question: str,
    conversation_history: Optional[List[Dict]] = None,
    show_steps: bool = True,
    stream: bool = True,
) -> Dict:
    """Run a single question through the agent and print formatted output."""
    from config import settings as _settings

    # ── Question banner ───────────────────────────────────────────────────
    if RICH:
        console.print()
        console.print(Rule(style="bright_black"))
        console.print(Panel(
            f"[bold white]{question}[/bold white]",
            title="[bold green]❓ Your Question[/bold green]",
            border_style="green",
            padding=(0, 2),
        ))
    else:
        print("\n" + "═" * 72)
        print(f"  ❓ {question}")
        print("═" * 72)

    final_state: dict   = {}
    step_num            = 0
    detected_intent     = "data_query"

    # ── Phase 1: pipeline steps ───────────────────────────────────────────
    if show_steps:
        if RICH:
            console.print()
            console.print(Rule("[bold cyan]🤖  Agent Pipeline[/bold cyan]", style="cyan"))
            console.print()
        else:
            print("\n" + "─" * 72)
            print("  🤖  Agent Pipeline")
            print("─" * 72)

    for event in stream_agent_steps(question, conversation_history or []):
        if event["type"] == "step":
            node   = event["node"]
            label  = event["label"]
            output = event.get("output", {}) or {}

            # ── skip infra nodes that have no user-visible output ──
            if node in ("init", "cache_result"):
                continue

            # ── interaction ──
            if node == "interaction":
                detected_intent = output.get("intent", "data_query")
                badge = _INTENT_BADGES.get(detected_intent, detected_intent)
                if show_steps:
                    step_num += 1
                    _print_step_box(step_num, label, badge, rows=[
                        ("Intent",   detected_intent),
                        ("Routing",  "SQL pipeline" if detected_intent in DATA_INTENTS else "Direct response"),
                    ])
                continue

            # ── for non-data intents nothing SQL-specific is relevant ──
            if detected_intent not in DATA_INTENTS:
                continue

            if not show_steps:
                continue

            step_num += 1

            # ── check_cache ──
            if node == "check_cache":
                if not _settings.enable_semantic_cache:
                    _print_step_box(step_num, label, "DISABLED",
                                    rows=[("Status", "Cache is off — proceeding to full pipeline")],
                                    status_style="dim")
                elif output.get("cache_hit"):
                    _print_step_box(step_num, label, "⚡ HIT",
                                    rows=[("Result", "Cached answer found — skipping generation")],
                                    status_style="yellow")
                else:
                    _print_step_box(step_num, label, "MISS",
                                    rows=[("Result", "No cache match — proceeding to full pipeline")],
                                    status_style="dim")
                continue

            # ── planner ──
            if node == "planner":
                rows = []
                plan = output.get("plan", "")
                if plan:
                    for i, line in enumerate(plan.strip().splitlines(), 1):
                        rows.append((f"  {i}.", line.strip()))
                _print_step_box(step_num, label, "", rows=rows)
                continue

            # ── retrieve_few_shot ──
            if node == "retrieve_few_shot":
                examples = output.get("few_shot_examples") or []
                rows = [(f"  {i+1}.", ex.get("question", "—")[:80])
                        for i, ex in enumerate(examples)]
                _print_step_box(step_num, label, f"{len(examples)} example(s)",
                                rows=rows or [("Result", "No examples found")])
                continue

            # ── schema_retriever ──
            if node == "schema_retriever":
                meta = output.get("schema_metadata") or {}
                cols = list(meta.keys())
                rows = [(f"  {i+1}.", c) for i, c in enumerate(cols)]
                _print_step_box(step_num, label, f"{len(cols)} column(s) selected",
                                rows=rows or [("Result", "No columns selected")])
                continue

            # ── filter_resolver ──
            if node == "filter_resolver":
                log = output.get("filter_log") or []
                needs_clarify = output.get("needs_clarification", False)
                corrected = [e for e in log if e.get("action") == "corrected"]
                unchanged = [e for e in log if e.get("action") == "unchanged"]
                n_clarify  = sum(1 for e in log if e.get("action") == "clarify")
                rows = []
                for e in corrected:
                    rows.append((f"  ✏", f"{e['column']}: '{e['sql_value']}' → '{e['db_match']}' ({e['confidence']:.0%})"))
                for e in unchanged:
                    rows.append((f"  ✓", f"{e['column']}: '{e['sql_value']}' (exact match)"))
                if n_clarify:
                    rows.append(("  ⚠", f"{n_clarify} filter(s) need clarification"))
                if not rows:
                    rows = [("Status", "No string filters found to verify")]
                badge = "⚠ NEEDS CLARIFICATION" if needs_clarify else (
                    f"✅ {len(corrected)} corrected" if corrected else "✅ all matched"
                )
                _print_step_box(step_num, label, badge,
                                rows=rows,
                                status_style="yellow" if needs_clarify else "green")
                continue

            # ── generator ──
            if node == "generator":
                sql = output.get("sql_query", "")
                _print_step_box(step_num, label, "",
                                rows=[("Lines", str(len(sql.splitlines()))
                                       if sql else "—")])
                if sql:
                    _print_sql(sql)
                continue

            # ── sql_validator ──
            if node == "sql_validator":
                passed = output.get("validation_passed")
                issues = output.get("validation_issues") or []
                attempts = output.get("sql_validation_attempts", "—")
                if passed is True:
                    _print_step_box(step_num, label, "✅ PASSED",
                                    rows=[("Attempt", str(attempts)),
                                          ("Issues",  "None")],
                                    status_style="green")
                elif passed is False:
                    issue_rows = [(f"  ✗", iss) for iss in issues] or [("Issues", "Unknown")]
                    _print_step_box(step_num, label, "⚠  FAILED",
                                    rows=[("Attempt", str(attempts))] + issue_rows,
                                    status_style="yellow")
                else:
                    _print_step_box(step_num, label, "SKIPPED",
                                    rows=[("Reason", "No SQL to validate")],
                                    status_style="dim")
                continue

            # ── executor ──
            if node == "executor":
                if output.get("error"):
                    _print_step_box(step_num, label, "⚠  ERROR",
                                    rows=[("Error",  output["error"][:200]),
                                          ("Retry",  "Will attempt self-correction")],
                                    status_style="yellow")
                else:
                    rows_out = output.get("query_result") or []
                    _print_step_box(step_num, label, "✅ SUCCESS",
                                    rows=[("Rows returned", str(len(rows_out))),
                                          ("Exec time",     f"{output.get('execution_time_ms', 0):.0f} ms")],
                                    status_style="green")
                continue

            # ── reflector ──
            if node == "reflector":
                msgs = output.get("messages") or []
                last = msgs[-1].content[:120] if msgs and hasattr(msgs[-1], "content") else "Rewriting SQL…"
                _print_step_box(step_num, label, "",
                                rows=[("Action", last)],
                                status_style="yellow")
                continue

            # ── responder / direct_respond ──
            if node in ("responder", "direct_respond"):
                _print_step_box(step_num, label, "",
                                rows=[("Status", "Generating natural language answer…")])
                continue

            # ── any other node ──
            _print_step_box(step_num, label, "",
                            rows=[(k, str(v)[:80]) for k, v in output.items()
                                  if v is not None and k not in ("messages",)][:5])

        elif event["type"] == "final":
            final_state = event["state"]

    # ── Final SQL ─────────────────────────────────────────────────────────
    is_data   = detected_intent in DATA_INTENTS
    final_sql = final_state.get("sql_query") if is_data else None

    if final_sql and show_steps:
        if RICH:
            console.print()
            console.print(Rule("[bold blue]🗃  Final SQL  (executed)[/bold blue]", style="blue"))
            console.print()
        else:
            print("\n" + "─" * 72)
            print("  🗃  Final SQL  (executed)")
            print("─" * 72)
        _print_sql(final_sql)

    # ── Phase 2: NL answer ────────────────────────────────────────────────
    if RICH:
        console.print()
        console.print(Rule("[bold green]💬  Answer[/bold green]", style="green"))
        console.print()
        console.print("  [bold white]Agent:[/bold white] ", end="")
    else:
        print("\n" + "─" * 72)
        print("  💬  Answer")
        print("─" * 72 + "\n")
        print("  Agent: ", end="", flush=True)

    nl_text = ""
    if stream:
        for token in stream_nl_response(final_state, conversation_history):
            print(token, end="", flush=True)
            nl_text += token
        print()
    else:
        for token in stream_nl_response(final_state, conversation_history):
            nl_text += token
        print(nl_text)

    # ── Phase 3: data table ───────────────────────────────────────────────
    if is_data and final_state.get("query_result") and not final_state.get("error"):
        if RICH:
            console.print()
            console.print(Rule("[bold blue]📊  Results[/bold blue]", style="blue"))
            console.print()
        else:
            print("\n" + "─" * 72)
            print("  📊  Results")
            print("─" * 72)
        _print_df(final_state.get("query_result"))

    # ── metrics ───────────────────────────────────────────────────────────
    _metrics_line(final_state)
    if RICH:
        console.print(Rule(style="bright_black"))
    else:
        print("\n" + "═" * 72)

    final_state["nl_response"] = nl_text
    return final_state


# ─── interactive conversation loop ────────────────────────────────────────────

def interactive_loop(show_steps: bool = True, stream: bool = True):
    """Multi-turn conversation loop with history."""
    if RICH:
        console.print(Panel(
            "[bold]📊 Nielsen Text-to-SQL Agent[/bold]\n\n"
            "Type your question and press [cyan]Enter[/cyan].\n"
            "Commands:  [cyan]exit[/cyan]  ·  [cyan]quit[/cyan]  ·  "
            "[cyan]clear[/cyan] [dim](reset history)[/dim]  ·  "
            "[cyan]history[/cyan] [dim](show turns)[/dim]",
            border_style="green", expand=False,
        ))
    else:
        print("=" * 70)
        print("  📊 Nielsen Text-to-SQL Agent")
        print("  Commands: exit | quit | clear | history")
        print("=" * 70)

    conversation_history: List[Dict] = []

    while True:
        try:
            if RICH:
                question = console.input("\n[bold green]You ▶[/bold green] ").strip()
            else:
                question = input("\nYou ▶ ").strip()
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
                    if RICH:
                        console.print(f"  [bold]{i}.[/bold] Q: {t['question']}")
                        console.print(f"     A: [dim]{textwrap.shorten(t.get('nl_response',''), 80)}[/dim]")
                    else:
                        print(f"  [{i}] Q: {t['question']}")
                        print(f"      A: {textwrap.shorten(t.get('nl_response',''), 80)}")
            continue

        try:
            final_state = run_question(
                question,
                conversation_history=conversation_history,
                show_steps=show_steps,
                stream=stream,
            )
            if final_state.get("nl_response"):
                conversation_history.append({
                    "question":             question,
                    "nl_response":          final_state["nl_response"],
                    # Store draft SQL even for clarification turns so the next
                    # turn's interaction_node can patch it via correction pipeline.
                    "sql":                  final_state.get("sql_query"),
                    "filter_clarification": final_state.get("pending_filter_clarification"),
                })
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
          python cli.py
          python cli.py "What was MONDELEZ market share in TOTAL BARS in 2024?"
          python cli.py "Compare OREO sales Dec 2023 vs Mar 2024" --no-stream --no-steps
          python cli.py "How did sales trend YTD?" --cache
        """),
    )
    parser.add_argument("question", nargs="?", default=None,
                        help="Question to ask (omit for interactive mode)")
    parser.add_argument("--no-stream", action="store_true",
                        help="Print full answer at once instead of streaming tokens")
    parser.add_argument("--no-steps", action="store_true",
                        help="Hide step-by-step agent thinking")
    parser.add_argument("--cache", action="store_true",
                        help="Enable semantic cache (off by default)")
    args = parser.parse_args()

    do_stream = not args.no_stream
    do_steps  = not args.no_steps

    if args.cache:
        from config import settings
        from tools import semantic_cache
        settings.enable_semantic_cache = True
        semantic_cache._init_backend()
        if RICH:
            console.print("[yellow]⚡ Semantic cache enabled[/yellow]")
        else:
            print("⚡ Semantic cache enabled")

    if args.question:
        run_question(args.question, show_steps=do_steps, stream=do_stream)
    else:
        interactive_loop(show_steps=do_steps, stream=do_stream)


if __name__ == "__main__":
    main()

