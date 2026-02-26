#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          Nielsen Text-to-SQL Agent  —  One-Click Business Setup            ║
║                     Databricks (DBRX) Provider Only                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Run this script ONCE to set up everything, then daily to launch the agent.

    python start.py          # prompts: Web UI or CLI?
    python start.py --ui     # always launch Web UI (Streamlit)
    python start.py --cli    # always launch CLI (terminal chat)
    python start.py --setup  # force re-run the credential wizard

You will be guided through credential entry on the first run.
On subsequent runs the saved credentials are reused automatically and you
are prompted to choose between the Web UI or the CLI.
"""

from __future__ import annotations

import os
import sys
import subprocess
import shutil
import textwrap
import re
import time
import platform
from pathlib import Path

# ─── constants ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
VENV_DIR     = SCRIPT_DIR / ".venv_dbrx"
ENV_FILE     = SCRIPT_DIR / ".env"
REQ_FILE     = SCRIPT_DIR / "requirements.txt"
PYTHON_MIN   = (3, 10)

# Packages NOT needed for Databricks-only mode (skips heavy local models etc.)
_SKIP_PKGS = {
    "langchain-groq",
    "sentence-transformers",
    "langchain-huggingface",
}

# ─── colour helpers (no external deps) ────────────────────────────────────────
_IS_WIN = platform.system() == "Windows"
_TERM   = shutil.get_terminal_size((80, 24))

def _c(text: str, code: str) -> str:
    """ANSI colour/style wrapper — skipped on Windows cmd without ANSI support."""
    if _IS_WIN and "WT_SESSION" not in os.environ and "TERM" not in os.environ:
        return text
    return f"\033[{code}m{text}\033[0m"

def _green(t):  return _c(t, "32")
def _yellow(t): return _c(t, "33")
def _red(t):    return _c(t, "31")
def _cyan(t):   return _c(t, "36")
def _bold(t):   return _c(t, "1")

def _banner():
    width = min(_TERM.columns, 78)
    line  = "═" * width
    print()
    print(_cyan(_bold(f"{'Nielsen Text-to-SQL Agent':^{width}}")))
    print(_cyan(_bold(f"{'Databricks Setup & Launch':^{width}}")))
    print(_cyan(line))
    print()

def _step(n: int, total: int, msg: str):
    print(_bold(_cyan(f"\n[{n}/{total}]")) + f"  {msg}")

def _ok(msg: str):
    print(_green("  ✓ ") + msg)

def _warn(msg: str):
    print(_yellow("  ⚠ ") + msg)

def _fail(msg: str):
    print(_red("  ✗ ") + msg)

def _info(msg: str):
    print("    " + msg)

def _ask(prompt: str, default: str = "", secret: bool = False) -> str:
    """Prompt the user for input, showing the current/default value."""
    disp_default = ("*" * min(len(default), 6) + "…" if secret and default else default)
    hint = f"  [{disp_default}]" if default else ""
    try:
        if secret:
            import getpass
            raw = getpass.getpass(f"  {_bold(prompt)}{hint}: ")
        else:
            raw = input(f"  {_bold(prompt)}{hint}: ")
    except (KeyboardInterrupt, EOFError):
        print()
        sys.exit(0)
    return raw.strip() or default


# ─── Step helpers ─────────────────────────────────────────────────────────────

def _check_python():
    v = sys.version_info
    if v < PYTHON_MIN:
        _fail(f"Python {PYTHON_MIN[0]}.{PYTHON_MIN[1]}+ required. "
              f"You are running Python {v.major}.{v.minor}.")
        _info("Download from https://www.python.org/downloads/")
        sys.exit(1)
    _ok(f"Python {v.major}.{v.minor}.{v.micro}")


def _venv_python() -> Path:
    """Return the Python executable inside the virtual environment."""
    if _IS_WIN:
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _create_venv():
    if VENV_DIR.exists():
        _ok(f"Virtual environment already exists ({VENV_DIR.name})")
        return
    _info("Creating virtual environment — this takes ~30 seconds …")
    subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])
    _ok("Virtual environment created")


def _install_deps():
    """Install DBRX-relevant packages from requirements.txt."""
    if not REQ_FILE.exists():
        _fail(f"requirements.txt not found at {REQ_FILE}")
        sys.exit(1)

    # Build filtered package list
    packages: list[str] = []
    with REQ_FILE.open() as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # Extract package name (before any version specifier)
            pkg_name = re.split(r"[>=<!;\[]", line)[0].strip().lower()
            if pkg_name in _SKIP_PKGS:
                continue
            packages.append(line)

    pip = str(_venv_python()).replace("python", "pip").replace(
        "Scripts\\python.exe", "Scripts\\pip.exe"
    ).replace("bin/python", "bin/pip")

    # Prefer pip from venv directly
    pip_exe = _venv_python().parent / ("pip.exe" if _IS_WIN else "pip")
    if not pip_exe.exists():
        pip_exe = _venv_python()
        pip_args = [str(pip_exe), "-m", "pip", "install", "--quiet"]
    else:
        pip_args = [str(pip_exe), "install", "--quiet"]

    _info("Installing packages (first run may take 3–5 minutes) …")

    # Install in one shot for speed
    try:
        subprocess.check_call(pip_args + packages)
        _ok(f"All {len(packages)} packages installed")
    except subprocess.CalledProcessError:
        _fail("pip install failed. See error above.")
        sys.exit(1)


# ─── .env management ──────────────────────────────────────────────────────────

def _load_env() -> dict[str, str]:
    """Parse existing .env into a dict (returns {} if file absent)."""
    env: dict[str, str] = {}
    if not ENV_FILE.exists():
        return env
    for raw in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def _write_env(env: dict[str, str]):
    lines = [
        "# ── Nielsen Text-to-SQL Agent — DBRX Configuration ──────────────────────────",
        "# Auto-generated by start.py — edit here or re-run start.py to update.",
        "",
        "# Provider (keep as dbrx — do not change)",
        "LLM_PROVIDER=dbrx",
        "",
        "# ── Databricks LLM API ───────────────────────────────────────────────────────",
        f'DBRX_API_KEY={env.get("DBRX_API_KEY", "")}',
        f'DBRX_BASE_URL={env.get("DBRX_BASE_URL", "")}',
        f'DBRX_MODEL_REASONING={env.get("DBRX_MODEL_REASONING", "databricks-gemini-2-5-pro")}',
        f'DBRX_MODEL={env.get("DBRX_MODEL", "databricks-gemini-2-5-flash")}',
        f'DBRX_TEMPERATURE={env.get("DBRX_TEMPERATURE", "0.0")}',
        f'DBRX_MAX_TOKENS={env.get("DBRX_MAX_TOKENS", "8192")}',
        "",
        "# ── Databricks SQL Connection ────────────────────────────────────────────────",
        f'DATABRICKS_SERVER_HOSTNAME={env.get("DATABRICKS_SERVER_HOSTNAME", "")}',
        f'DATABRICKS_HTTP_PATH={env.get("DATABRICKS_HTTP_PATH", "")}',
        f'DATABRICKS_ACCESS_TOKEN={env.get("DATABRICKS_ACCESS_TOKEN", "")}',
        f'DBX_CATALOG={env.get("DBX_CATALOG", "dev-amer-customer-catalog")}',
        f'DBX_SCHEMA={env.get("DBX_SCHEMA", "dev-amer-analyt-arisegenai-schema")}',
        f'DBX_TABLE={env.get("DBX_TABLE", "nielsen_market_ci_new_1")}',
        "",
        "# ── Embeddings ───────────────────────────────────────────────────────────────",
        f'DBRX_EMBEDDING_MODEL={env.get("DBRX_EMBEDDING_MODEL", "databricks-gte-large-en")}',
        f'DBRX_EMBEDDING_ENDPOINT_URL={env.get("DBRX_EMBEDDING_ENDPOINT_URL", "")}',
        "",
        "# ── Agent Behaviour ──────────────────────────────────────────────────────────",
        f'MAX_ITERATIONS={env.get("MAX_ITERATIONS", "3")}',
        f'ENABLE_SELF_CORRECTION={env.get("ENABLE_SELF_CORRECTION", "true")}',
        f'ENABLE_DYNAMIC_FEW_SHOT={env.get("ENABLE_DYNAMIC_FEW_SHOT", "true")}',
        f'FEW_SHOT_EXAMPLES_COUNT={env.get("FEW_SHOT_EXAMPLES_COUNT", "3")}',
        f'ENABLE_SQL_VALIDATOR={env.get("ENABLE_SQL_VALIDATOR", "true")}',
        f'QUERY_TIMEOUT_SECONDS={env.get("QUERY_TIMEOUT_SECONDS", "60")}',
        f'ENABLE_SEMANTIC_CACHE={env.get("ENABLE_SEMANTIC_CACHE", "false")}',
        "",
        "# ── Vector Store (ChromaDB) ───────────────────────────────────────────────────",
        f'VECTOR_STORE_PATH={env.get("VECTOR_STORE_PATH", "./data/vector_store")}',
        f'CHROMA_COLLECTION_NAME={env.get("CHROMA_COLLECTION_NAME", "sql_examples")}',
    ]
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ─── Credential collection ────────────────────────────────────────────────────

_REQUIRED_KEYS = [
    "DBRX_API_KEY",
    "DBRX_BASE_URL",
    "DATABRICKS_SERVER_HOSTNAME",
    "DATABRICKS_HTTP_PATH",
    "DATABRICKS_ACCESS_TOKEN",
    "DBRX_EMBEDDING_ENDPOINT_URL",
]

_PROMPTS = {
    "DBRX_API_KEY": (
        "Databricks LLM API Key\n"
        "    (Personal Access Token from your Databricks workspace settings)",
        True,
    ),
    "DBRX_BASE_URL": (
        "Databricks LLM Base URL\n"
        "    e.g. https://<workspace>.gcp.databricks.com/serving-endpoints",
        False,
    ),
    "DATABRICKS_SERVER_HOSTNAME": (
        "Databricks SQL Warehouse Hostname\n"
        "    e.g. <workspace>.gcp.databricks.com",
        False,
    ),
    "DATABRICKS_HTTP_PATH": (
        "Databricks SQL Warehouse HTTP Path\n"
        "    e.g. /sql/1.0/warehouses/<warehouse-id>",
        False,
    ),
    "DATABRICKS_ACCESS_TOKEN": (
        "Databricks Personal Access Token (for SQL)\n"
        "    (same token as LLM key if using a single workspace)",
        True,
    ),
    "DBRX_EMBEDDING_ENDPOINT_URL": (
        "Databricks Embedding Endpoint URL\n"
        "    e.g. https://<workspace>.gcp.databricks.com/serving-endpoints/databricks-gte-large-en/invocations",
        False,
    ),
}

_OPTIONAL_WITH_DEFAULT = {
    "DBX_CATALOG":  "dev-amer-customer-catalog",
    "DBX_SCHEMA":   "dev-amer-analyt-arisegenai-schema",
    "DBX_TABLE":    "nielsen_market_ci_new_1",
    "DBRX_MODEL_REASONING": "databricks-gemini-2-5-pro",
    "DBRX_MODEL":           "databricks-gemini-2-5-flash",
}


def _collect_credentials(env: dict[str, str]) -> dict[str, str]:
    """Interactively collect missing or update existing credentials."""
    missing = [k for k in _REQUIRED_KEYS if not env.get(k)]

    if not missing:
        print()
        _ok("All required credentials are already saved in .env")
        try:
            ans = input("  Do you want to update any credentials? [y/N]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print()
            sys.exit(0)
        if ans not in ("y", "yes"):
            return env

    print()
    print(_bold("  Please provide the following Databricks credentials."))
    print("  Press Enter to keep the existing value shown in [brackets].\n")

    updated = dict(env)

    for key in _REQUIRED_KEYS:
        prompt_text, is_secret = _PROMPTS[key]
        print()
        print(_cyan(f"  {prompt_text}"))
        val = _ask(key.replace("_", " ").title(), default=updated.get(key, ""), secret=is_secret)
        if val:
            updated[key] = val
        elif key not in updated:
            _warn(f"{key} left empty — you may need to update .env manually before running.")

    # Optional overrides
    print()
    print(_bold("  Table / Model settings (press Enter to keep defaults):"))
    for key, default in _OPTIONAL_WITH_DEFAULT.items():
        cur = updated.get(key, default)
        print()
        val = _ask(key, default=cur)
        updated[key] = val or cur

    return updated


# ─── Connection & vector-store setup (runs inside venv) ──────────────────────

_VERIFY_SCRIPT = """\
import sys, os
sys.path.insert(0, {root!r})
os.chdir({root!r})

# Force DBRX provider before config is imported
os.environ.setdefault("LLM_PROVIDER", "dbrx")

from dotenv import load_dotenv
load_dotenv({env_file!r}, override=True)

from config import settings

# ── 1. Test Databricks SQL connection ────────────────────────────────────────
print("[verify] Testing Databricks SQL connection …", flush=True)
try:
    from dbx_connection import run_dbx_query
    df = run_dbx_query(
        f"SELECT COUNT(*) AS cnt FROM {{settings.dbx_full_table}} LIMIT 1"
    )
    rows = int(df["cnt"].iloc[0]) if df is not None and "cnt" in df.columns else "?"
    print(f"[verify:ok] SQL connection OK — table has {{rows:,}} rows", flush=True)
except Exception as e:
    print(f"[verify:fail] SQL connection failed: {{e}}", flush=True)
    sys.exit(1)

# ── 2. Seed few-shot examples into the vector store ─────────────────────────
print("[verify] Seeding few-shot vector store …", flush=True)
try:
    from tools import seed_examples
    seed_examples()
    print("[verify:ok] Few-shot vector store ready", flush=True)
except Exception as e:
    print(f"[verify:fail] Vector store seeding failed: {{e}}", flush=True)
    sys.exit(2)

print("[verify:done]", flush=True)
"""

_LAUNCH_SCRIPT = """\
import sys, os
sys.path.insert(0, {root!r})
os.chdir({root!r})

os.environ.setdefault("LLM_PROVIDER", "dbrx")

# Pre-load .env so streamlit subprocess picks it up
from dotenv import load_dotenv
load_dotenv({env_file!r}, override=True)

import subprocess
cmd = [sys.executable, "-m", "streamlit", "run", {app!r},
       "--server.headless", "false",
       "--browser.gatherUsageStats", "false",
       "--theme.base", "light",
       "--theme.primaryColor", "#0053A5",
]
subprocess.run(cmd)
"""

_LAUNCH_CLI_SCRIPT = """\
import sys, os
sys.path.insert(0, {root!r})
os.chdir({root!r})

os.environ.setdefault("LLM_PROVIDER", "dbrx")

from dotenv import load_dotenv
load_dotenv({env_file!r}, override=True)

import subprocess
subprocess.run([sys.executable, {cli!r}])
"""


def _run_in_venv(code: str, live_output: bool = True) -> int:
    """Execute a Python snippet inside the project's virtual environment."""
    py = str(_venv_python())
    env = os.environ.copy()
    env["LLM_PROVIDER"] = "dbrx"
    env["PYTHONPATH"]   = str(SCRIPT_DIR)
    proc = subprocess.Popen(
        [py, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    ok = True
    for line in proc.stdout:  # type: ignore[union-attr]
        stripped = line.rstrip()
        if "[verify:ok]" in stripped:
            _ok(stripped.replace("[verify:ok]", "").strip())
        elif "[verify:fail]" in stripped:
            _fail(stripped.replace("[verify:fail]", "").strip())
            ok = False
        elif "[verify]" in stripped:
            _info(stripped.replace("[verify]", "").strip())
        elif stripped:
            if live_output:
                _info(stripped)
    proc.wait()
    return proc.returncode


def _verify_connection():
    script = _VERIFY_SCRIPT.format(
        root=str(SCRIPT_DIR),
        env_file=str(ENV_FILE),
    )
    rc = _run_in_venv(script)
    if rc != 0:
        _fail("Setup verification failed. Check the errors above and update .env.")
        _info(f"You can edit credentials directly in: {ENV_FILE}")
        sys.exit(1)


def _choose_interface() -> str:
    """Ask the user whether they want the Web UI or the CLI."""
    # Allow skipping the prompt via command-line flags
    if "--ui" in sys.argv:
        return "ui"
    if "--cli" in sys.argv:
        return "cli"

    width = min(_TERM.columns, 78)
    print()
    print(_cyan("─" * width))
    print(_bold("  How would you like to run the agent?"))
    print()
    print(_bold(_green("  1")) + "  Web UI    — browser-based chat  " +
          _yellow("(app.py)"))
    print(_bold(_green("  2")) + "  CLI       — terminal chat        " +
          _yellow("(cli.py)"))
    print(_cyan("─" * width))
    while True:
        try:
            choice = input(_bold("  Enter 1 or 2 [default: 1]: ")).strip() or "1"
        except (KeyboardInterrupt, EOFError):
            print()
            sys.exit(0)
        if choice in ("1", "ui", "web"):
            return "ui"
        if choice in ("2", "cli", "terminal"):
            return "cli"
        print(_yellow("  Please enter 1 or 2."))


def _build_env() -> dict:
    """Return os.environ copy with DBRX settings and .env values pre-loaded."""
    env = os.environ.copy()
    env["LLM_PROVIDER"] = "dbrx"
    env["PYTHONPATH"]   = str(SCRIPT_DIR)
    if ENV_FILE.exists():
        for raw in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return env


def _launch_ui():
    app_path = str(SCRIPT_DIR / "app.py")
    script = _LAUNCH_SCRIPT.format(
        root=str(SCRIPT_DIR),
        env_file=str(ENV_FILE),
        app=app_path,
    )
    subprocess.run([str(_venv_python()), "-c", script], env=_build_env())


def _launch_cli():
    cli_path = str(SCRIPT_DIR / "cli.py")
    script = _LAUNCH_CLI_SCRIPT.format(
        root=str(SCRIPT_DIR),
        env_file=str(ENV_FILE),
        cli=cli_path,
    )
    subprocess.run([str(_venv_python()), "-c", script], env=_build_env())


def _launch(mode: str):
    """Launch the selected interface."""
    if mode == "cli":
        _ok("Starting CLI …  (type 'exit' or Ctrl+C to quit)")
        print()
        _launch_cli()
    else:
        _ok("Starting Web UI …  (press Ctrl+C in this terminal to stop)")
        _info("The Streamlit app will open in your browser automatically.")
        print()
        _launch_ui()


# ─── First-run flag ───────────────────────────────────────────────────────────

def _is_first_run() -> bool:
    """True when .env is missing or any required credential is empty."""
    env = _load_env()
    return any(not env.get(k) for k in _REQUIRED_KEYS)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    _banner()

    total = 5

    # ── Step 1: Python version ───────────────────────────────────────────────
    _step(1, total, "Checking Python version")
    _check_python()

    # ── Step 2: Virtual environment ──────────────────────────────────────────
    _step(2, total, f"Setting up isolated Python environment ({VENV_DIR.name})")
    _create_venv()

    # ── Step 3: Install packages ─────────────────────────────────────────────
    _step(3, total, "Installing required packages")
    _install_deps()

    # ── Step 4: Credentials ──────────────────────────────────────────────────
    _step(4, total, "Configuring Databricks credentials")
    env = _load_env()
    env = _collect_credentials(env)
    env["LLM_PROVIDER"] = "dbrx"
    _write_env(env)
    _ok(f"Credentials saved to {ENV_FILE.name}")

    # ── Step 5: Verify connection + seed vector store ────────────────────────
    _step(5, total, "Verifying Databricks connection & seeding vector store")
    _verify_connection()

    # ── Done — choose interface & launch ─────────────────────────────────────
    print()
    print(_cyan("─" * min(_TERM.columns, 78)))
    print(_bold(_green("  ✓ Setup complete!")))
    print(_cyan("─" * min(_TERM.columns, 78)))
    time.sleep(0.5)
    mode = _choose_interface()
    _launch(mode)


if __name__ == "__main__":
    # ── Quick-launch mode: if .env is already configured, skip setup wizard ──
    if not _is_first_run() and "--setup" not in sys.argv:
        _banner()
        total = 3  # venv check, deps, launch

        _step(1, 3, "Checking Python version")
        _check_python()

        _step(2, 3, "Verifying virtual environment & packages")
        _create_venv()
        _install_deps()

        _step(3, 3, "Choosing interface & launching")

        env = _load_env()
        env["LLM_PROVIDER"] = "dbrx"
        _write_env(env)

        print()
        print(_bold(_green("  ✓ Environment ready.")))
        time.sleep(0.5)
        mode = _choose_interface()
        _launch(mode)
    else:
        main()
