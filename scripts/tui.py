#!/usr/bin/env python3
"""
ClawTop — OpenClaw Mission Control
A read-only btop-style dashboard for local OpenClaw development.

Usage:
    python3 {baseDir}/scripts/tui.py
    python3 {baseDir}/scripts/tui.py --refresh 5 --no-logs
    python3 {baseDir}/scripts/tui.py --once

Requires: rich (auto-installed on first run), psutil
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# ── Auto-bootstrap: install rich/psutil into a local venv if missing ─────────
_VENV_DIR = Path(__file__).parent / ".venv"

def _ensure_deps() -> None:
    """
    Install missing deps into a skill-local venv at scripts/.venv/.
    Works on Ubuntu 24's externally-managed Python without --break-system-packages.
    Re-execs the script through the venv interpreter on first install.
    """
    # If we're already running inside our venv, just check imports are present.
    in_our_venv = sys.prefix == str(_VENV_DIR)

    missing = []
    for pkg in ("rich", "psutil"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if not missing:
        return  # Everything importable — nothing to do.

    # Not yet in venv: create it if needed, install, then re-exec inside it.
    venv_python = _VENV_DIR / "bin" / "python3"
    if not venv_python.exists():
        print("[clawtop] Creating local venv at scripts/.venv/ …")
        result = subprocess.run([sys.executable, "-m", "venv", str(_VENV_DIR)])
        if result.returncode != 0:
            sys.exit("[clawtop] Failed to create venv. Ensure python3-venv is installed:\n"
                     "  sudo apt install python3-venv")

    print(f"[clawtop] Installing: {', '.join(missing)} …")
    result = subprocess.run([str(venv_python), "-m", "pip", "install", "--quiet", *missing])
    if result.returncode != 0:
        sys.exit(f"[clawtop] pip install failed. Try manually:\n"
                 f"  {venv_python} -m pip install {' '.join(missing)}")

    # Re-exec using the venv python so imports resolve correctly.
    import os
    os.execv(str(venv_python), [str(venv_python)] + sys.argv)

_ensure_deps()

# If running inside the venv, site-packages are already on sys.path.
# If running system python but venv exists (deps already installed), add it.
if not sys.prefix == str(_VENV_DIR) and _VENV_DIR.exists():
    import site
    venv_site = next(_VENV_DIR.glob("lib/python*/site-packages"), None)
    if venv_site:
        sys.path.insert(0, str(venv_site))

# ── Imports (after deps guaranteed) ──────────────────────────────────────────
import argparse
import time
from datetime import datetime
from typing import Optional

# Allow importing data.py from the same scripts/ directory
sys.path.insert(0, str(Path(__file__).parent))
import data as D
import psutil

from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ── Palette ───────────────────────────────────────────────────────────────────

C_ACCENT   = "cyan"
C_TITLE    = "bold cyan"
C_OK       = "bright_green"
C_WARN     = "yellow"
C_ERR      = "bright_red"
C_MUTED    = "grey58"
C_DIM      = "grey42"
C_BORDER   = "grey35"


# ── Shared widgets ────────────────────────────────────────────────────────────

def _bar(pct: float, width: int = 18) -> Text:
    """Render a Unicode block progress bar, colour-coded by percentage."""
    clamped = max(0.0, min(100.0, pct))
    filled  = round(clamped / 100 * width)
    empty   = width - filled
    color   = C_OK if clamped < 60 else C_WARN if clamped < 80 else C_ERR
    t = Text()
    t.append("█" * filled, style=color)
    t.append("░" * empty,  style=C_DIM)
    return t


def _fmt_bytes(b: float) -> str:
    """Format bytes/s as human-readable rate string."""
    if b < 1024:
        return f"{b:.0f} B/s"
    if b < 1024 ** 2:
        return f"{b/1024:.1f} KB/s"
    return f"{b/1024**2:.1f} MB/s"


def _fmt_age(secs: Optional[int]) -> str:
    if secs is None:
        return "—"
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}m"
    return f"{secs // 3600}h"


def _pct_color(pct: Optional[float]) -> str:
    if pct is None:
        return C_MUTED
    if pct >= 85:
        return C_ERR
    if pct >= 65:
        return C_WARN
    return C_OK


def _fmt_tokens(total: Optional[int], ctx: Optional[int], pct: Optional[float]) -> Text:
    t = Text()
    if total is None:
        t.append("—", style=C_MUTED)
        return t
    def _k(n: int) -> str:
        return f"{n/1000:.0f}k" if n >= 10_000 else f"{n/1000:.1f}k"
    label = _k(total)
    if ctx:
        label += f"/{_k(ctx)}"
    if pct is not None:
        label += f" ({pct:.0f}%)"
    t.append(label, style=_pct_color(pct))
    return t


# ── Panel renderers ───────────────────────────────────────────────────────────

def render_header(d: D.DashboardData, now_str: str) -> Panel:
    gw = d.gateway
    status_dot = Text("● ", style=C_OK if gw.up else C_ERR)
    status_txt = Text("UP" if gw.up else "DOWN", style=C_OK if gw.up else C_ERR)

    row = Text()
    row.append("🦀  ClawTop", style=C_TITLE)
    row.append("   │   gateway: ", style=C_MUTED)
    row.append_text(status_dot)
    row.append_text(status_txt)
    row.append(f"  {gw.url}", style=C_DIM)
    if gw.default_model:
        row.append(f"   │   model: ", style=C_MUTED)
        row.append(str(gw.default_model), style=C_ACCENT)
    row.append(f"   │   {now_str}", style=C_MUTED)

    return Panel(row, style=C_BORDER, box=box.HORIZONTALS, padding=(0, 1))


def render_sessions(sessions: list[D.SessionInfo]) -> Panel:
    if not sessions:
        body = Text("\n  No sessions found", style=C_MUTED)
        return Panel(body, title=f"[{C_TITLE}]SESSIONS[/{C_TITLE}]",
                     border_style=C_BORDER, box=box.ROUNDED, padding=(0, 1))

    tbl = Table(box=None, show_header=True, pad_edge=False,
                header_style=C_DIM, expand=True)
    tbl.add_column("KEY",    style=C_ACCENT, no_wrap=True, max_width=18)
    tbl.add_column("MODEL",  style="white",  no_wrap=True, max_width=22)
    tbl.add_column("TOKENS", no_wrap=True,   min_width=16)
    tbl.add_column("AGE",    style=C_MUTED,  no_wrap=True, width=5)
    tbl.add_column("KIND",   style=C_DIM,    no_wrap=True, width=7)

    for s in sessions:
        kind_color = C_ACCENT if s.kind == "direct" else C_WARN if s.kind == "group" else C_MUTED
        tbl.add_row(
            s.key,
            s.model,
            _fmt_tokens(s.total_tokens, s.context_tokens, s.percent_used),
            _fmt_age(s.age),
            Text(s.kind[:7], style=kind_color),
        )

    return Panel(tbl, title=f"[{C_TITLE}]SESSIONS  [{C_MUTED}]{len(sessions)} active[/{C_MUTED}][/{C_TITLE}]",
                 border_style=C_BORDER, box=box.ROUNDED, padding=(0, 1))


def render_system(sys_m: D.SystemMetrics) -> Panel:
    lines: list[Text] = []

    def metric_row(label: str, pct: float, used: str, total: str) -> None:
        row = Text()
        row.append(f"  {label:<5}", style=C_MUTED)
        row.append_text(_bar(pct))
        color = C_OK if pct < 60 else C_WARN if pct < 80 else C_ERR
        row.append(f"  {pct:5.1f}%", style=color)
        row.append(f"  {used}/{total}", style=C_DIM)
        lines.append(row)

    metric_row("CPU",  sys_m.cpu_pct,
               f"{sys_m.cpu_pct:.0f}%",       f"{psutil.cpu_count()} cores")
    metric_row("MEM",  sys_m.mem_pct,
               f"{sys_m.mem_used_gb:.1f}G", f"{sys_m.mem_total_gb:.1f}G")
    metric_row("DISK", sys_m.disk_pct,
               f"{sys_m.disk_used_gb:.0f}G", f"{sys_m.disk_total_gb:.0f}G")

    # Network row
    net_row = Text()
    net_row.append("  NET  ", style=C_MUTED)
    net_row.append("↑ ", style=C_OK)
    net_row.append(f"{_fmt_bytes(sys_m.net_sent_bps):<12}", style="white")
    net_row.append("↓ ", style=C_ACCENT)
    net_row.append(_fmt_bytes(sys_m.net_recv_bps), style="white")
    lines.append(net_row)

    body = Text("\n").join([Text(""), *lines, Text("")])

    return Panel(body, title=f"[{C_TITLE}]SYSTEM[/{C_TITLE}]",
                 border_style=C_BORDER, box=box.ROUNDED, padding=(0, 0))





def render_token_breakdown(tb: D.TokenBreakdown) -> Panel:
    """Full panel showing per-type token bars + cache hit rate + compactions."""
    lines: list[Text] = []

    if tb.total_tokens == 0:
        body = Text("\n  No token data yet", style=C_MUTED)
        return Panel(body, title=f"[{C_TITLE}]TOKENS[/{C_TITLE}]",
                     border_style=C_BORDER, box=box.ROUNDED, padding=(0, 0))

    total = max(tb.total_tokens, 1)

    def _tok_row(label: str, count: int, suffix: str = "", bar_width: int = 14) -> Text:
        pct = count / total * 100
        row = Text()
        row.append(f"  {label:<8}", style=C_MUTED)
        row.append_text(_bar(pct, width=bar_width))
        color = C_OK if pct < 60 else C_WARN if pct < 80 else C_ERR
        row.append(f"  {pct:4.0f}%", style=color)
        row.append(f"  {_k(count)}", style="white")
        if suffix:
            row.append(f"  {suffix}", style=C_DIM)
        return row

    has_breakdown = (tb.input_tokens > 0 or tb.output_tokens > 0
                     or tb.cache_read > 0 or tb.cache_write > 0)

    if has_breakdown:
        lines.append(_tok_row("input",   tb.input_tokens))
        lines.append(_tok_row("output",  tb.output_tokens))
        lines.append(_tok_row("c.read",  tb.cache_read,  "↩"))
        lines.append(_tok_row("c.write", tb.cache_write))
    else:
        # Only totalTokens available — show single bar
        lines.append(_tok_row("total",   tb.total_tokens))
        note = Text()
        note.append("  breakdown unavailable", style=C_MUTED)
        note.append("  (runs before token tracking)", style=C_DIM)
        lines.append(note)

    lines.append(Text(""))

    # Summary row
    stat = Text()
    if has_breakdown and tb.cache_hit_rate > 0:
        stat.append("  cache hit  ", style=C_MUTED)
        hit_color = C_OK if tb.cache_hit_rate >= 50 else C_WARN if tb.cache_hit_rate >= 20 else C_ERR
        stat.append(f"{tb.cache_hit_rate:.0f}%", style=hit_color)
        stat.append("   compactions  ", style=C_MUTED)
    else:
        stat.append("  compactions  ", style=C_MUTED)
    stat.append(str(tb.total_compactions), style=C_ACCENT if tb.total_compactions else C_MUTED)
    lines.append(stat)

    body = Text("\n").join([Text(""), *lines, Text("")])
    total_k = _k(tb.total_tokens)
    title = f"[{C_TITLE}]TOKENS  [{C_MUTED}]{total_k} total[/{C_MUTED}][/{C_TITLE}]"
    return Panel(body, title=title, border_style=C_BORDER, box=box.ROUNDED, padding=(0, 0))


def render_memory(mem: D.MemorySummary) -> Panel:
    """Panel showing memory files + last flush info."""
    lines: list[Text] = []

    has_any = mem.memory_md or mem.dir_files or mem.dir_total_kb > 0

    if not has_any:
        body = Text("\n  No memory files found", style=C_MUTED)
        return Panel(body, title=f"[{C_TITLE}]MEMORY[/{C_TITLE}]",
                     border_style=C_BORDER, box=box.ROUNDED, padding=(0, 0))

    if mem.memory_md:
        row = Text()
        row.append("  MEMORY.md  ", style=C_ACCENT)
        row.append(f"{mem.memory_md.size_kb:.1f} KB", style="white")
        row.append(f"  {_fmt_age(mem.memory_md.age_secs)} ago", style=C_MUTED)
        lines.append(row)

    if mem.dir_files or mem.dir_total_kb > 0:
        dir_row = Text()
        dir_row.append("  memory/    ", style=C_ACCENT)
        dir_row.append(f"{len(mem.dir_files)} files", style="white")
        dir_row.append(f"  {mem.dir_total_kb:.1f} KB total", style=C_MUTED)
        lines.append(dir_row)
        for i, mf in enumerate(mem.dir_files[:3]):
            sub = Text()
            marker = "└" if i == len(mem.dir_files[:3]) - 1 else "├"
            sub.append(f"    {marker} {mf.name:<22}", style=C_DIM)
            sub.append(f"{mf.size_kb:.1f} KB", style=C_MUTED)
            if i == 0:
                sub.append("  ← newest", style=C_DIM)
            lines.append(sub)

    # Flush status
    if mem.last_flush_at is not None or mem.current_comp > 0:
        lines.append(Text(""))
        flush_row = Text()
        flush_row.append("  last flush  ", style=C_MUTED)
        if mem.last_flush_at is not None:
            flush_row.append(f"{_fmt_age(mem.last_flush_at)} ago", style="white")
        else:
            flush_row.append("never", style=C_MUTED)
        if mem.last_flush_comp is not None:
            flush_row.append(f"  ∷{mem.last_flush_comp}", style=C_DIM)
        if mem.flush_overdue:
            flush_row.append("  ⚠ overdue", style=C_WARN)
        elif mem.current_comp > 0:
            flush_row.append("  ✓", style=C_OK)
        lines.append(flush_row)

    body = Text("\n").join([Text(""), *lines, Text("")])
    return Panel(body, title=f"[{C_TITLE}]MEMORY[/{C_TITLE}]",
                 border_style=C_BORDER, box=box.ROUNDED, padding=(0, 0))


def render_status_bar(sk: D.SkillSummary, pl: D.PluginSummary) -> Panel:
    """Compact single-line panel combining skills + plugins."""
    t = Text()

    # Skills section
    t.append("  skills ", style=C_MUTED)
    t.append(str(sk.total), style=C_ACCENT)
    t.append("  ✓ ", style=C_MUTED)
    t.append(str(sk.eligible), style=C_OK)
    if sk.disabled:
        t.append("  ⏸ ", style=C_MUTED)
        t.append(str(sk.disabled), style=C_WARN)
    if sk.missing_names:
        t.append("  ✗ ", style=C_MUTED)
        t.append(str(len(sk.missing_names)), style=C_ERR)
        t.append(f" ({', '.join(sk.missing_names[:3])})", style=C_DIM)

    # Separator
    t.append("     │     ", style=C_BORDER)

    # Plugins section
    t.append("plugins ", style=C_MUTED)
    if pl.total == 0:
        t.append("none", style=C_MUTED)
    else:
        t.append(str(pl.loaded), style=C_OK)
        t.append(f"/{pl.total}", style=C_MUTED)
        t.append(" loaded", style=C_MUTED)
        if pl.entries:
            names = "  " + "  ".join(
                f"{'✓' if e.status=='loaded' else '✗'} {e.name[:12]}"
                for e in pl.entries[:4]
            )
            t.append(names, style=C_DIM)

    return Panel(t, style=C_BORDER, box=box.HORIZONTALS, padding=(0, 0))


# ── Token formatter helper (shared) ─────────────────────────────────────────

def _k(n: int) -> str:
    """Format large token counts as compact strings."""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 10_000:
        return f"{n/1000:.0f}k"
    if n >= 1_000:
        return f"{n/1000:.1f}k"
    return str(n)




def render_logs(log_lines: list[str], gateway_up: bool) -> Panel:
    if not gateway_up:
        body = Text("\n  Gateway offline — logs unavailable", style=C_MUTED)
        return Panel(body, title=f"[{C_TITLE}]LOGS[/{C_TITLE}]",
                     border_style=C_BORDER, box=box.ROUNDED, padding=(0, 1))

    if not log_lines:
        body = Text("\n  No logs available", style=C_MUTED)
        return Panel(body, title=f"[{C_TITLE}]LOGS[/{C_TITLE}]",
                     border_style=C_BORDER, box=box.ROUNDED, padding=(0, 1))

    rows: list[Text] = []
    for raw_line in log_lines[-30:]:
        line = raw_line.strip()
        if not line:
            continue
        t = Text()
        # Colour-code by log level keyword
        if "[error]" in line or " ERROR " in line:
            t.append(line, style=C_ERR)
        elif "[warn]" in line or " WARN " in line:
            t.append(line, style=C_WARN)
        elif "[info]" in line or " INFO " in line:
            # Dim the timestamp prefix (first token), highlight rest
            parts = line.split(" ", 1)
            t.append(parts[0] + " ", style=C_DIM)
            if len(parts) > 1:
                t.append(parts[1], style="white")
        else:
            t.append(line, style=C_MUTED)
        rows.append(t)

    body = Text("\n").join(rows) if rows else Text("  (empty)", style=C_MUTED)
    title = f"[{C_TITLE}]LOGS  [{C_MUTED}]last {len(rows)} lines[/{C_MUTED}][/{C_TITLE}]"
    return Panel(body, title=title, border_style=C_BORDER, box=box.ROUNDED, padding=(0, 1))


def render_footer(refresh: float, last_ms: int) -> Panel:
    t = Text()
    t.append("  Ctrl+C", style=C_ACCENT)
    t.append(" to quit   ", style=C_MUTED)
    t.append("↻", style=C_DIM)
    t.append(f" refreshing every {refresh:.0f}s", style=C_MUTED)
    t.append(f"   last fetch: {last_ms}ms", style=C_DIM)
    return Panel(t, style=C_BORDER, box=box.HORIZONTALS, padding=(0, 0))


# ── Layout assembly ───────────────────────────────────────────────────────────

def build_layout(show_logs: bool) -> Layout:
    root = Layout(name="root")
    root.split_column(
        Layout(name="header",     size=3),
        Layout(name="body"),
        Layout(name="footer",     size=3),
    )
    if show_logs:
        root["body"].split_column(
            Layout(name="panels", ratio=4),
            Layout(name="logs",   ratio=2),
        )
        panels = root["body"]["panels"]
    else:
        root["body"].split_column(Layout(name="panels"))
        panels = root["body"]["panels"]

    panels.split_column(
        Layout(name="top_row",    ratio=3),
        Layout(name="mid_row",    ratio=3),
        Layout(name="status_bar", size=3),
    )
    panels["top_row"].split_row(
        Layout(name="sessions", ratio=3),
        Layout(name="system",   ratio=2),
    )
    panels["mid_row"].split_row(
        Layout(name="tokens", ratio=3),
        Layout(name="memory", ratio=2),
    )
    return root


def update_layout(layout: Layout, d: D.DashboardData,
                  refresh: float, last_ms: int, show_logs: bool) -> None:
    now_str = datetime.now().strftime("%H:%M:%S")
    layout["header"].update(render_header(d, now_str))
    layout["footer"].update(render_footer(refresh, last_ms))

    panels = layout["body"]["panels"]
    panels["top_row"]["sessions"].update(render_sessions(d.sessions))
    panels["top_row"]["system"].update(render_system(d.system))
    panels["mid_row"]["tokens"].update(render_token_breakdown(d.tokens))
    panels["mid_row"]["memory"].update(render_memory(d.memory))
    panels["status_bar"].update(render_status_bar(d.skills, d.plugins))

    if show_logs:
        layout["body"]["logs"].update(render_logs(d.logs, d.gateway.up))


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="openclaw clawtop",
        description="ClawTop — OpenClaw local dev dashboard",
    )
    p.add_argument("--refresh",   type=float, default=2.0,
                   metavar="SECS", help="Refresh interval in seconds (default: 2)")
    p.add_argument("--no-logs",   action="store_true",
                   help="Hide the logs panel")
    p.add_argument("--log-lines", type=int, default=22,
                   metavar="N", help="Number of log lines to display (default: 22)")
    p.add_argument("--once",      action="store_true",
                   help="Render a single snapshot and exit")
    p.add_argument("--debug",     action="store_true",
                   help="Run diagnostics: show per-command timing and errors, then exit")
    return p.parse_args()


def run_debug() -> None:
    """Print a diagnostic table: checks each data source, shows path/timing/errors."""
    from rich.table import Table
    console = Console()

    console.print()
    console.print(Text("🦀  ClawTop — diagnostics", style=C_TITLE))
    console.print(Text("Checking each data source…\n", style=C_MUTED))

    tbl = Table(box=box.ROUNDED, border_style=C_BORDER, show_header=True,
                header_style=C_DIM, expand=True)
    tbl.add_column("SOURCE",  style="white",  no_wrap=True, min_width=20)
    tbl.add_column("STATUS",  no_wrap=True,   width=10)
    tbl.add_column("TIME",    style=C_MUTED,  width=8, justify="right")
    tbl.add_column("DETAIL",  style=C_DIM,    ratio=3)
    tbl.add_column("ERROR",   style=C_ERR,    ratio=2)

    results = D.run_diagnostics()
    for r in results:
        status = Text("✓  ok", style=C_OK) if r["ok"] else Text("✗  fail", style=C_ERR)
        elapsed = f"{r['elapsed_ms']}ms"
        tbl.add_row(r["label"], status, elapsed,
                    r["detail"] or "—",
                    r["error"] or "")

    console.print(tbl)
    console.print()
    console.print(Text("All data is read from local files — no openclaw process is spawned.", style=C_MUTED))
    console.print()

def main() -> None:
    args = parse_args()

    if args.debug:
        run_debug()
        return

    show_logs = not args.no_logs
    console = Console()

    layout = build_layout(show_logs)

    # Initial fetch (seeds network rate counter)
    t0 = time.monotonic()
    dash = D.fetch_all(log_limit=args.log_lines, include_logs=show_logs)
    last_ms = int((time.monotonic() - t0) * 1000)
    update_layout(layout, dash, args.refresh, last_ms, show_logs)

    if args.once:
        console.print(layout)
        return

    try:
        with Live(layout, console=console, refresh_per_second=2,
                  screen=True, vertical_overflow="visible") as live:
            while True:
                time.sleep(args.refresh)
                t0 = time.monotonic()
                dash = D.fetch_all(log_limit=args.log_lines, include_logs=show_logs)
                last_ms = int((time.monotonic() - t0) * 1000)
                update_layout(layout, dash, args.refresh, last_ms, show_logs)
                live.refresh()
    except KeyboardInterrupt:
        pass  # clean exit on Ctrl+C


if __name__ == "__main__":
    main()


def run_debug() -> None:
    """Print a diagnostic table: checks each data source, shows path/timing/errors."""
    from rich.table import Table
    console = Console()

    console.print()
    console.print(Text("🦀  ClawTop — diagnostics", style=C_TITLE))
    console.print(Text("Checking each data source…\n", style=C_MUTED))

    tbl = Table(box=box.ROUNDED, border_style=C_BORDER, show_header=True,
                header_style=C_DIM, expand=True)
    tbl.add_column("SOURCE",  style="white",  no_wrap=True, min_width=20)
    tbl.add_column("STATUS",  no_wrap=True,   width=10)
    tbl.add_column("TIME",    style=C_MUTED,  width=8, justify="right")
    tbl.add_column("DETAIL",  style=C_DIM,    ratio=3)
    tbl.add_column("ERROR",   style=C_ERR,    ratio=2)

    results = D.run_diagnostics()
    for r in results:
        status = Text("✓  ok", style=C_OK) if r["ok"] else Text("✗  fail", style=C_ERR)
        elapsed = f"{r['elapsed_ms']}ms"
        tbl.add_row(r["label"], status, elapsed,
                    r["detail"] or "—",
                    r["error"] or "")

    console.print(tbl)
    console.print()
    console.print(Text("All data is read from local files — no openclaw process is spawned.", style=C_MUTED))
    console.print()
