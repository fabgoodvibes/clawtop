---
name: clawtop
description: Launch ClawTop, a btop-style terminal dashboard for local OpenClaw development. Shows sessions with token/model usage, system metrics (CPU, memory, disk, network), skills status, plugins status, and live gateway logs — all in a single read-only live display. Trigger when asked to open ClawTop, the mission control TUI, dev dashboard, or monitor openclaw locally.
metadata:
  {
    "openclaw":
      {
        "emoji": "🦀",
        "os": ["linux", "darwin"],
        "requires": { "bins": ["python3"] },
      },
  }
---

# ClawTop — OpenClaw Mission Control

Read-only terminal dashboard: sessions · system · skills · plugins · logs.

## Launch

```bash
python3 {baseDir}/scripts/tui.py
```

`rich` and `psutil` are auto-installed into a local venv on first run. Refreshes every 2 seconds. Press **Ctrl+C** to quit.

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--refresh N` | `2` | Refresh interval in seconds |
| `--no-logs` | off | Hide the logs panel |
| `--log-lines N` | `22` | Log lines to display |
| `--once` | off | Render one snapshot and exit (no loop) |

```bash
python3 {baseDir}/scripts/tui.py --refresh 5 --no-logs
python3 {baseDir}/scripts/tui.py --once
```

## Panels

- **SESSIONS** — key, model, token usage with colour-coded % bars, age, kind
- **SYSTEM** — CPU / MEM / DISK bars + network I/O rates (live, non-blocking)
- **SKILLS** — eligible / disabled / missing counts with missing names listed
- **PLUGINS** — loaded / total counts with per-plugin status
- **LOGS** — live tail from `openclaw logs`; hidden when gateway is offline

## Requirements

- `python3` in PATH — `rich` and `psutil` auto-installed on first run
- `openclaw` CLI in PATH (reads sessions, skills, plugins, logs)
- Gateway does **not** need to be running — all panels except LOGS degrade gracefully
