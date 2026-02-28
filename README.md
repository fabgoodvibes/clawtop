# 🦀 ClawTop

A `btop`-style read-only terminal dashboard for local [OpenClaw](https://openclaw.ai) development.

<img width="946" height="930" alt="Screenshot from 2026-02-28 19-39-25" src="https://github.com/user-attachments/assets/6883b042-1ac3-41bd-b82c-542ec00146d9" />


## Installation

Clone this repo and drop the `clawtop/` folder into your OpenClaw skills directory:

```bash
git clone git@github.com:fabgoodvibes/clawtop.git
cp -r clawtop/ ~/.openclaw/workspace/skills/
```

Or install via `.skill` file from the OpenClaw desktop app.

## Usage

```bash
python3 ~/.openclaw/workspace/skills/clawtop/scripts/tui.py
```

On first run, `rich` and `psutil` are automatically installed into a local venv at `scripts/.venv/`. No `sudo` or system package changes required.

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--refresh N` | `2` | Refresh interval in seconds |
| `--no-logs` | off | Hide the logs panel |
| `--log-lines N` | `22` | Number of log lines shown |
| `--once` | — | Render a single snapshot and exit |
| `--debug` | — | Diagnose data sources and exit |

### Examples

```bash
# Default: refresh every 2s
python3 ~/.openclaw/workspace/skills/clawtop/scripts/tui.py

# Calmer refresh, no logs
python3 ~/.openclaw/workspace/skills/clawtop/scripts/tui.py --refresh 5 --no-logs

# One-shot snapshot (useful for scripting / screenshots)
python3 ~/.openclaw/workspace/skills/clawtop/scripts/tui.py --once

# Diagnose why a panel shows no data
python3 ~/.openclaw/workspace/skills/clawtop/scripts/tui.py --debug
```

## Panels

| Panel | Data source | Notes |
|-------|------------|-------|
| **SESSIONS** | `~/.openclaw/agents/*/sessions/sessions.json` | Token bars colour green/yellow/red at 60%/80% |
| **SYSTEM** | `psutil` (in-process) | CPU sampled in background thread — never blocks |
| **TOKENS** | `*.jsonl` transcript files | Aggregated across all sessions; handles all provider field-name variants |
| **MEMORY** | `workspace/MEMORY.md` + `workspace/memory/` | Shows flush status and whether a flush is overdue |
| **Status bar** | Same as above | Compact one-liner for skills + plugins counts |
| **LOGS** | `/tmp/openclaw*/openclaw-*.log` | Colour-coded by level; hidden when gateway offline |

## Architecture

**No Node.js spawned.** All data is read directly from the filesystem:

- Sessions → `~/.openclaw/agents/<id>/sessions/sessions.json`
- Token usage → `*.jsonl` transcript files (last 256 KB of each, up to 8 files)
- Memory → `workspace/MEMORY.md` and `workspace/memory/*.md` (file stats)
- Skills → `workspace/skills/*/SKILL.md` frontmatter scan
- Plugins → `~/.openclaw/openclaw.json` config
- Gateway → TCP probe on port 18789 (0.5s timeout)
- Logs → direct tail of `/tmp/openclaw-<uid>/openclaw-*.log`

Typical fetch time: **< 20ms**.

## Requirements

- Python 3.10+
- `openclaw` CLI in `$PATH`
- Gateway does **not** need to be running — all panels degrade gracefully when offline

## License

MIT — see [LICENSE](LICENSE).
