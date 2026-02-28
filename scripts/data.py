"""
ClawTop data layer — reads openclaw state files directly.

All data is sourced from the filesystem or a cheap TCP probe:
  sessions  → ~/.openclaw/agents/default/sessions/sessions.json
  skills    → ~/.openclaw/workspace/skills/*/SKILL.md  (YAML frontmatter)
  plugins   → ~/.openclaw/openclaw.json  (plugins config stanza)
  gateway   → TCP probe on port 18789
  logs      → /tmp/openclaw-<uid>/openclaw.log  (direct tail)
  system    → psutil (in-process, no subprocess)

No Node.js processes are spawned. Fetch time is typically <50ms.
"""

from __future__ import annotations

import json
import math
import os
import re
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class SessionInfo:
    key: str
    model: str
    total_tokens: Optional[int]
    context_tokens: Optional[int]
    percent_used: Optional[float]
    age: Optional[int]       # seconds since last activity
    kind: str                # direct | group | global | unknown


@dataclass
class SkillSummary:
    total: int
    eligible: int
    disabled: int
    missing_names: list[str]


@dataclass
class PluginEntry:
    name: str
    status: str
    version: Optional[str]
    error: Optional[str]


@dataclass
class PluginSummary:
    total: int
    loaded: int
    entries: list[PluginEntry]


@dataclass
class SystemMetrics:
    cpu_pct: float
    mem_pct: float
    mem_used_gb: float
    mem_total_gb: float
    disk_pct: float
    disk_used_gb: float
    disk_total_gb: float
    net_sent_bps: float
    net_recv_bps: float


@dataclass
class GatewayStatus:
    up: bool
    url: str
    default_model: Optional[str]




@dataclass
class TokenBreakdown:
    input_tokens:   int
    output_tokens:  int
    cache_read:     int
    cache_write:    int
    total_tokens:   int
    cache_hit_rate: float          # cacheRead / (input + cacheRead) * 100, or 0
    total_compactions: int         # sum of compactionCount across all sessions


@dataclass
class MemoryFile:
    name:     str
    size_kb:  float
    age_secs: Optional[int]        # seconds since last modified


@dataclass
class MemorySummary:
    memory_md:        Optional[MemoryFile]   # MEMORY.md / memory.md
    dir_files:        list[MemoryFile]       # memory/*.md, sorted newest-first (capped 4)
    dir_total_kb:     float
    last_flush_at:    Optional[int]          # seconds ago, from sessions.json
    last_flush_comp:  Optional[int]          # memoryFlushCompactionCount
    current_comp:     int                    # max compactionCount across sessions
    flush_overdue:    bool                   # compactions happened since last flush

@dataclass
class DashboardData:
    gateway: GatewayStatus
    sessions: list[SessionInfo]
    skills: SkillSummary
    plugins: PluginSummary
    system: SystemMetrics
    logs: list[str]
    tokens: TokenBreakdown
    memory: MemorySummary
    fetched_at: float = field(default_factory=time.time)
    errors: list[str] = field(default_factory=list)


# ── Path resolution ───────────────────────────────────────────────────────────

def _state_dir() -> Path:
    override = os.environ.get("OPENCLAW_STATE_DIR") or os.environ.get("CLAWDBOT_STATE_DIR")
    if override:
        return Path(override.strip()).expanduser().resolve()
    base = Path.home() / ".openclaw"
    if base.exists():
        return base
    for legacy in [".clawdbot", ".moldbot"]:
        p = Path.home() / legacy
        if p.exists():
            return p
    return base


def _config_path() -> Path:
    override = os.environ.get("OPENCLAW_CONFIG_PATH")
    if override:
        return Path(override.strip()).expanduser()
    sd = _state_dir()
    for name in ["openclaw.json", "clawdbot.json", "moldbot.json"]:
        p = sd / name
        if p.exists():
            return p
    return sd / "openclaw.json"


def _workspace_dir() -> Path:
    profile = os.environ.get("OPENCLAW_PROFILE", "").strip()
    if profile and profile.lower() != "default":
        return Path.home() / ".openclaw" / f"workspace-{profile}"
    override = os.environ.get("OPENCLAW_STATE_DIR")
    base = Path(override).expanduser() if override else Path.home() / ".openclaw"
    return base / "workspace"


def _session_store_path(agent_id: str = "default") -> Path:
    return _state_dir() / "agents" / agent_id / "sessions" / "sessions.json"


def _log_file() -> Optional[Path]:
    """Resolve the active openclaw log file."""
    # Explicit override in config
    cfg = _load_config()
    log_cfg = cfg.get("logging") or {}
    if log_cfg.get("file"):
        p = Path(str(log_cfg["file"])).expanduser()
        if p.exists():
            return p

    # Scan all /tmp/openclaw* dirs — covers openclaw-<uid>, openclaw, etc.
    tmp = Path("/tmp")
    candidate_dirs: list[Path] = sorted(tmp.glob("openclaw*"), reverse=True)

    # Prefer the uid-specific dir if it exists
    try:
        uid = os.getuid()
        uid_dir = tmp / f"openclaw-{uid}"
        if uid_dir in candidate_dirs:
            candidate_dirs.insert(0, uid_dir)
    except AttributeError:
        pass

    for log_dir in candidate_dirs:
        if not log_dir.is_dir():
            continue
        # Prefer newest date-stamped file, then fall back to openclaw.log
        dated = sorted(log_dir.glob("openclaw-*.log"), reverse=True)
        if dated:
            return dated[0]
        legacy = log_dir / "openclaw.log"
        if legacy.exists():
            return legacy

    return None


def _load_config() -> dict:
    p = _config_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ── Background CPU poller ─────────────────────────────────────────────────────

_cpu_lock  = threading.Lock()
_cpu_value = 0.0


def _cpu_poll_loop() -> None:
    global _cpu_value
    if not _PSUTIL:
        return
    psutil.cpu_percent()          # prime (discard first reading)
    while True:
        val = psutil.cpu_percent(interval=1.0)
        with _cpu_lock:
            _cpu_value = val


_cpu_thread = threading.Thread(target=_cpu_poll_loop, daemon=True)
_cpu_thread.start()


def _cpu_now() -> float:
    with _cpu_lock:
        return _cpu_value


# ── Context token lookup ──────────────────────────────────────────────────────

# Well-known context windows, kept intentionally minimal.
_CTX_MAP: dict[str, int] = {
    "claude-opus-4":        200_000,
    "claude-sonnet-4":      200_000,
    "claude-haiku-4":       200_000,
    "claude-3-7-sonnet":    200_000,
    "claude-3-5-sonnet":    200_000,
    "claude-3-5-haiku":     200_000,
    "claude-3-opus":        200_000,
    "gpt-4o":               128_000,
    "gpt-4-turbo":          128_000,
    "gpt-4":                  8_192,
    "gemini-1.5-pro":     1_000_000,
    "gemini-1.5-flash":   1_000_000,
    "gemini-2.0-flash":   1_000_000,
}
_DEFAULT_CTX = 200_000


def _resolve_model_str(model: Any) -> str:
    """Safely coerce any model value (str, dict, None) to a plain string."""
    if model is None:
        return "—"
    if isinstance(model, str):
        return model or "—"
    if isinstance(model, dict):
        # e.g. {'primary': 'blockrun/free'} — take the first string value
        for v in model.values():
            if isinstance(v, str) and v:
                return v
    return str(model) or "—"


def _context_tokens(model: str) -> int:
    if not model or model == "—":
        return _DEFAULT_CTX
    ml = model.lower()
    for key, val in _CTX_MAP.items():
        if key in ml:
            return val
    return _DEFAULT_CTX


# ── Session kind classifier ───────────────────────────────────────────────────

def _classify_key(key: str) -> str:
    """Mirror openclaw's classifySessionKey heuristic."""
    if not key:
        return "unknown"
    if key == "global":
        return "global"
    if "/" in key or "@" in key:
        return "group"
    return "direct"


# ── Individual fetchers (all file-based) ──────────────────────────────────────

def fetch_gateway() -> GatewayStatus:
    """TCP probe + config fallback — no subprocess."""
    cfg = _load_config()
    port: int = int(
        os.environ.get("OPENCLAW_GATEWAY_PORT")
        or (cfg.get("gateway") or {}).get("port")
        or 18789
    )
    raw_model = ((cfg.get("agents") or {}).get("defaults") or {}).get("model")
    default_model: Optional[str] = _resolve_model_str(raw_model) if raw_model else None

    url = f"localhost:{port}"
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return GatewayStatus(up=True, url=url, default_model=default_model)
    except OSError:
        return GatewayStatus(up=False, url=url, default_model=default_model)


def fetch_sessions() -> list[SessionInfo]:
    """Read sessions.json directly — no subprocess."""
    cfg = _load_config()
    default_model = _resolve_model_str(((cfg.get("agents") or {}).get("defaults") or {}).get("model"))
    default_ctx   = ((cfg.get("agents") or {}).get("defaults") or {}).get("contextTokens")

    # Collect session store paths across all agents
    agents_dir = _state_dir() / "agents"
    store_paths: list[Path] = []

    if agents_dir.exists():
        for agent_dir in sorted(agents_dir.iterdir()):
            if not agent_dir.is_dir():
                continue
            sp = agent_dir / "sessions" / "sessions.json"
            if sp.exists():
                store_paths.append(sp)

    # Also check legacy and alternative layouts
    for candidate in [
        _state_dir() / "sessions.json",                        # legacy flat layout
        _workspace_dir() / "sessions" / "sessions.json",       # workspace-relative
        _state_dir() / "sessions" / "sessions.json",           # alt flat layout
    ]:
        if candidate.exists() and candidate not in store_paths:
            store_paths.append(candidate)

    if not store_paths:
        return []

    rows: list[SessionInfo] = []
    now_ms = int(time.time() * 1000)

    for sp in store_paths:
        try:
            store: dict[str, Any] = json.loads(sp.read_text(encoding="utf-8"))
        except Exception:
            continue

        for key, entry in store.items():
            if not isinstance(entry, dict):
                continue

            # Resolve model: entry override > config default (both may be dicts)
            model = _resolve_model_str(entry.get("modelOverride") or default_model)

            total    = entry.get("totalTokens")
            ctx      = entry.get("contextTokens") or default_ctx or _context_tokens(model)
            pct: Optional[float] = None
            if isinstance(total, (int, float)) and ctx:
                pct = round(min(999, total / ctx * 100), 1)

            updated_at = entry.get("updatedAt")  # ms timestamp
            age: Optional[int] = None
            if isinstance(updated_at, (int, float)):
                age = max(0, int((now_ms - updated_at) / 1000))

            rows.append(SessionInfo(
                key           = str(key),
                model         = str(model),
                total_tokens  = total,
                context_tokens= ctx,
                percent_used  = pct,
                age           = age,
                kind          = _classify_key(key),
            ))

    # Sort by most recently active
    rows.sort(key=lambda r: r.age if r.age is not None else math.inf)
    return rows[:12]


def fetch_skills() -> SkillSummary:
    """Scan skills directories and parse SKILL.md frontmatter — no subprocess."""
    cfg = _load_config()
    skills_cfg = (cfg.get("skills") or {})
    disabled_set: set[str] = set(
        k for k, v in (skills_cfg.get("config") or {}).items()
        if isinstance(v, dict) and v.get("disabled")
    )
    # allowlist: if set, only listed skills are eligible
    allowlist_raw = skills_cfg.get("allowlist")
    allowlist: Optional[set[str]] = (
        set(allowlist_raw) if isinstance(allowlist_raw, list) else None
    )

    # Skills directories (workspace skills + managed ~/.openclaw/skills/)
    search_dirs: list[Path] = []
    workspace = _workspace_dir()
    for d in [
        workspace / "skills",
        _state_dir() / "skills",
        workspace / ".agents" / "skills",
        Path.home() / ".agents" / "skills",
    ]:
        if d.exists():
            search_dirs.append(d)

    total = eligible = disabled = 0
    missing_names: list[str] = []

    for skills_dir in search_dirs:
        for skill_dir in sorted(skills_dir.iterdir()):
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue

            name = _parse_skill_name(skill_md) or skill_dir.name
            total += 1

            if name in disabled_set:
                disabled += 1
                continue

            if allowlist is not None and name not in allowlist:
                # blocked by allowlist — count as non-eligible
                missing_names.append(name)
                continue

            # Check binary requirements from SKILL.md metadata
            reqs = _parse_skill_requires(skill_md)
            ok = all(_bin_exists(b) for b in reqs.get("bins", []))
            if ok:
                eligible += 1
            else:
                missing_names.append(name)

    return SkillSummary(
        total        = total,
        eligible     = eligible,
        disabled     = disabled,
        missing_names= missing_names[:5],
    )


def fetch_plugins() -> PluginSummary:
    """Read plugin records from openclaw.json — no subprocess."""
    cfg = _load_config()
    raw_plugins: list[dict] = []

    # plugins can live under cfg.plugins (list or dict)
    plugins_cfg = cfg.get("plugins")
    if isinstance(plugins_cfg, list):
        raw_plugins = plugins_cfg
    elif isinstance(plugins_cfg, dict):
        for k, v in plugins_cfg.items():
            if isinstance(v, dict):
                raw_plugins.append({"id": k, **v})

    loaded  = sum(1 for p in raw_plugins if not p.get("disabled"))
    entries = [
        PluginEntry(
            name    = str(p.get("name") or p.get("id") or "?"),
            status  = "disabled" if p.get("disabled") else "loaded",
            version = p.get("version"),
            error   = None,
        )
        for p in raw_plugins[:8]
    ]
    return PluginSummary(total=len(raw_plugins), loaded=loaded, entries=entries)


# Network rate tracking
_net_snapshot: dict = {}


def fetch_system() -> SystemMetrics:
    if not _PSUTIL:
        return SystemMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)

    global _net_snapshot
    mem  = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    net  = psutil.net_io_counters()
    now  = time.monotonic()

    sent_bps = recv_bps = 0.0
    if _net_snapshot:
        dt = now - _net_snapshot["t"]
        if dt > 0:
            sent_bps = max(0.0, (net.bytes_sent - _net_snapshot["s"]) / dt)
            recv_bps = max(0.0, (net.bytes_recv - _net_snapshot["r"]) / dt)
    _net_snapshot = {"t": now, "s": net.bytes_sent, "r": net.bytes_recv}

    return SystemMetrics(
        cpu_pct      = _cpu_now(),
        mem_pct      = mem.percent,
        mem_used_gb  = mem.used  / 1e9,
        mem_total_gb = mem.total / 1e9,
        disk_pct     = disk.percent,
        disk_used_gb = disk.used  / 1e9,
        disk_total_gb= disk.total / 1e9,
        net_sent_bps = sent_bps,
        net_recv_bps = recv_bps,
    )


def fetch_logs(limit: int = 22) -> list[str]:
    """Tail log file directly — no subprocess."""
    log_file = _log_file()
    if not log_file:
        return []
    try:
        # Efficient tail: read last ~32 KB
        size = log_file.stat().st_size
        chunk = min(size, 32_768)
        with open(log_file, "rb") as fh:
            fh.seek(max(0, size - chunk))
            raw = fh.read().decode("utf-8", errors="replace")
        lines = [l for l in raw.splitlines() if l.strip()]
        return lines[-limit:]
    except Exception:
        return []


# ── SKILL.md parsing helpers ──────────────────────────────────────────────────

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)


def _parse_skill_name(skill_md: Path) -> Optional[str]:
    try:
        text = skill_md.read_text(encoding="utf-8", errors="replace")[:2000]
    except Exception:
        return None
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return None
    for line in m.group(1).splitlines():
        if line.startswith("name:"):
            return line.split(":", 1)[1].strip().strip('"').strip("'") or None
    return None


def _parse_skill_requires(skill_md: Path) -> dict:
    """Extract requires.bins from SKILL.md openclaw metadata block."""
    try:
        text = skill_md.read_text(encoding="utf-8", errors="replace")[:3000]
    except Exception:
        return {}
    # Look for "requires": { "bins": [...] }  in the YAML/JSON metadata block
    bins_match = re.search(r'"bins"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if not bins_match:
        return {}
    bins_raw = bins_match.group(1)
    bins = re.findall(r'"([^"]+)"', bins_raw)
    return {"bins": bins}


def _bin_exists(name: str) -> bool:
    import shutil
    return shutil.which(name) is not None


# ── Parallel fetch_all ────────────────────────────────────────────────────────

def fetch_all(log_limit: int = 22, include_logs: bool = True) -> DashboardData:
    tasks = {
        "gateway" : fetch_gateway,
        "sessions": fetch_sessions,
        "skills"  : fetch_skills,
        "plugins" : fetch_plugins,
        "system"  : fetch_system,
        "tokens"  : fetch_token_breakdown,
        "memory"  : fetch_memory,
    }
    if include_logs:
        tasks["logs"] = lambda: fetch_logs(log_limit)

    results: dict = {}
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures = {pool.submit(fn): key for key, fn in tasks.items()}
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception:
                results[key] = None

    return DashboardData(
        gateway  = results.get("gateway")  or GatewayStatus(False, "localhost:18789", None),
        sessions = results.get("sessions") or [],
        skills   = results.get("skills")   or SkillSummary(0, 0, 0, []),
        plugins  = results.get("plugins")  or PluginSummary(0, 0, []),
        system   = results.get("system")   or SystemMetrics(0,0,0,0,0,0,0,0,0),
        logs     = results.get("logs")     or [],
        tokens   = results.get("tokens")   or TokenBreakdown(0,0,0,0,0,0.0,0),
        memory   = results.get("memory")   or MemorySummary(None,[],0.0,None,None,0,False),
    )


# ── Diagnostics ───────────────────────────────────────────────────────────────

def run_diagnostics() -> list[dict]:
    """
    Check each data source individually and report what was found.
    Used by --debug flag in tui.py.
    """
    import shutil
    report = []

    def _check(label: str, fn, detail_fn=None):
        t0 = time.monotonic()
        try:
            result = fn()
            elapsed = int((time.monotonic() - t0) * 1000)
            detail = detail_fn(result) if detail_fn else str(result)
            report.append({"label": label, "ok": True,
                           "elapsed_ms": elapsed, "detail": detail[:120], "error": ""})
        except Exception as e:
            elapsed = int((time.monotonic() - t0) * 1000)
            report.append({"label": label, "ok": False,
                           "elapsed_ms": elapsed, "detail": "", "error": str(e)[:120]})

    _check("openclaw binary",
           lambda: shutil.which("openclaw") or (_ for _ in ()).throw(FileNotFoundError("not in PATH")),
           lambda r: str(r))
    _check("config file",
           lambda: _config_path(),
           lambda r: str(r) + (" ✓" if r.exists() else " ✗ not found"))
    def _find_session_stores():
        found = []
        agents_dir = _state_dir() / "agents"
        if agents_dir.exists():
            for ad in sorted(agents_dir.iterdir()):
                sp = ad / "sessions" / "sessions.json"
                if sp.exists():
                    found.append(sp)
        for cand in [_state_dir() / "sessions.json",
                     _state_dir() / "sessions" / "sessions.json"]:
            if cand.exists() and cand not in found:
                found.append(cand)
        if not found:
            raise FileNotFoundError("no sessions.json found in any agent dir")
        return found
    _check("session store",
           _find_session_stores,
           lambda r: "  ".join(str(p) for p in r))
    _check("workspace dir",
           lambda: _workspace_dir(),
           lambda r: str(r) + (" ✓" if r.exists() else " ✗ not found"))
    _check("log file",
           lambda: _log_file() or (_ for _ in ()).throw(FileNotFoundError("no log file found")),
           lambda r: str(r))
    _check("gateway TCP",
           lambda: fetch_gateway(),
           lambda r: f"{'UP' if r.up else 'DOWN'}  {r.url}  model={r.default_model}")
    _check("sessions",
           lambda: fetch_sessions(),
           lambda r: f"{len(r)} session(s) found")
    _check("skills",
           lambda: fetch_skills(),
           lambda r: f"{r.total} total  {r.eligible} eligible  {r.disabled} disabled")
    _check("plugins",
           lambda: fetch_plugins(),
           lambda r: f"{r.total} total  {r.loaded} loaded")
    _check("logs",
           lambda: fetch_logs(5),
           lambda r: f"{len(r)} line(s)" + (f"  last: {r[-1][:60]}" if r else ""))
    _check("tokens",
           lambda: fetch_token_breakdown(),
           lambda r: (
               f"total={r.total_tokens:,}  in={r.input_tokens:,}  out={r.output_tokens:,}  "
               f"cache_read={r.cache_read:,}  hit={r.cache_hit_rate}%  compactions={r.total_compactions}"
           ))

    return report


# ── Token breakdown fetcher ────────────────────────────────────────────────────

def _normalize_usage(u: dict) -> tuple[int,int,int,int]:
    """Extract (input, output, cache_read, cache_write) from any usage dict shape."""
    def _n(v): return int(v) if isinstance(v, (int, float)) and v == v else 0
    inp = _n(u.get("input") or u.get("inputTokens") or u.get("input_tokens")
             or u.get("promptTokens") or u.get("prompt_tokens") or 0)
    out = _n(u.get("output") or u.get("outputTokens") or u.get("output_tokens")
             or u.get("completionTokens") or u.get("completion_tokens") or 0)
    cr  = _n(u.get("cacheRead") or u.get("cache_read") or u.get("cache_read_input_tokens") or 0)
    cw  = _n(u.get("cacheWrite") or u.get("cache_write")
             or u.get("cache_creation_input_tokens") or 0)
    return inp, out, cr, cw


def _scan_jsonl_for_tokens(jsonl_path: Path) -> tuple[int,int,int,int,int]:
    """
    Scan a transcript JSONL for cumulative token usage.
    Returns (input, output, cache_read, cache_write, total_fallback).
    Only counts assistant-role entries to avoid double-counting.
    """
    inp = out = cr = cw = total_fb = 0
    try:
        size = jsonl_path.stat().st_size
        # Read last 256 KB — enough to capture recent turns without full scan
        chunk = min(size, 262_144)
        with open(jsonl_path, "rb") as fh:
            fh.seek(max(0, size - chunk))
            raw = fh.read().decode("utf-8", errors="replace")
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            if not isinstance(entry, dict):
                continue
            # Usage can be at top level or inside message{}
            msg = entry.get("message") if isinstance(entry.get("message"), dict) else {}
            role = msg.get("role") or entry.get("role")
            if role != "assistant":
                continue
            usage_raw = msg.get("usage") or entry.get("usage")
            if isinstance(usage_raw, dict):
                ei, eo, ec, ew = _normalize_usage(usage_raw)
                inp += ei; out += eo; cr += ec; cw += ew
                # total fallback: some providers only emit total
                if ei == 0 and eo == 0:
                    t = usage_raw.get("total") or usage_raw.get("totalTokens") or usage_raw.get("total_tokens")
                    if isinstance(t, (int, float)):
                        total_fb += int(t)
    except Exception:
        pass
    return inp, out, cr, cw, total_fb


def fetch_token_breakdown() -> TokenBreakdown:
    """
    Aggregate token counts from JSONL transcripts (primary) with
    sessions.json compactionCount as supplement.
    """
    # Collect all agent session directories
    agents_dir = _state_dir() / "agents"
    session_dirs: list[Path] = []
    if agents_dir.exists():
        for ad in sorted(agents_dir.iterdir()):
            if ad.is_dir() and (ad / "sessions").is_dir():
                session_dirs.append(ad / "sessions")

    # Collect compactionCount from sessions.json files
    total_comp = 0
    for sd in session_dirs:
        sp = sd / "sessions.json"
        if not sp.exists():
            continue
        try:
            store = json.loads(sp.read_text(encoding="utf-8"))
            for entry in store.values():
                if isinstance(entry, dict):
                    total_comp += entry.get("compactionCount", 0) or 0
        except Exception:
            pass

    # Scan JSONL transcripts for token data (last 5 files by mtime = most active)
    inp = out = cr = cw = total_fb = 0
    jsonl_files: list[Path] = []
    for sd in session_dirs:
        jsonl_files.extend(sd.glob("*.jsonl"))
    # Sort by mtime descending, scan up to 8 most recent
    jsonl_files.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    for jf in jsonl_files[:8]:
        ei, eo, ec, ew, efb = _scan_jsonl_for_tokens(jf)
        inp += ei; out += eo; cr += ec; cw += ew; total_fb += efb

    total = inp + out + cr + cw
    if total == 0 and total_fb > 0:
        inp = total_fb
        total = total_fb

    denom = inp + cr
    hit_rate = round(cr / denom * 100, 1) if denom > 0 else 0.0

    return TokenBreakdown(
        input_tokens      = inp,
        output_tokens     = out,
        cache_read        = cr,
        cache_write       = cw,
        total_tokens      = total,
        cache_hit_rate    = hit_rate,
        total_compactions = total_comp,
    )


# ── Memory fetcher ─────────────────────────────────────────────────────────────

def fetch_memory() -> MemorySummary:
    """
    Stat memory files in the workspace and extract flush metadata from sessions.
    """
    workspace = _workspace_dir()
    now = time.time()

    def _file_stat(p: Path) -> Optional[MemoryFile]:
        try:
            st = p.stat()
            return MemoryFile(
                name     = p.name,
                size_kb  = st.st_size / 1024,
                age_secs = int(now - st.st_mtime),
            )
        except OSError:
            return None

    # MEMORY.md (case-insensitive fallback)
    memory_md: Optional[MemoryFile] = None
    for name in ["MEMORY.md", "memory.md"]:
        f = _file_stat(workspace / name)
        if f:
            memory_md = f
            break

    # memory/ directory
    memory_dir = workspace / "memory"
    dir_files: list[MemoryFile] = []
    dir_total_kb = 0.0
    if memory_dir.is_dir():
        for p in sorted(memory_dir.glob("*.md"), key=lambda x: x.stat().st_mtime, reverse=True):
            mf = _file_stat(p)
            if mf:
                dir_total_kb += mf.size_kb
                dir_files.append(mf)
        # include non-md files in size total
        for p in memory_dir.iterdir():
            if not p.suffix == ".md":
                try:
                    dir_total_kb += p.stat().st_size / 1024
                except OSError:
                    pass

    # Flush metadata from sessions.json
    last_flush_ms:   Optional[int] = None
    last_flush_comp: Optional[int] = None
    max_comp = 0

    agents_dir = _state_dir() / "agents"
    store_paths: list[Path] = []
    if agents_dir.exists():
        for ad in agents_dir.iterdir():
            if not ad.is_dir():
                continue
            sp = ad / "sessions" / "sessions.json"
            if sp.exists():
                store_paths.append(sp)
    for cand in [_state_dir() / "sessions.json"]:
        if cand.exists() and cand not in store_paths:
            store_paths.append(cand)

    for sp in store_paths:
        try:
            store: dict = json.loads(sp.read_text(encoding="utf-8"))
        except Exception:
            continue
        for entry in store.values():
            if not isinstance(entry, dict):
                continue
            flush_at = entry.get("memoryFlushAt")
            if isinstance(flush_at, (int, float)):
                if last_flush_ms is None or flush_at > last_flush_ms:
                    last_flush_ms  = int(flush_at)
                    last_flush_comp = entry.get("memoryFlushCompactionCount")
            comp = entry.get("compactionCount")
            if isinstance(comp, int):
                max_comp = max(max_comp, comp)

    last_flush_secs: Optional[int] = None
    if last_flush_ms is not None:
        last_flush_secs = max(0, int(now - last_flush_ms / 1000))

    flush_overdue = (
        last_flush_comp is not None
        and max_comp > last_flush_comp
    )

    return MemorySummary(
        memory_md       = memory_md,
        dir_files       = dir_files[:4],   # cap for display
        dir_total_kb    = dir_total_kb,
        last_flush_at   = last_flush_secs,
        last_flush_comp = last_flush_comp,
        current_comp    = max_comp,
        flush_overdue   = flush_overdue,
    )
