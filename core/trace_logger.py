"""
Lightweight pipeline tracer — independent from Langfuse.

Each query run gets a unique trace_id (same as session_id + timestamp).
Events are buffered in memory while running, then persisted to
logs/traces.jsonl when the trace is finished.

SSE clients (browser) subscribe to a trace_id and receive events
as they are logged — enabling real-time pipeline visualization.
"""

import json
import time
import threading
from pathlib import Path
from dataclasses import dataclass, field, asdict

import numpy as np


class _NumpySafeEncoder(json.JSONEncoder):
    """Handle numpy types that slip through to json.dumps."""
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

TRACES_FILE = Path(__file__).parent.parent / "logs" / "traces.jsonl"
TRACES_FILE.parent.mkdir(parents=True, exist_ok=True)

# Max lines before truncation
_MAX_LINES = 5000
_KEEP_LINES = 1000

# In-memory store: trace_id -> {meta, events, subscribers}
_store: dict[str, dict] = {}
_lock = threading.Lock()


# ──────────────────────────────────────────────
# Event schema
# ──────────────────────────────────────────────

@dataclass
class TraceEvent:
    node: str                    # e.g. "IntentGuard", "Orchestrator"
    type: str                    # "start" | "end" | "llm" | "decision" | "error" | "info"
    ts: float = field(default_factory=time.time)
    duration_ms: float = 0.0     # filled on "end" events
    data: dict = field(default_factory=dict)  # arbitrary payload


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def start_trace(trace_id: str, query: str, user_id: str = "", platform: str = "") -> None:
    """Create a new in-memory trace entry."""
    with _lock:
        _store[trace_id] = {
            "trace_id": trace_id,
            "query": query,
            "user_id": user_id,
            "platform": platform,
            "started_at": time.time(),
            "finished_at": None,
            "total_ms": None,
            "answer": "",
            "is_fallback": False,
            "events": [],
            "subscribers": [],   # list of asyncio.Queue for SSE
        }


def log_event(trace_id: str, node: str, type: str, data: dict = None,
              duration_ms: float = 0.0) -> None:
    """Append an event to the trace and notify SSE subscribers."""
    if trace_id not in _store:
        return
    event = TraceEvent(
        node=node,
        type=type,
        duration_ms=duration_ms,
        data=data or {},
    )
    with _lock:
        entry = _store[trace_id]
        entry["events"].append(asdict(event))
        # Notify all SSE subscribers
        for q in entry["subscribers"]:
            try:
                q.put_nowait(asdict(event))
            except Exception:
                pass


def finish_trace(trace_id: str, answer: str, is_fallback: bool = False) -> None:
    """Mark trace as complete, persist to logs/traces.jsonl."""
    if trace_id not in _store:
        return
    with _lock:
        entry = _store[trace_id]
        entry["finished_at"] = time.time()
        entry["total_ms"] = round((entry["finished_at"] - entry["started_at"]) * 1000, 1)
        entry["answer"] = answer
        entry["is_fallback"] = is_fallback
        # Notify subscribers of completion
        for q in entry["subscribers"]:
            try:
                q.put_nowait({"node": "__done__", "type": "done",
                              "data": {"total_ms": entry["total_ms"]}})
            except Exception:
                pass
    # Persist (without subscribers list)
    record = {k: v for k, v in _store[trace_id].items() if k != "subscribers"}
    with open(TRACES_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, cls=_NumpySafeEncoder) + "\n")

    # Truncate file if too large
    _maybe_truncate()

    # Clean up in-memory entry after a delay (keep for 60s for late SSE subscribers)
    def _cleanup():
        time.sleep(60)
        with _lock:
            _store.pop(trace_id, None)
    cleanup_thread = threading.Thread(target=_cleanup, daemon=True)
    cleanup_thread.start()


def get_trace(trace_id: str) -> dict | None:
    """Return in-memory trace (running or finished) or load from file."""
    if trace_id in _store:
        entry = dict(_store[trace_id])
        entry.pop("subscribers", None)
        return entry
    # Fall back to file search
    if TRACES_FILE.exists():
        for line in reversed(TRACES_FILE.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if record.get("trace_id") == trace_id:
                    return record
            except Exception:
                continue
    return None


def list_traces(limit: int = 50) -> list[dict]:
    """Return recent traces (summary only, no events list)."""
    results = []
    if TRACES_FILE.exists():
        lines = [l for l in TRACES_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
        for line in reversed(lines[-200:]):
            try:
                record = json.loads(line)
                results.append({
                    "trace_id": record["trace_id"],
                    "query": record["query"][:80],
                    "user_id": record.get("user_id", ""),
                    "platform": record.get("platform", ""),
                    "started_at": record["started_at"],
                    "total_ms": record.get("total_ms"),
                    "is_fallback": record.get("is_fallback", False),
                    "node_count": len(record.get("events", [])),
                })
            except Exception:
                continue
    # Also include active in-memory traces
    with _lock:
        for tid, entry in _store.items():
            if not entry.get("finished_at"):
                results.insert(0, {
                    "trace_id": tid,
                    "query": entry["query"][:80],
                    "user_id": entry.get("user_id", ""),
                    "platform": entry.get("platform", ""),
                    "started_at": entry["started_at"],
                    "total_ms": None,
                    "is_fallback": False,
                    "node_count": len(entry["events"]),
                    "running": True,
                })
    return results[:limit]


def subscribe_sse(trace_id: str):
    """Return an asyncio.Queue that receives events for this trace.
    If the trace is already finished or not found, returns None.
    """
    import asyncio
    if trace_id not in _store:
        return None
    if _store[trace_id].get("finished_at"):
        return None
    q = asyncio.Queue()
    with _lock:
        _store[trace_id]["subscribers"].append(q)
    return q


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

def _maybe_truncate() -> None:
    """Truncate traces.jsonl to last _KEEP_LINES when it exceeds _MAX_LINES."""
    try:
        if not TRACES_FILE.exists():
            return
        lines = TRACES_FILE.read_text(encoding="utf-8").splitlines()
        if len(lines) > _MAX_LINES:
            keep = lines[-_KEEP_LINES:]
            TRACES_FILE.write_text("\n".join(keep) + "\n", encoding="utf-8")
            print(f"[TRACE_LOGGER] Truncated traces.jsonl: {len(lines)} → {_KEEP_LINES} lines")
    except Exception as e:
        print(f"[TRACE_LOGGER] Truncation error: {e}")


if __name__ == "__main__":
    print("=== trace_logger.py standalone test ===\n")

    # Test basic flow
    tid = "test-001"
    start_trace(tid, "in bảng kê không ra", user_id="dr_nguyen", platform="telegram")

    log_event(tid, "IntentGuard", "decision", {"result": True, "duration_ms": 210.5})
    log_event(tid, "FastRetriever", "end", {"chunks": [{"subject": "In bảng kê", "score": 0.71}], "duration_ms": 380.2})
    log_event(tid, "Orchestrator", "decision", {"action": "answer", "tool": "search_faq", "reasoning": "clear query"})
    log_event(tid, "Generator", "end", {"prompt_chars": 3240, "answer_chars": 412, "duration_ms": 2100.0})

    finish_trace(tid, "Bạn vào menu In → chọn Bảng kê...", is_fallback=False)

    # Verify
    trace = get_trace(tid)
    print(f"Trace ID: {trace['trace_id']}")
    print(f"Query: {trace['query']}")
    print(f"Total: {trace['total_ms']}ms")
    print(f"Events: {len(trace['events'])}")
    for e in trace["events"]:
        print(f"  [{e['node']}] {e['type']} — {e.get('duration_ms', 0):.0f}ms")

    # Test list
    traces = list_traces()
    print(f"\nList traces: {len(traces)} entries")

    print("\n✓ trace_logger.py works correctly.")
