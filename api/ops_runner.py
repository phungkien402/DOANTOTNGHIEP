"""
Background job runner for admin operations.
Runs subprocesses, captures output line by line, streams to SSE subscribers.
"""

import asyncio
import subprocess
import threading
import time
import uuid
from pathlib import Path

# In-memory jobs: job_id -> {status, lines, subscribers, started_at, finished_at}
_jobs: dict[str, dict] = {}
_lock = threading.Lock()

PROJECT_DIR = Path(__file__).parent.parent


def _run_job(job_id: str, cmd: list[str], cwd: str | None = None) -> None:
    """Execute cmd in background, stream output to subscribers."""
    entry = _jobs[job_id]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=cwd or str(PROJECT_DIR),
        )
        for line in proc.stdout:
            line = line.rstrip()
            with _lock:
                entry["lines"].append(line)
                for q in entry["subscribers"]:
                    try:
                        q.put_nowait({"type": "log", "text": line})
                    except Exception:
                        pass
        proc.wait()
        status = "ok" if proc.returncode == 0 else f"error (exit {proc.returncode})"
    except Exception as e:
        status = f"error: {e}"

    with _lock:
        entry["status"] = status
        entry["finished_at"] = time.time()
        for q in entry["subscribers"]:
            try:
                q.put_nowait({"type": "done", "status": status})
            except Exception:
                pass


def start_job(cmd: list[str], cwd: str | None = None) -> str:
    """Create a job and start it in a background thread. Returns job_id."""
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {
        "job_id": job_id,
        "cmd": " ".join(cmd),
        "status": "running",
        "lines": [],
        "subscribers": [],
        "started_at": time.time(),
        "finished_at": None,
    }
    t = threading.Thread(target=_run_job, args=(job_id, cmd, cwd), daemon=True)
    t.start()
    return job_id


def get_job(job_id: str) -> dict | None:
    """Get job info (without subscribers list)."""
    entry = _jobs.get(job_id)
    if not entry:
        return None
    return {k: v for k, v in entry.items() if k != "subscribers"}


def subscribe_job(job_id: str) -> asyncio.Queue | None:
    """Return an asyncio.Queue that receives log + done events."""
    if job_id not in _jobs:
        return None
    q: asyncio.Queue = asyncio.Queue()
    with _lock:
        _jobs[job_id]["subscribers"].append(q)
    return q
