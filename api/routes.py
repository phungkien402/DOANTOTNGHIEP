"""
FastAPI application routes.

POST /webhook/{platform}  — receive messages from Zalo / Telegram / Web
GET  /health              — health check
GET  /admin/logs          — view unanswered / fallback query logs
POST /admin/reindex       — trigger a fresh data pull from Redmine

Run: uvicorn api.routes:app --host 0.0.0.0 --port 8080
"""

import os
import sys
import json
import time
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from redis import Redis
from rq import Queue

from config import SESSION_MAX_TURNS, ADMIN_TOKEN
from core.models import Message
from core.langgraph_agent import run as run_pipeline, set_maintenance_mode, is_maintenance_mode, set_session_manager as set_agent_session_mgr
from api.session import SessionManager
from api.logger import QueryLogger
from api.ops_runner import start_job, get_job, subscribe_job
from adapters.telegram_adapter import TelegramAdapter
from adapters.telegram_adapter import set_session_manager as set_telegram_session_mgr
from adapters.zalo_adapter import ZaloAdapter
from adapters.web_adapter import WebAdapter
from adapters.slack_adapter import SlackAdapter

app = FastAPI(title="EHC AI Helpdesk")

# CORS for web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared instances
_session_mgr = SessionManager(max_turns=SESSION_MAX_TURNS, ttl_seconds=1800)
set_telegram_session_mgr(_session_mgr)
set_agent_session_mgr(_session_mgr)
_logger = QueryLogger()

# Adapter registry
_adapters = {
    "telegram": TelegramAdapter(),
    "zalo": ZaloAdapter(),
    "web": WebAdapter(),
    "slack": SlackAdapter(),
}

# Redis Queue for async Telegram processing
_redis_conn = Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
_queue = Queue("ehc-queue", connection=_redis_conn)

# Slack event deduplication
_processed_slack_events: set[str] = set()


@app.get("/health")
def health():
    """Health check endpoint."""
    from datetime import datetime, timezone
    return {"status": "ok", "service": "ehc-helpdesk", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/")
def serve_ui():
    """Serve the web chat UI."""
    ui_path = Path(__file__).parent.parent / "ui" / "index.html"
    if ui_path.exists():
        return FileResponse(str(ui_path))
    return JSONResponse({"error": "UI not found"}, status_code=404)


@app.post("/webhook/{platform}")
async def handle_webhook(platform: str, request: Request, background_tasks: BackgroundTasks):
    """
    Unified webhook handler for all platforms.
    Selects the appropriate adapter, parses the message, runs the pipeline,
    logs the query, and sends the response back.
    """
    # Validate platform
    adapter = _adapters.get(platform)
    if adapter is None:
        raise HTTPException(status_code=400, detail=f"Unknown platform: {platform}")

    # --- Slack slash commands (form-encoded) ---
    content_type = request.headers.get("content-type", "")
    if platform == "slack" and "application/x-www-form-urlencoded" in content_type:
        form = await request.form()
        form_data = dict(form)
        # Slash commands have a "command" field
        if "command" in form_data:
            response_text = await adapter.handle_slash_command(form_data)
            return JSONResponse(
                content={"response_type": "ephemeral", "text": response_text},
                status_code=200,
            )

    # Parse raw payload
    try:
        raw = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # Handle Slack URL verification challenge
    if platform == "slack" and raw.get("type") == "url_verification":
        return {"challenge": raw.get("challenge")}

    # Slack deduplication — ignore retried events
    if platform == "slack":
        event_id = raw.get("event_id", "")
        if event_id and event_id in _processed_slack_events:
            return {"status": "duplicate"}
        if event_id:
            _processed_slack_events.add(event_id)
            if len(_processed_slack_events) > 1000:
                _processed_slack_events.clear()

    # Parse into Message
    message = adapter.parse_message(raw)
    if message is None:
        # Non-actionable event (delivery receipt, typing, etc.) — acknowledge
        return {"status": "ignored"}

    # Get session history
    session_history = _session_mgr.get_history(message.session_id)

    # --- Telegram: enqueue to Redis Queue and return immediately ---
    if platform == "telegram":
        chat_id = message.session_id.replace("tg_", "")
        _queue.enqueue(
            "workers.pipeline_worker.process_telegram_query",
            chat_id=chat_id,
            text=message.text,
            session_id=message.session_id,
            history=session_history,
        )
        print(f"[WEBHOOK] Enqueued | chat_id={chat_id} | query=\"{message.text}\"")
        return {"ok": True}
        
    if platform == "slack":
        thread_ts = adapter.pop_thread_ts(message.session_id)
        parts = message.session_id.replace("slack_", "").split("_")
        channel_id = parts[0]
        _queue.enqueue(
            "workers.pipeline_worker.process_slack_query",
            session_id=message.session_id,
            channel_id=channel_id,
            text=message.text,
            thread_ts=thread_ts or "",
            history=session_history,
        )
        print(f"[WEBHOOK] Enqueued Slack | channel={channel_id} | query=\"{message.text}\"")
        return {"ok": True}

    # --- Other platforms: synchronous processing ---
    t0 = time.time()
    answer = run_pipeline(message, session_history)
    latency_ms = (time.time() - t0) * 1000

    # Store turns in session
    _session_mgr.add_turn(message.session_id, "user", message.text)
    _session_mgr.add_turn(message.session_id, "bot", answer.text)

    # Log the query
    _logger.log(message, answer, latency_ms=latency_ms)

    # Format response for platform
    response_text = adapter.format_response(
        answer.text,
        confidence=0.0 if answer.is_fallback else answer.confidence,
    )

    # Send response back via platform API (async, in background for Zalo/Slack)
    if platform != "web":
        chat_id = message.user_id
        if platform == "slack":
            chat_id = message.session_id
        background_tasks.add_task(adapter.send_message, chat_id, response_text)

    # Return response (used directly by web adapter)
    return {
        "status": "ok",
        "answer": answer.text,
        "confidence": answer.confidence,
        "is_fallback": answer.is_fallback,
        "rewritten_question": answer.rewritten_question,
        "sources": [
            {
                "subject": c.metadata.get("subject", ""),
                "score": round(c.score, 4),
                "url": c.metadata.get("url", ""),
            }
            for c in answer.source_chunks
        ],
    }


@app.get("/admin/logs")
async def get_logs(limit: int = 50, fallback_only: bool = False):
    """Return query logs as JSON. Optionally filter to fallback-only."""
    logs = _logger.read_logs(limit=limit, fallback_only=fallback_only)
    return {"count": len(logs), "logs": logs}


@app.post("/admin/reindex")
async def trigger_reindex(background_tasks: BackgroundTasks):
    """Trigger a full reindex from Redmine (runs in background)."""
    from data.reindex import full_reindex

    background_tasks.add_task(full_reindex)
    return {"status": "reindex_started", "message": "Full reindex triggered in background."}


@app.post("/admin/maintenance")
async def toggle_maintenance(request: Request):
    """Toggle maintenance mode at runtime. Requires ADMIN_TOKEN."""
    # Auth check
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=500, detail="ADMIN_TOKEN not configured")

    auth_header = request.headers.get("Authorization", "")
    token = auth_header.replace("Bearer ", "").strip()
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid admin token")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    enabled = body.get("enabled")
    if not isinstance(enabled, bool):
        raise HTTPException(status_code=400, detail="Body must contain {\"enabled\": true/false}")

    set_maintenance_mode(enabled)
    return {
        "status": "ok",
        "maintenance_mode": is_maintenance_mode(),
        "message": f"Maintenance mode {'enabled' if enabled else 'disabled'}.",
    }


@app.get("/tickets")
async def get_tickets():
    """Return all tickets from data/tickets.db."""
    from core.tools.create_ticket import list_tickets
    tickets = list_tickets()
    return {"count": len(tickets), "tickets": tickets}


@app.get("/unanswered")
async def list_unanswered():
    """Return all entries from data/unanswered.jsonl (newest first)."""
    import json as _json
    path = Path(__file__).parent.parent / "data" / "unanswered.jsonl"
    if not path.exists():
        return []
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(_json.loads(line))
                except _json.JSONDecodeError:
                    pass
    return list(reversed(entries))


# --- Trace endpoints ---

@app.get("/traces")
async def list_traces():
    """Return recent run traces (last 50)."""
    from core import tracer
    return tracer.get_all()


@app.get("/traces/{run_id}")
async def get_trace(run_id: str):
    """Return a single run trace by ID."""
    from core import tracer
    r = tracer.get_one(run_id)
    if not r:
        raise HTTPException(status_code=404, detail="Run not found")
    return r


@app.get("/traces-ui")
async def traces_ui():
    """Serve the traces dashboard."""
    ui_path = Path(__file__).parent.parent / "static" / "traces.html"
    if ui_path.exists():
        return FileResponse(str(ui_path))
    return JSONResponse({"error": "traces.html not found"}, status_code=404)


# ---------------------------------------------------------------------------
# rtk: Admin Monitoring Dashboard
# ---------------------------------------------------------------------------


@app.get("/admin/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the admin monitoring dashboard."""
    html_path = Path(__file__).parent.parent / "ui" / "dashboard.html"
    if html_path.exists():
        html = html_path.read_text(encoding="utf-8")
        token = os.getenv("ADMIN_TOKEN", "")
        html = html.replace("__ADMIN_TOKEN__", token)
        return HTMLResponse(content=html)
    return HTMLResponse(content="<h1>dashboard.html not found</h1>", status_code=404)


@app.get("/admin/stats/queries")
async def stats_queries(days: int = 7):
    """Aggregated query statistics from logs/queries.jsonl."""
    import json as _json
    from datetime import datetime, timedelta, timezone
    from collections import defaultdict

    log_path = Path(__file__).parent.parent / "logs" / "queries.jsonl"
    unanswered_path = Path(__file__).parent.parent / "data" / "unanswered.jsonl"

    now = datetime.now(timezone.utc)
    cutoff_ts = (now - timedelta(days=days)).timestamp()

    total_queries = 0
    total_fallbacks = 0
    confidence_sum = 0.0
    latency_sum = 0.0
    latency_count = 0
    by_day: dict[str, dict] = defaultdict(lambda: {"count": 0, "fallbacks": 0})
    by_platform: dict[str, int] = defaultdict(int)

    if log_path.exists():
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                ts = entry.get("timestamp", 0)
                if ts < cutoff_ts:
                    continue
                total_queries += 1
                if entry.get("is_fallback"):
                    total_fallbacks += 1
                confidence_sum += entry.get("confidence", 0.0)
                lat = entry.get("latency_ms", 0.0)
                if lat > 0:
                    latency_sum += lat
                    latency_count += 1
                day_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
                by_day[day_str]["count"] += 1
                if entry.get("is_fallback"):
                    by_day[day_str]["fallbacks"] += 1
                platform = entry.get("platform", "unknown")
                by_platform[platform] += 1

    # Sort by_day
    sorted_days = sorted(by_day.items())
    by_day_list = [{"date": d, "count": v["count"], "fallbacks": v["fallbacks"]} for d, v in sorted_days]

    # Top unanswered
    top_unanswered: list[dict] = []
    if unanswered_path.exists():
        question_counts: dict[str, dict] = {}
        with open(unanswered_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                q = entry.get("question", "").strip()
                if not q:
                    continue
                ts = entry.get("timestamp", 0)
                last_seen = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d") if ts else ""
                if q in question_counts:
                    question_counts[q]["count"] += 1
                    if last_seen > question_counts[q]["last_seen"]:
                        question_counts[q]["last_seen"] = last_seen
                else:
                    question_counts[q] = {"question": q, "count": 1, "last_seen": last_seen}
        top_unanswered = sorted(question_counts.values(), key=lambda x: x["count"], reverse=True)[:10]

    fallback_rate = (total_fallbacks / total_queries) if total_queries > 0 else 0.0
    avg_confidence = (confidence_sum / total_queries) if total_queries > 0 else 0.0
    avg_latency_ms = (latency_sum / latency_count) if latency_count > 0 else 0.0

    return {
        "total_queries": total_queries,
        "total_fallbacks": total_fallbacks,
        "fallback_rate": round(fallback_rate, 4),
        "avg_confidence": round(avg_confidence, 4),
        "avg_latency_ms": round(avg_latency_ms, 1),
        "by_day": by_day_list,
        "by_platform": dict(by_platform),
        "top_unanswered": top_unanswered,
    }


@app.get("/admin/stats/health")
async def stats_health():
    """Check all service dependencies with per-service timeouts."""
    import asyncio
    import httpx

    async def check_service(name: str, url: str) -> tuple[str, str]:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(url)
                return (name, "ok" if resp.status_code < 500 else "error")
        except httpx.TimeoutException:
            return (name, "timeout")
        except Exception:
            return (name, "error")

    async def check_redis() -> tuple[str, str]:
        try:
            pong = _redis_conn.ping()
            return ("redis", "ok" if pong else "error")
        except Exception:
            return ("redis", "error")

    results = await asyncio.gather(
        check_service("fastapi", "http://localhost:8001/health"),
        check_service("qdrant", "http://localhost:6333/healthz"),
        check_service("vllm", "http://localhost:8000/health"),
        check_redis(),
        return_exceptions=True,
    )

    health_map: dict[str, str] = {}
    for r in results:
        if isinstance(r, tuple):
            health_map[r[0]] = r[1]
        else:
            # Exception from gather
            health_map["unknown"] = "error"

    health_map["timestamp"] = time.time()  # type: ignore[assignment]
    return health_map


@app.get("/admin/stats/resources")
async def stats_resources():
    """System resource snapshot: CPU, RAM, Disk, GPU."""
    import psutil

    cpu_percent = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    gpu_list: list[dict] = []
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_list.append({
                "index": i,
                "name": name,
                "vram_used_mb": round(mem_info.used / (1024 ** 2)),
                "vram_total_mb": round(mem_info.total / (1024 ** 2)),
                "vram_percent": round(mem_info.used / mem_info.total * 100, 1),
                "temperature_c": temp,
                "utilization_percent": util.gpu,
            })
        pynvml.nvmlShutdown()
    except Exception:
        gpu_list = []

    return {
        "cpu_percent": cpu_percent,
        "ram_used_gb": round(mem.used / (1024 ** 3), 1),
        "ram_total_gb": round(mem.total / (1024 ** 3), 1),
        "ram_percent": mem.percent,
        "disk_used_gb": round(disk.used / (1024 ** 3), 1),
        "disk_total_gb": round(disk.total / (1024 ** 3), 1),
        "disk_percent": round(disk.used / disk.total * 100, 1),
        "gpu": gpu_list,
    }


@app.get("/admin/stats/tickets")
async def stats_tickets():
    """Ticket summary from SQLite."""
    import sqlite3

    db_path = Path(__file__).parent.parent / "data" / "tickets.db"
    result = {"total": 0, "pending": 0, "pushed_to_redmine": 0, "recent": []}

    if not db_path.exists():
        return result

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Total counts
        cur.execute("SELECT COUNT(*) FROM tickets")
        result["total"] = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM tickets WHERE status = 'open'")
        result["pending"] = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM tickets WHERE status = 'pushed'")
        result["pushed_to_redmine"] = cur.fetchone()[0]

        # Recent 20
        cur.execute(
            "SELECT id, assigned_to, query, timestamp, status "
            "FROM tickets ORDER BY timestamp DESC LIMIT 20"
        )
        rows = cur.fetchall()
        result["recent"] = [
            {
                "ticket_id": row["id"],
                "user_id": row["assigned_to"] or "",
                "question": row["query"],
                "created_at": row["timestamp"],
                "status": row["status"],
            }
            for row in rows
        ]

        conn.close()
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# Pipeline Trace Viewer — independent from Langfuse
# ---------------------------------------------------------------------------

from core.trace_logger import list_traces as tl_list_traces, get_trace as tl_get_trace, subscribe_sse as tl_subscribe_sse


@app.get("/admin/traces", response_class=HTMLResponse)
async def traces_page():
    """Serve the pipeline trace viewer UI."""
    html_path = Path(__file__).parent.parent / "ui" / "traces.html"
    if html_path.exists():
        html = html_path.read_text(encoding="utf-8")
        return HTMLResponse(content=html)
    return HTMLResponse(content="<h1>traces.html not found</h1>", status_code=404)


@app.get("/admin/traces/list")
async def traces_list(limit: int = 50):
    """Return recent traces as JSON summary list."""
    return tl_list_traces(limit=limit)


@app.get("/admin/traces/stream/{trace_id}")
async def trace_stream(trace_id: str):
    """SSE endpoint — streams events for a running trace."""
    async def event_generator():
        # First, send already-buffered events
        trace = tl_get_trace(trace_id)
        if trace:
            for event in trace.get("events", []):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            if trace.get("finished_at"):
                yield f"data: {json.dumps({'node': '__done__', 'type': 'done'})}\n\n"
                return
        # Then subscribe for future events
        q = tl_subscribe_sse(trace_id)
        if not q:
            yield f"data: {json.dumps({'node': '__done__', 'type': 'not_found'})}\n\n"
            return
        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=30.0)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event.get("node") == "__done__":
                    break
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/admin/traces/{trace_id}")
async def trace_detail(trace_id: str):
    """Return full trace detail by trace_id."""
    trace = tl_get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace


# ---------------------------------------------------------------------------
# Admin Operations — run system tasks from the dashboard
# ---------------------------------------------------------------------------

ALLOWED_SERVICES = {
    "doantn": "doantn.service",
    "ehc-worker": "ehc-worker.service",
    "ehc-vllm": "ehc-vllm.service",
}


def _check_admin(request: Request) -> None:
    """Verify X-Admin-Token header matches ADMIN_TOKEN env var."""
    token = request.headers.get("X-Admin-Token", "")
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")


@app.post("/admin/ops/reindex")
async def ops_reindex(request: Request):
    """Run ingestor + embedder as subprocess, stream output via SSE."""
    _check_admin(request)
    job_id = start_job([
        "bash", "-c",
        "python3 -m data.ingestor && python3 -m data.embedder"
    ])
    return {"job_id": job_id}


@app.post("/admin/ops/restart/{service_key}")
async def ops_restart(service_key: str, request: Request):
    """Restart a systemd service. Requires sudoers NOPASSWD rule."""
    _check_admin(request)
    if service_key not in ALLOWED_SERVICES:
        raise HTTPException(status_code=400, detail=f"Unknown service '{service_key}'")
    service = ALLOWED_SERVICES[service_key]
    job_id = start_job(["sudo", "systemctl", "restart", service])
    return {"job_id": job_id}


@app.post("/admin/ops/reload-knowledge")
async def ops_reload_knowledge(request: Request):
    """Rebuild the knowledge _index.json in-process."""
    _check_admin(request)
    from core.knowledge_store import rebuild_index
    try:
        rebuild_index()
        return {"status": "ok", "message": "Knowledge index rebuilt"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/admin/ops/stream/{job_id}")
async def ops_stream(job_id: str):
    """SSE stream: sends buffered lines first, then new lines as they arrive."""
    import json as _json

    async def generator():
        job = get_job(job_id)
        if not job:
            yield f"data: {_json.dumps({'type': 'done', 'status': 'not_found'})}\n\n"
            return
        # Replay buffered lines
        for line in job["lines"]:
            yield f"data: {_json.dumps({'type': 'log', 'text': line})}\n\n"
        if job["status"] != "running":
            yield f"data: {_json.dumps({'type': 'done', 'status': job['status']})}\n\n"
            return
        # Subscribe for new lines
        q = subscribe_job(job_id)
        if not q:
            return
        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=30.0)
                yield f"data: {_json.dumps(event)}\n\n"
                if event.get("type") == "done":
                    break
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
