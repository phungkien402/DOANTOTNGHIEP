# FIX: Telegram session persistence

## Root cause (confirmed from source)

`api/routes.py` lines 141-152: Telegram returns `{"ok": True}` immediately after enqueue.
The `_session_mgr.add_turn()` calls at lines 160-161 are **never reached** for Telegram.
Every job receives `history=[]` because the in-memory SessionManager is never updated.

`pipeline_worker.py` uses the passed `history` directly:
```python
answer = run(msg, history)   # history is always []
```

Session history format (from `api/session.py` line 44):
```python
{"role": "user", "text": "..."}   # role = "user" or "bot"
```

---

## Fix 1 — pipeline_worker.py (main fix)

Add Redis session load/save. Worker is the source of truth for Telegram sessions.

Full replacement for `~/DOANTN/workers/pipeline_worker.py`:

```python
"""
pipeline_worker.py — RQ worker that processes queued Telegram queries.
Runs the LangGraph agent and sends the reply back via Telegram Bot API.

Start worker with:
    rtk rq worker ehc-queue --url redis://localhost:6379
"""

import sys
import time
import os
import json as _json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import redis as _redis_lib
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

_SESSION_TTL = 3600        # 1 hour idle timeout
_SESSION_MAX = 20          # keep last 10 turns (20 messages)
_SESSION_PREFIX = "tg_sess:"


def _get_redis():
    url = os.getenv("REDIS_URL", "redis://localhost:6379")
    return _redis_lib.from_url(url)


def _load_tg_session(session_id: str) -> list:
    """Load session history from Redis. Returns [] on any error."""
    try:
        r = _get_redis()
        data = r.get(_SESSION_PREFIX + session_id)
        if data:
            return _json.loads(data)
    except Exception as e:
        print(f"[WORKER] Session load error: {e}")
    return []


def _save_tg_session(session_id: str, history: list) -> None:
    """Save session history to Redis with TTL."""
    try:
        trimmed = history[-_SESSION_MAX:]
        r = _get_redis()
        r.setex(_SESSION_PREFIX + session_id, _SESSION_TTL, _json.dumps(trimmed))
    except Exception as e:
        print(f"[WORKER] Session save error: {e}")


def process_telegram_query(chat_id: str, text: str, session_id: str, history: list):
    """
    RQ job: run LangGraph agent and send reply to Telegram.
    Called by the RQ worker, not by FastAPI directly.
    """
    from core.models import Message
    from core.langgraph_agent import run

    # Load actual session from Redis.
    # Ignore the passed history= — routes.py never updates it for Telegram.
    session_history = _load_tg_session(session_id)
    print(f"[WORKER] Session | id={session_id} | turns={len(session_history)//2}")

    msg = Message(
        user_id=chat_id,
        session_id=session_id,
        text=text,
        timestamp=time.time(),
        platform="telegram",
    )

    answer_text = None
    try:
        answer = run(msg, session_history)
        answer_text = answer.text
        confidence = answer.confidence
    except Exception as e:
        print(f"[WORKER] Agent error: {e}")
        answer_text = None
        confidence = 0.0

    # Build reply
    if answer_text:
        reply_text = answer_text
        if confidence >= 0.4:
            reply_text += f"\n\n🟢 Độ tin cậy: {confidence*100:.0f}%"
        elif not reply_text.startswith("⚠️"):
            reply_text += "\n\n🔴 Độ tin cậy: thấp"
    else:
        reply_text = "⚠️ Hệ thống đang bận, vui lòng thử lại sau."

    _send_telegram(chat_id, reply_text)
    print(f"[WORKER] Done | chat_id={chat_id} | conf={confidence:.4f}")

    # Save session only if we got a real answer (not an error)
    if answer_text:
        session_history.append({"role": "user", "text": text})
        session_history.append({"role": "bot", "text": answer_text})
        _save_tg_session(session_id, session_history)


def _send_telegram(chat_id: str, text: str):
    """Send a message via Telegram Bot API."""
    try:
        resp = requests.post(
            f"{TELEGRAM_API}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
        if not resp.ok:
            print(f"[WORKER] Telegram send failed: {resp.status_code} {resp.text[:100]}")
    except Exception as e:
        print(f"[WORKER] Telegram send error: {e}")
```

---

## Fix 2 — telegram_adapter.py (/clear must also clear Redis)

In `~/DOANTN/adapters/telegram_adapter.py`, find the `/clear` handler (around line 65):

**Before:**
```python
if text.strip() == "/clear":
    session_id = f"tg_{chat_id}"
    if _session_mgr:
        _session_mgr.clear(session_id)
        print(f"[TELEGRAM] /clear: session {session_id} cleared")
    self._send_clear_reply(chat_id)
    return None
```

**After:**
```python
if text.strip() == "/clear":
    session_id = f"tg_{chat_id}"
    if _session_mgr:
        _session_mgr.clear(session_id)
        print(f"[TELEGRAM] /clear: session {session_id} cleared")
    # Also clear Redis session (used by pipeline_worker)
    try:
        import os
        from redis import Redis as _Redis
        _r = _Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
        _r.delete(f"tg_sess:{session_id}")
        print(f"[TELEGRAM] /clear: Redis session {session_id} deleted")
    except Exception as _e:
        print(f"[TELEGRAM] /clear: Redis delete failed: {_e}")
    self._send_clear_reply(chat_id)
    return None
```

---

## Steps

```bash
# 1. Apply changes
nano ~/DOANTN/workers/pipeline_worker.py
nano ~/DOANTN/adapters/telegram_adapter.py

# 2. Restart worker
sudo systemctl restart ehc-worker

# 3. Test in Telegram:
#    - Send: "làm sao để thêm vân tay cho bệnh nhân"
#    - Bot trả lời clarify question
#    - Reply: "Tôi muốn thêm trong module nhập viện"
#    - Bot dùng context từ câu trước để trả lời đúng

# 4. Check logs
sudo journalctl -u ehc-worker -n 30
```

Expected logs:
```
[WORKER] Session | id=tg_5770498222 | turns=0
[WORKER] Done | chat_id=5770498222 | conf=0.0000
[WORKER] Session | id=tg_5770498222 | turns=1   ← second message has history!
[WORKER] Done | chat_id=5770498222 | conf=0.8500
```

---

## Note on clarify UX

Even with session fixed, user replying "3" will still be rejected by intent guard.
The `/clear` note at line 67-70 of `telegram_adapter.py` only clears in-memory session — this fix also handles Redis, so `/clear` now fully resets the conversation.

Consider tweaking the Orchestrator clarify prompt to ask **one question** instead of numbered list:
> "Bạn muốn thêm vân tay ở bước nào? (ví dụ: đăng ký khám, nhập viện, hay chỗ khác?)"

That way user answers naturally with text, not a number.
