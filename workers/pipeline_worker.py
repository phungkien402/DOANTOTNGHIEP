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

_SESSION_TTL  = 3600   # 1 hour idle timeout
_SESSION_MAX  = 20     # keep last 10 turns (20 messages)
_SESSION_KEY  = "tg_sess:"   # Redis key prefix for session data


# ---------------------------------------------------------------------------
# Redis session helpers
# ---------------------------------------------------------------------------

def _get_redis():
    url = os.getenv("REDIS_URL", "redis://localhost:6379")
    return _redis_lib.from_url(url)


def _load_tg_session(session_id: str) -> dict:
    """
    Load session data from Redis. Returns:
    {
        "history": [{"role": "user"|"bot", "text": "..."}, ...],
        "awaiting_clarification": bool,
        "original_query": str,   # query that triggered the clarify
    }
    """
    try:
        r = _get_redis()
        data = r.get(_SESSION_KEY + session_id)
        if data:
            return _json.loads(data)
    except Exception as e:
        print(f"[WORKER] Session load error: {e}")
    return {"history": [], "awaiting_clarification": False, "original_query": ""}


def _save_tg_session(session_id: str, session_data: dict) -> None:
    """Save session data to Redis with TTL."""
    try:
        history = session_data.get("history", [])
        session_data["history"] = history[-_SESSION_MAX:]  # trim
        r = _get_redis()
        r.setex(_SESSION_KEY + session_id, _SESSION_TTL, _json.dumps(session_data))
    except Exception as e:
        print(f"[WORKER] Session save error: {e}")


# ---------------------------------------------------------------------------
# Minimal SessionManager shim for langgraph_agent._session_mgr
# ---------------------------------------------------------------------------

class _RedisSessionMgr:
    """
    Implements just the methods langgraph_agent.py calls on _session_mgr.
    Backed by a mutable dict so state changes (set_awaiting_clarification)
    are visible after run() returns.
    """
    def __init__(self, session_id: str, data: dict):
        self._sid = session_id
        self._data = data  # mutable reference — changes are reflected after run()

    def is_awaiting_clarification(self, session_id: str) -> bool:
        return self._data.get("awaiting_clarification", False)

    def set_awaiting_clarification(self, session_id: str, value: bool) -> None:
        self._data["awaiting_clarification"] = value
        if not value:
            self._data["clarify_count"] = 0

    def get_clarify_count(self, session_id: str) -> int:
        return self._data.get("clarify_count", 0)

    def increment_clarify_count(self, session_id: str) -> None:
        self._data["clarify_count"] = self._data.get("clarify_count", 0) + 1

    def reset_clarify_count(self, session_id: str) -> None:
        self._data["clarify_count"] = 0    

    def get_fast_chunks(self, session_id: str) -> list:
        # Can't serialize RetrievedChunk objects — return [] and rely on
        # enriched query (original_query + follow-up) for fresh retrieval.
        return []

    def set_fast_chunks(self, session_id: str, chunks: list) -> None:
        pass  # not persisted — handled via original_query enrichment

    def get_history(self, session_id: str) -> list:
        return self._data.get("history", [])

    def add_turn(self, session_id: str, role: str, text: str) -> None:
        pass  # history saved by worker after run()

    def clear(self, session_id: str) -> None:
        self._data["awaiting_clarification"] = False
        self._data["original_query"] = ""
        self._data["history"] = []


# ---------------------------------------------------------------------------
# Main job function
# ---------------------------------------------------------------------------

def process_telegram_query(chat_id: str, text: str, session_id: str, history: list):
    """
    RQ job: run LangGraph agent and send reply to Telegram.
    Called by the RQ worker, not by FastAPI directly.
    """
    from core.models import Message
    from core.langgraph_agent import run, set_session_manager

    # 1. Load actual session from Redis (ignore history= passed by enqueue — always stale)
    session_data = _load_tg_session(session_id)
    session_history = session_data.get("history", [])
    awaiting = session_data.get("awaiting_clarification", False)
    original_query = session_data.get("original_query", "")

    print(f"[WORKER] Session | id={session_id} | turns={len(session_history)//2} | awaiting={awaiting}")

    # 2. Inject Redis-backed session manager so langgraph_agent can track
    #    awaiting_clarification state across Telegram turns.
    shim = _RedisSessionMgr(session_id, session_data)
    set_session_manager(shim)

    # 3. If bot was awaiting clarification, enrich the follow-up query with
    #    the original question so FastRetriever gets relevant chunks.
    effective_text = text
    if awaiting and original_query:
        effective_text = f"{original_query} — {text}"
        print(f"[WORKER] Enriched query: \"{effective_text}\"")

    msg = Message(
        user_id=chat_id,
        session_id=session_id,
        text=effective_text,
        timestamp=time.time(),
        platform="telegram",
    )

    # 4. Run the agent
    answer_text = None
    try:
        answer = run(msg, session_history)
        answer_text = answer.text
        confidence = answer.confidence
    except Exception as e:
        print(f"[WORKER] Agent error: {e}")
        answer_text = None
        confidence = 0.0

    # 5. Build reply with confidence badge
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

    # 6. Save session only if we got a real answer
    if answer_text:
        session_data["history"].append({"role": "user", "text": text})  # original text, not enriched
        session_data["history"].append({"role": "bot",  "text": answer_text})

        # Track clarification state for next turn
        new_awaiting = session_data.get("awaiting_clarification", False)  # updated by shim
        if new_awaiting and not awaiting:
            # Bot just asked a clarify question — save the original (non-enriched) query
            session_data["original_query"] = text
        elif not new_awaiting:
            session_data["original_query"] = ""

        _save_tg_session(session_id, session_data)
        print(f"[WORKER] Session saved | awaiting={new_awaiting} | turns={len(session_data['history'])//2}")


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

def process_slack_query(session_id: str, channel_id: str, text: str, thread_ts: str, history: list):
    """RQ job: run LangGraph agent and send reply to Slack."""
    from core.models import Message
    from core.langgraph_agent import run, set_session_manager

    # 1. Load session từ Redis
    session_data = _load_tg_session(session_id)
    session_history = session_data.get("history", [])
    awaiting = session_data.get("awaiting_clarification", False)
    original_query = session_data.get("original_query", "")

    print(f"[WORKER-SLACK] Session | id={session_id} | turns={len(session_history)//2} | awaiting={awaiting}")

    # 2. Inject session manager shim
    shim = _RedisSessionMgr(session_id, session_data)
    set_session_manager(shim)

    # 3. Enrich query nếu đang awaiting clarification
    effective_text = text
    if awaiting and original_query:
        effective_text = f"{original_query} — {text}"
        print(f"[WORKER-SLACK] Enriched query: \"{effective_text}\"")

    msg = Message(
        user_id=session_id,
        session_id=session_id,
        text=effective_text,
        timestamp=time.time(),
        platform="slack",
    )

    # 4. Run agent
    answer_text = None
    confidence = 0.0
    try:
        answer = run(msg, session_history)
        answer_text = answer.text
        confidence = answer.confidence
    except Exception as e:
        print(f"[WORKER-SLACK] Agent error: {e}")

    # 5. Build reply
    if answer_text:
        reply_text = answer_text
        if confidence >= 0.4:
            reply_text += f"\n\n📊 Độ tin cậy: {confidence*100:.0f}%"
        else:
            reply_text += "\n\n🔴 Độ tin cậy: thấp"
    else:
        reply_text = "⚠️ Hệ thống đang bận, vui lòng thử lại sau."

    _send_slack(channel_id, reply_text, thread_ts)
    print(f"[WORKER-SLACK] Done | channel={channel_id} | conf={confidence:.4f}")

    # 6. Save session
    if answer_text:
        session_data["history"].append({"role": "user", "text": text})
        session_data["history"].append({"role": "bot",  "text": answer_text})

        new_awaiting = session_data.get("awaiting_clarification", False)
        if new_awaiting and not awaiting:
            session_data["original_query"] = text
        elif not new_awaiting:
            session_data["original_query"] = ""

        _save_tg_session(session_id, session_data)
        print(f"[WORKER-SLACK] Session saved | awaiting={new_awaiting}")


def _send_slack(channel_id: str, text: str, thread_ts: str = None):
    """Send a message via Slack Web API (synchronous)."""
    token = os.getenv("SLACK_BOT_TOKEN", "")
    if not token:
        print("[WORKER-SLACK] No SLACK_BOT_TOKEN, skipping send")
        return
    payload = {"channel": channel_id, "text": text}
    if thread_ts:
        payload["thread_ts"] = thread_ts
    try:
        resp = requests.post(
            "https://slack.com/api/chat.postMessage",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        data = resp.json()
        if not data.get("ok"):
            print(f"[WORKER-SLACK] Send failed: {data.get('error')}")
        else:
            print(f"[WORKER-SLACK] Sent to {channel_id}")
    except Exception as e:
        print(f"[WORKER-SLACK] Send error: {e}")