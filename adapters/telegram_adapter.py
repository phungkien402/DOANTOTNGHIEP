"""
Telegram Adapter — implements BaseAdapter for the Telegram Bot API.

Handles:
  - Parsing Telegram Update webhook payloads into Message objects
  - Formatting responses (plain text, Telegram supports basic markdown but
    we keep it simple to avoid escaping issues)
  - Sending messages via the Telegram Bot API
  - /clear command: clears session history for the current user

Run standalone: python -m adapters.telegram_adapter
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import httpx

from adapters.base_adapter import BaseAdapter
from core.models import Message
from config import TELEGRAM_BOT_TOKEN


# --- Session manager injection (set from api/routes.py) ---

_session_mgr = None


def set_session_manager(mgr):
    """Inject the SessionManager instance from api/routes.py."""
    global _session_mgr
    _session_mgr = mgr
    print("[TELEGRAM] SessionManager injected")


class TelegramAdapter(BaseAdapter):

    def __init__(self):
        self._api_base = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

    def parse_message(self, raw: dict) -> Message | None:
        """
        Parse a Telegram Update object. Only handles text messages.
        Returns None for non-text updates (photos, stickers, edits, etc.)
        Also returns None for /clear command (handled internally).
        """
        msg_data = raw.get("message")
        if not msg_data:
            return None

        text = msg_data.get("text")
        if not text:
            return None

        user_id = str(msg_data.get("from", {}).get("id", ""))
        chat_id = str(msg_data.get("chat", {}).get("id", ""))
        ts = msg_data.get("date", time.time())

        if not user_id:
            return None

        # Handle /clear command — clear session and reply immediately
        if text.strip() == "/clear":
            session_id = f"tg_{chat_id}"
            if _session_mgr:
                _session_mgr.clear(session_id)
                print(f"[TELEGRAM] /clear: session {session_id} cleared")
            # Also clear Redis session (used by pipeline_worker)
            try:
                import os as _os
                from redis import Redis as _Redis
                _r = _Redis.from_url(_os.getenv("REDIS_URL", "redis://localhost:6379"))
                _r.delete(f"tg_sess:{session_id}")
                print(f"[TELEGRAM] /clear: Redis session {session_id} deleted")
            except Exception as _e:
                print(f"[TELEGRAM] /clear: Redis delete failed: {_e}")
            self._send_clear_reply(chat_id)
            return None  # skip pipeline

        return Message(
            user_id=user_id,
            session_id=f"tg_{chat_id}",
            text=text,
            timestamp=float(ts),
            platform="telegram",
        )

    def _send_clear_reply(self, chat_id: str) -> None:
        """Send /clear confirmation synchronously (immediate feedback)."""
        if not TELEGRAM_BOT_TOKEN:
            print("[TELEGRAM] No bot token configured, skipping /clear reply")
            return

        url = f"{self._api_base}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": "✅ Đã xoá lịch sử chat.",
        }

        try:
            with httpx.Client(timeout=10) as client:
                resp = client.post(url, json=payload)
                if resp.status_code != 200:
                    print(f"[TELEGRAM] /clear reply failed: {resp.status_code} {resp.text}")
                else:
                    print(f"[TELEGRAM] /clear reply sent to {chat_id}")
        except Exception as e:
            print(f"[TELEGRAM] /clear reply error: {e}")

    def format_response(self, answer_text: str, confidence: float) -> str:
        """Format response for Telegram — plain text with confidence footer."""
        if confidence > 0:
            return f"{answer_text}\n\n📊 Độ tin cậy: {confidence:.0%}"
        return answer_text

    async def send_message(self, user_id: str, text: str) -> None:
        """Send message via Telegram Bot API (sendMessage)."""
        if not TELEGRAM_BOT_TOKEN:
            print("[TELEGRAM] No bot token configured, skipping send")
            return

        url = f"{self._api_base}/sendMessage"
        payload = {
            "chat_id": user_id,
            "text": text,
            "parse_mode": "HTML",
        }

        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code != 200:
                print(f"[TELEGRAM] Send failed: {resp.status_code} {resp.text}")
            else:
                print(f"[TELEGRAM] Message sent to {user_id}")


if __name__ == "__main__":
    adapter = TelegramAdapter()

    # Test parse
    sample_update = {
        "message": {
            "from": {"id": 12345},
            "chat": {"id": 12345},
            "date": 1700000000,
            "text": "how to merge patient records"
        }
    }
    msg = adapter.parse_message(sample_update)
    print(f"Parsed: {msg}")

    # Test format
    formatted = adapter.format_response("Vào Module Hành chính → Gộp hồ sơ", 0.95)
    print(f"Formatted:\n{formatted}")

    # Test ignore non-text
    non_text = {"message": {"from": {"id": 123}, "chat": {"id": 123}, "photo": []}}
    assert adapter.parse_message(non_text) is None

    # Test /clear returns None
    clear_msg = {"message": {"from": {"id": 123}, "chat": {"id": 123}, "date": 1700000000, "text": "/clear"}}
    assert adapter.parse_message(clear_msg) is None
    print("\n✓ TelegramAdapter works correctly.")
