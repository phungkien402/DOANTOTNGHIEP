"""
Slack Adapter — implements BaseAdapter for Slack Events API.

Handles:
  - app_mention events — when someone tags @EHC Bot in a channel
  - Slash commands (/health, /stats, /top, /clear, /refresh, /create_ticket)
  - Sending replies via Slack Web API (chat.postMessage)
  - Signature verification using HMAC-SHA256

Note: URL verification challenge is handled in api/routes.py before
reaching this adapter, since it's not a real message event.

Env vars needed:
  SLACK_BOT_TOKEN=xoxb-...
  SLACK_SIGNING_SECRET=...
  SLACK_ADMIN_USERS=U12345,U67890  (for /refresh command)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import hashlib
import hmac
import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone

import httpx

from adapters.base_adapter import BaseAdapter
from core.models import Message
from config import SLACK_BOT_TOKEN, SLACK_SIGNING_SECRET, SLACK_ADMIN_USERS


# Module-level dict to store thread_ts for replies (session_id -> event ts)
_pending_thread_ts: dict[str, str] = {}


class SlackAdapter(BaseAdapter):

    def pop_thread_ts(self, session_id: str) -> str | None:
        """Pop thread_ts for a session (used before enqueuing to RQ worker)."""
        return _pending_thread_ts.pop(session_id, None)

    def parse_message(self, raw: dict) -> Message | None:
        """
        Parse a Slack Events API payload.
        Only handles event_type == "app_mention". Ignores everything else.
        """
        event = raw.get("event")
        if not event:
            return None

        # Only respond to app_mention events
        if event.get("type") != "app_mention":
            return None

        text = event.get("text", "")
        # Strip the bot mention (e.g. <@U12345>) from the text
        text = re.sub(r"<@[A-Z0-9]+>", "", text).strip()

        if not text:
            return None

        user_id = event.get("user", "")
        channel_id = event.get("channel", "")
        ts_str = event.get("ts", "")
        ts = float(ts_str) if ts_str else time.time()

        if not user_id or not channel_id:
            return None

        # Store thread_ts so send_message can reply in-thread
        thread_root = event.get("thread_ts") or ts_str
        session_id = f"slack_{channel_id}_{thread_root}"
        _pending_thread_ts[session_id] = thread_root

        return Message(
            user_id=user_id,
            session_id=session_id,
            text=text,
            timestamp=ts,
            platform="slack",
        )

    def format_response(self, answer_text: str, confidence: float) -> str:
        """Format response for Slack — plain text with confidence footer."""
        if confidence > 0:
            return f"{answer_text}\n\n📊 Độ tin cậy: {confidence:.0%}"
        return answer_text

    async def send_message(self, channel_id: str, text: str) -> None:
        """Send message via Slack Web API (chat.postMessage), replying in-thread."""
        if not SLACK_BOT_TOKEN:
            print("[SLACK] No bot token configured, skipping send")
            return

        url = "https://slack.com/api/chat.postMessage"
        headers = {
            "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
            "Content-Type": "application/json",
        }

        # channel_id may be full session_id like "slack_C12345_U67890"
        # extract actual channel for Slack API
        parts = channel_id.replace("slack_", "").split("_")
        actual_channel = parts[0]  # first part is channel ID

        # Reply in-thread if we have the original message ts
        thread_ts = _pending_thread_ts.pop(channel_id, None)

        payload = {
            "channel": actual_channel,
            "text": text,
        }
        if thread_ts:
            payload["thread_ts"] = thread_ts

        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json=payload, headers=headers)
            data = resp.json()
            if not data.get("ok"):
                print(f"[SLACK] Send failed: {data.get('error', 'unknown')}")
            else:
                print(f"[SLACK] Message sent to {actual_channel} (thread={thread_ts or 'none'})")

    @staticmethod
    def verify_signature(
        signing_secret: str, timestamp: str, body: bytes, signature: str
    ) -> bool:
        """
        Verify the X-Slack-Signature header using HMAC-SHA256.
        Returns True if the request is authentic.
        """
        # Reject requests older than 5 minutes to prevent replay attacks
        if abs(time.time() - int(timestamp)) > 300:
            return False

        sig_basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
        computed = (
            "v0="
            + hmac.new(
                signing_secret.encode(), sig_basestring.encode(), hashlib.sha256
            ).hexdigest()
        )

        return hmac.compare_digest(computed, signature)

    # ------------------------------------------------------------------
    # Slash command handlers
    # ------------------------------------------------------------------

    async def handle_slash_command(self, form_data: dict) -> str:
        """
        Route a slash command to the appropriate handler.
        Returns plain text response to send back to Slack (within 3s).
        """
        command = form_data.get("command", "").strip()
        user_id = form_data.get("user_id", "")
        channel_id = form_data.get("channel_id", "")
        text = form_data.get("text", "").strip()

        handlers = {
            "/health": self._cmd_health,
            "/stats": self._cmd_stats,
            "/top": self._cmd_top,
            "/clear": self._cmd_clear,
            "/refresh": self._cmd_refresh,
            "/create_ticket": self._cmd_create_ticket,
        }

        handler = handlers.get(command)
        if handler is None:
            return f"❓ Lệnh không hợp lệ: `{command}`. Các lệnh hỗ trợ: /health, /stats, /top, /clear, /refresh, /create_ticket"

        return await handler(user_id=user_id, channel_id=channel_id, text=text)

    async def _cmd_health(self, **kwargs) -> str:
        """Check vLLM, Qdrant, and API health."""
        results = []

        async with httpx.AsyncClient(timeout=5) as client:
            # vLLM
            try:
                resp = await client.get("http://localhost:8000/health")
                vllm_ok = resp.status_code == 200
            except Exception:
                vllm_ok = False
            results.append(f"{'🟢' if vllm_ok else '🔴'} vLLM: {'OK' if vllm_ok else 'DOWN'}")

            # Qdrant
            try:
                resp = await client.get("http://localhost:6333/healthz")
                qdrant_ok = resp.status_code == 200
            except Exception:
                qdrant_ok = False
            results.append(f"{'🟢' if qdrant_ok else '🔴'} Qdrant: {'OK' if qdrant_ok else 'DOWN'}")

        # API is always up if we're responding
        results.append("🟢 API: OK")

        return "  |  ".join(results)

    async def _cmd_stats(self, **kwargs) -> str:
        """Query logs for last 24h stats."""
        from api.logger import QueryLogger

        logger = QueryLogger()
        all_logs = logger.read_logs(limit=10000)

        now = time.time()
        cutoff = now - 86400  # 24 hours
        recent = [log for log in all_logs if log.get("timestamp", 0) >= cutoff]

        if not recent:
            return "📊 Stats (last 24h): Chưa có câu hỏi nào."

        total = len(recent)
        success = sum(1 for log in recent if not log.get("is_fallback", True))
        success_rate = (success / total * 100) if total > 0 else 0
        avg_confidence = sum(log.get("confidence", 0) for log in recent) / total

        return (
            f"📊 Stats (last 24h): {total} questions | "
            f"{success_rate:.0f}% success | "
            f"avg confidence {avg_confidence:.2f}"
        )

    async def _cmd_top(self, **kwargs) -> str:
        """Show top 5 most asked questions in last 7 days."""
        from api.logger import QueryLogger

        logger = QueryLogger()
        all_logs = logger.read_logs(limit=50000)

        now = time.time()
        cutoff = now - 7 * 86400  # 7 days
        recent = [log for log in all_logs if log.get("timestamp", 0) >= cutoff]

        if not recent:
            return "📋 Top questions (7 days): Chưa có dữ liệu."

        # Count by top_chunk_subject (FAQ matched), fall back to question text
        questions = []
        for log in recent:
            subject = log.get("top_chunk_subject", "")
            if subject:
                questions.append(subject)
            else:
                questions.append(log.get("question", "unknown"))

        counter = Counter(questions)
        top5 = counter.most_common(5)

        lines = ["📋 Top 5 câu hỏi (7 ngày qua):"]
        for i, (question, count) in enumerate(top5, 1):
            # Truncate long questions
            display = question[:80] + "..." if len(question) > 80 else question
            lines.append(f"{i}. ({count} lần) {display}")

        return "\n".join(lines)

    async def _cmd_clear(self, user_id: str = "", channel_id: str = "", **kwargs) -> str:
        """Clear session history for the calling user."""
        from api.session import SessionManager
        from config import SESSION_MAX_TURNS

        session_id = f"slack_{channel_id}_{user_id}"

        # We need access to the shared session manager from routes
        # Import it at call time to get the singleton
        from api.routes import _session_mgr

        _session_mgr.clear(session_id)
        return "🧹 Đã xóa lịch sử hội thoại của bạn."

    async def _cmd_refresh(self, user_id: str = "", **kwargs) -> str:
        """Trigger reindex — admin only."""
        if user_id not in SLACK_ADMIN_USERS:
            return "🚫 Bạn không có quyền thực hiện lệnh này. Liên hệ admin."

        # Trigger reindex in background
        import asyncio
        from data.reindex import full_reindex

        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, full_reindex)

        return "🔄 Đang cập nhật dữ liệu từ Redmine, vui lòng chờ..."

    async def _cmd_create_ticket(self, user_id: str = "", channel_id: str = "", text: str = "", **kwargs) -> str:
        """Log an unresolved ticket for manual review."""
        from api.logger import QueryLogger
        from core.models import Message, Answer

        if not text:
            return "🎫 Vui lòng mô tả vấn đề sau lệnh, ví dụ: `/create_ticket Không in được phiếu khám`"

        # Log as unresolved query flagged for manual review
        logger = QueryLogger()
        msg = Message(
            user_id=user_id,
            session_id=f"slack_{channel_id}_{user_id}",
            text=text,
            timestamp=time.time(),
            platform="slack",
        )
        ans = Answer(
            text="[TICKET] Chuyển đội hỗ trợ kỹ thuật",
            confidence=0.0,
            is_fallback=True,
            source_chunks=[],
            rewritten_question=text,
        )
        logger.log(msg, ans)

        return "🎫 Câu hỏi của bạn đã được ghi nhận và chuyển đến đội hỗ trợ kỹ thuật."


if __name__ == "__main__":
    adapter = SlackAdapter()

    # Test parse — app_mention event
    sample_event = {
        "event": {
            "type": "app_mention",
            "user": "U12345",
            "channel": "C67890",
            "text": "<@UBOTID> in phiếu không lên form view",
            "ts": "1700000000.000100",
        }
    }
    msg = adapter.parse_message(sample_event)
    print(f"Parsed: {msg}")
    assert msg is not None
    assert msg.text == "in phiếu không lên form view"
    assert msg.session_id == "slack_C67890_U12345"
    assert msg.platform == "slack"

    # Test ignore non-mention events
    non_mention = {"event": {"type": "message", "user": "U12345", "text": "hello"}}
    assert adapter.parse_message(non_mention) is None

    # Test format
    formatted = adapter.format_response("Vào Module Hành chính → Gộp hồ sơ", 0.95)
    print(f"Formatted:\n{formatted}")

    print("\n✓ SlackAdapter works correctly.")
