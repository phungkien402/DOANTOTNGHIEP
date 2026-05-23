"""
orchestrator.py — LLM Orchestrator node.

Takes: query + fast_chunks (top 3) + session_history
Returns: {
    "action": "answer" | "ticket",
    "reasoning": str,
    "search_query": str,      # if action=answer: use this for full retrieve
}

Replaces: score-spread heuristic, clarification_count routing, Block X node.

Run standalone: python3 -m core.orchestrator
"""

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI, APIConnectionError

from config import VLLM_BASE_URL, VLLM_MODEL
from core.knowledge_store import list_topics

# Module-level client — same pattern as query_rewriter.py
_client = OpenAI(base_url=f"{VLLM_BASE_URL}/v1", api_key="not-needed")


def _safe_content(response) -> str | None:
    """Return message content or None if missing (Qwen3 thinking overflow)."""
    try:
        return response.choices[0].message.content
    except Exception:
        return None


_BUDGET_ORCHESTRATOR = 512  # enough reasoning for routing decisions

ORCHESTRATOR_PROMPT = """Bạn là bộ não của hệ thống hỗ trợ phần mềm EHC (quản lý bệnh viện).

Nhiệm vụ: Đọc câu hỏi của người dùng, lịch sử hội thoại, và 3 đoạn FAQ tìm được. Quyết định hành động tiếp theo.

---
LỊCH SỬ HỘI THOẠI:
{history}

---
CÂU HỎI HIỆN TẠI: {query}

---
3 ĐOẠN FAQ TÌM ĐƯỢC (theo thứ tự liên quan):
{chunks}

---
CÁC FILE HƯỚNG DẪN NGHIỆP VỤ ĐANG CÓ:
{knowledge_topics}

---
HƯỚNG DẪN QUYẾT ĐỊNH:

1. action = "answer" — khi user đề cập rõ chủ thể cụ thể VÀ chunk #1 chứa đúng chủ thể đó.
   Ví dụ chủ thể cụ thể: "bảng kê", "tài liệu chưa ký", "bệnh án", "phiếu thu", "phiếu khám",
   "giấy ra viện", "bảng kê 6556", "phiếu chỉ định", v.v.
   Ví dụ KHÔNG phải chủ thể cụ thể: "không in được", "bị lỗi", "không dùng được", "mình không làm được"
   → search_query = câu truy vấn tối ưu, tiếng Việt, cụ thể, bỏ từ thừa ("mình", "ấy", "nhỉ", "vậy")

2. action = "ticket" — CHỈ khi đã có đủ thông tin về vấn đề nhưng không tìm được chunk phù hợp.
   KHÔNG tạo ticket nếu câu hỏi vẫn còn mơ hồ — lúc đó IntentAnalyzer đã xử lý rồi.
   Ngoại lệ: nếu query rõ ràng là câu lệnh phá hoại, không liên quan gì đến phần mềm EHC → ticket ngay.
---
CHỌN TOOL TÌM KIẾM (field "tool"):
- "search_manual" — khi:
  * Chunk #1 hoặc #2 có source chứa "hdsd" VÀ score > 0.5 → LUÔN chọn search_manual.
  * User hỏi CÁCH SỬ DỤNG một chức năng: cài đặt, cấu hình, hướng dẫn từng bước, quy trình.
  Ví dụ: "cách cài đặt...", "hướng dẫn kết nối...", "làm thế nào để...", "thao tác...", "các bước để...", "quy trình..."
- "search_faq" — khi:
  * Các chunk đầu có source="faq" chiếm ưu thế, HOẶC
  * User báo lỗi, hỏi tại sao, hoặc gặp vấn đề không hoạt động.
  Ví dụ: "không in được", "bị lỗi", "tại sao không...", "không đăng nhập được", "bấm không được"
- QUAN TRỌNG: Nếu chunk #1 hoặc #2 có source chứa "hdsd" và score > 0.5, luôn chọn "search_manual".
- Mặc định: "search_faq" khi không chắc chắn.

---
CÔNG CỤ HỖ TRỢ NGỮ CẢNH NGHIỆP VỤ (field "knowledge_topic"):
Khi nào dùng search_knowledge:
1. Trước tiên, đọc fast_chunks đã truy xuất được.
2. Nếu fast_chunks đã giải thích được nguyên nhân gốc (root cause) của vấn đề → KHÔNG cần gọi search_knowledge, để knowledge_topic = "".
3. Chỉ đặt knowledge_topic khi fast_chunks cho thấy vấn đề thuộc một lĩnh vực nghiệp vụ cụ thể (in ấn, mạng, thuốc, xuất viện...) VÀ người dùng cần hướng dẫn thao tác chi tiết hơn những gì chunks cung cấp.
4. Nếu chưa chắc chắn topic nào phù hợp → để knowledge_topic = "".

---
TRẢ LỜI THEO ĐỊNH DẠNG JSON (không giải thích thêm):
{{
  "action": "answer" | "ticket",
  "tool": "search_faq" | "search_manual",
  "knowledge_topic": "" | "<stem từ danh sách trên>",
  "reasoning": "lý do ngắn gọn",
  "search_query": "..."
}}"""


def _format_chunks(chunks) -> str:
    """Format fast_chunks for the orchestrator prompt, including scores and source."""
    if not chunks:
        return "(không có)"
    lines = []
    for i, c in enumerate(chunks, 1):
        title = getattr(c, "title", "") or (c.metadata.get("subject", "") if hasattr(c, "metadata") else "") or (c.text or "")[:80] if hasattr(c, "text") else str(c)[:80]
        score = getattr(c, "score", 0.0)
        source = c.metadata.get("source", "faq") if hasattr(c, "metadata") else "faq"
        lines.append(f"{i}. [score={score:.3f}][source={source}] {title}")
    return "\n".join(lines)


def _format_history(session_history: list) -> str:
    """Format session history for the orchestrator prompt."""
    if not session_history:
        return "(không có)"
    lines = []
    for turn in session_history[-4:]:  # last 4 turns max
        role = "Người dùng" if turn.get("role") == "user" else "Bot"
        text = turn.get("text", turn.get("content", ""))[:150]
        lines.append(f"{role}: {text}")
    return "\n".join(lines)


def _format_knowledge_topics() -> str:
    """Format available knowledge topics for the orchestrator prompt."""
    topics = list_topics()
    if not topics:
        return "(không có file hướng dẫn nào)"
    lines = []
    for t in topics:
        lines.append(f"- {t['stem']}: {t['title']} — bao gồm: {t['covers']}")
    return "\n".join(lines)


def orchestrate(query: str, fast_chunks: list, session_history: list = None) -> dict:
    """
    Call the LLM to decide the next action.

    Returns dict with keys: action, reasoning, search_query, tool, knowledge_topic.
    Fallback to {"action": "answer", "search_query": query} on any error.
    """
    prompt = ORCHESTRATOR_PROMPT.format(
        query=query,
        chunks=_format_chunks(fast_chunks),
        history=_format_history(session_history or []),
        knowledge_topics=_format_knowledge_topics(),
    )

    print(f"[ORCHESTRATOR] Query: \"{query}\"")

    messages = [{"role": "system", "content": prompt}]

    try:
        response = _client.chat.completions.create(
            model=VLLM_MODEL,
            messages=messages,
            max_tokens=1000,
            temperature=0.1,
            extra_body={"chat_template_kwargs": {"enable_thinking": True, "thinking_budget": _BUDGET_ORCHESTRATOR}},
        )
        raw = _safe_content(response)
        if raw is None:
            print("[ORCHESTRATOR] content=None (thinking overflow) → fallback")
            return _fallback_result(query)
        raw = raw.strip()
    except APIConnectionError:
        # Retry once after 1s
        print("[ORCHESTRATOR] Connection error, retrying in 1s...")
        time.sleep(1)
        try:
            response = _client.chat.completions.create(
                model=VLLM_MODEL,
                messages=messages,
                max_tokens=1000,
                temperature=0.1,
                extra_body={"chat_template_kwargs": {"enable_thinking": True, "thinking_budget": _BUDGET_ORCHESTRATOR}},
            )
            raw = _safe_content(response)
            if raw is None:
                print("[ORCHESTRATOR] content=None (thinking overflow, retry) → fallback")
                return _fallback_result(query)
            raw = raw.strip()
        except Exception as e:
            print(f"[ORCHESTRATOR] Retry failed: {e} → fallback to answer")
            return _fallback_result(query)
    except Exception as e:
        print(f"[ORCHESTRATOR] LLM failed: {e} → fallback to answer")
        return _fallback_result(query)

    print(f"[ORCHESTRATOR] Raw output: {raw[:200]}")

    # Parse JSON — extract from markdown code block if wrapped
    return _parse_response(raw, query)


def _parse_response(raw: str, query: str) -> dict:
    """Parse the LLM JSON response into a structured dict."""
    try:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match:
            raise ValueError("No JSON found in response")
        result = json.loads(match.group())
        action = result.get("action", "answer")
        if action not in ("answer", "ticket"):
            action = "answer"
        result["action"] = action
        result.setdefault("search_query", query)
        result.setdefault("reasoning", "")
        result.setdefault("tool", "search_faq")
        result.setdefault("knowledge_topic", "")
        print(f"[ORCHESTRATOR] Action={action} | tool={result['tool']} | knowledge_topic={result['knowledge_topic']} | reasoning=\"{result['reasoning'][:80]}\"")
        return result
    except Exception as e:
        print(f"[ORCHESTRATOR] Parse error: {e} → fallback to answer")
        return _fallback_result(query)


def _fallback_result(query: str) -> dict:
    """Return a safe fallback when orchestrator fails."""
    return {
        "action": "answer",
        "reasoning": "orchestrator fallback",
        "search_query": query,
        "tool": "search_faq",
        "knowledge_topic": "",
    }


if __name__ == "__main__":
    print("=== Orchestrator standalone test ===\n")

    # Simulate a RetrievedChunk-like object for testing
    class FakeChunk:
        def __init__(self, text, subject):
            self.text = text
            self.metadata = {"subject": subject}
            self.score = 0.5

    fake_chunks = [
        FakeChunk("Lỗi in phiếu thu...", "Lỗi in phiếu thu không hiển thị"),
        FakeChunk("Cách in phiếu thu...", "Cách in phiếu thu từ module viện phí"),
        FakeChunk("Lỗi máy in...", "Lỗi máy in không kết nối"),
    ]

    # Test 1: ambiguous query
    print("--- Test 1: Ambiguous query ---")
    result = orchestrate("không in được", fake_chunks, [])
    print(f"Result: {json.dumps(result, ensure_ascii=False, indent=2)}\n")

    # Test 2: clear query
    print("--- Test 2: Clear query ---")
    result = orchestrate("lỗi in phiếu thu không hiển thị form view", fake_chunks, [])
    print(f"Result: {json.dumps(result, ensure_ascii=False, indent=2)}\n")

    # Test 3: with history (already clarified once)
    print("--- Test 3: With clarification history ---")
    history = [
        {"role": "user", "text": "không in được"},
        {"role": "bot", "text": "Bạn đang gặp vấn đề nào?\n1. Lỗi in phiếu thu\n2. Cách in phiếu thu\n3. Lỗi máy in"},
        {"role": "user", "text": "1"},
    ]
    result = orchestrate("1", fake_chunks, history)
    print(f"Result: {json.dumps(result, ensure_ascii=False, indent=2)}\n")

    print("✓ Orchestrator tests completed.")
