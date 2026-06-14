"""
orchestrator.py — LLM Orchestrator node.

Takes: query + fast_chunks (top 3) + session_history
Returns: {
    "action": "answer" | "clarify" | "ticket",
    "reasoning": str,
    "search_query": str,      # if action=answer: use this for full retrieve
    "clarify_message": str,   # if action=clarify: send this to user
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

ORCHESTRATOR_PROMPT = """Bạn là bộ não của hệ thống hỗ trợ phần mềm EHC (quản lý bệnh viện).

Nhiệm vụ: Đọc câu hỏi của người dùng, lịch sử hội thoại, và 3 đoạn FAQ tìm được. Quyết định hành động tiếp theo.

---
LỊCH SỬ HỘI THOẠI:
{history}

---
CÂU HỎI HIỆN TẠI: {query}

---
3 ĐOẠN TÌM ĐƯỢC (theo thứ tự liên quan):
{chunks}

---
CÁC FILE HƯỚNG DẪN NGHIỆP VỤ ĐANG CÓ:
{knowledge_topics}

---
HƯỚNG DẪN QUYẾT ĐỊNH (theo thứ tự ưu tiên):

1. [ƯU TIÊN CAO NHẤT] Kiểm tra lịch sử trước:
   Nếu lịch sử KHÔNG rỗng → đã xác lập chủ thể rồi → action=answer, TUYỆT ĐỐI KHÔNG clarify.
   Câu hỏi hiện tại chỉ là follow-up. Dùng search_query = chủ thể từ lịch sử + nội dung câu hỏi mới.
   Ví dụ: lịch sử có "Minipacs" + câu hỏi "cần chuẩn bị gì" → search_query = "cần chuẩn bị gì để kết nối Minipacs"

2. action = "clarify" — CHỈ khi lịch sử RỖNG VÀ query không đề cập chủ thể cụ thể VÀ nhiều chunk có thể phù hợp.
   → clarify_message = liệt kê các trường hợp từ chunks theo danh sách đánh số.
     Kết thúc bằng: "Nếu không có trường hợp nào phù hợp, bạn có thể mô tả chi tiết vấn đề bằng lời của mình."
   → Nếu query chứa " — " ở giữa → đây là follow-up sau clarification. TUYỆT ĐỐI KHÔNG clarify. Chỉ được "answer" hoặc "ticket".

3. action = "answer" — khi lịch sử rỗng nhưng query đề cập rõ chủ thể cụ thể.
   Ví dụ chủ thể cụ thể: "bảng kê", "tài liệu chưa ký", "bệnh án", "phiếu thu", "phiếu khám",
   "giấy ra viện", "bảng kê 6556", "phiếu chỉ định", tên phần mềm cụ thể, v.v.
   → search_query = câu truy vấn tối ưu, tiếng Việt, cụ thể, bỏ từ thừa ("mình", "ấy", "nhỉ", "vậy")

4. action = "ticket" — CHỈ khi đã clarify ít nhất 1 lần mà vẫn không tìm được chunk phù hợp.
   KHÔNG tạo ticket ngay lần đầu khi chunks không match — hãy dùng action="clarify" để hỏi thêm.
   Ngoại lệ: query rõ ràng không liên quan gì đến phần mềm EHC → ticket ngay.

---
CHỌN TOOL TÌM KIẾM (field "tool"):
- "search_manual" — khi:
  * Chunk #1 hoặc #2 có source chứa "hdsd" VÀ score > 0.5 → LUÔN chọn search_manual.
  * User hỏi CÁCH SỬ DỤNG một chức năng: cài đặt, cấu hình, hướng dẫn từng bước, quy trình.
  * Lịch sử đang thảo luận về HDSD/hướng dẫn sử dụng.
  Ví dụ: "cách cài đặt...", "hướng dẫn kết nối...", "làm thế nào để...", "thao tác...", "các bước để..."
- "search_faq" — khi:
  * Các chunk đầu có source="faq" chiếm ưu thế, HOẶC
  * User báo lỗi, hỏi tại sao, hoặc gặp vấn đề không hoạt động.
  Ví dụ: "không in được", "bị lỗi", "tại sao không...", "không đăng nhập được"
- Mặc định: "search_faq" khi không chắc chắn.

---
CÔNG CỤ HỖ TRỢ NGỮ CẢNH NGHIỆP VỤ (field "knowledge_topic"):
1. Nếu fast_chunks đã giải thích được nguyên nhân gốc → KHÔNG cần, để knowledge_topic = "".
2. Chỉ đặt knowledge_topic khi chunks cho thấy vấn đề thuộc lĩnh vực nghiệp vụ cụ thể VÀ user cần hướng dẫn chi tiết hơn.
3. Nếu chưa chắc chắn topic nào phù hợp → để knowledge_topic = "".

---
TRẢ LỜI THEO ĐỊNH DẠNG JSON (không giải thích thêm):
{{
  "action": "answer" | "clarify" | "ticket",
  "tool": "search_faq" | "search_manual",
  "knowledge_topic": "" | "<stem từ danh sách trên>",
  "reasoning": "lý do ngắn gọn",
  "search_query": "...",
  "clarify_message": "..."
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


def orchestrate(query: str, fast_chunks: list, session_history: list = None, retry_count: int = 0) -> dict:
    """
    Call the LLM to decide the next action.

    Returns dict with keys: action, reasoning, search_query, clarify_message.
    Fallback to {"action": "answer", "search_query": query} on any error.
    """

    retry_hint = ""
    if retry_count > 0:
        retry_hint = "\nLẦN THỬ LẠI: Lần retrieve trước không đủ tin cậy. Hãy dùng search_query RỘNG HƠN hoặc đổi sang tool khác (search_manual nếu trước dùng search_faq)."

    prompt = ORCHESTRATOR_PROMPT.format(
        query=query,
        chunks=_format_chunks(fast_chunks),
        history=_format_history(session_history or []),
        knowledge_topics=_format_knowledge_topics(),
    ) + retry_hint

    print(f"[ORCHESTRATOR] Query: \"{query}\"")

    messages = [{"role": "user", "content": prompt}]

    try:
        response = _client.chat.completions.create(
            model=VLLM_MODEL,
            messages=messages,
            max_tokens=300,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
    except APIConnectionError:
        # Retry once after 1s
        print("[ORCHESTRATOR] Connection error, retrying in 1s...")
        time.sleep(1)
        try:
            response = _client.chat.completions.create(
                model=VLLM_MODEL,
                messages=messages,
                max_tokens=300,
                temperature=0.1,
            )
            raw = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[ORCHESTRATOR] Retry failed: {e} → fallback to answer")
            return _fallback_result(query)
    except Exception as e:
        print(f"[ORCHESTRATOR] LLM failed: {e} → fallback to answer")
        return _fallback_result(query)

    print(f"[ORCHESTRATOR] Raw output: {raw[:200]}")

    return _parse_response(raw, query, session_history or [], fast_chunks)


def _parse_response(raw: str, query: str, session_history: list = None, fast_chunks: list = None) -> dict:
    """Parse the LLM JSON response into a structured dict."""
    try:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match:
            raise ValueError("No JSON found in response")
        result = json.loads(match.group())
        action = result.get("action", "answer")
        if action not in ("answer", "clarify", "ticket"):
            action = "answer"

        # Safety override: nếu có lịch sử mà LLM vẫn trả clarify → force answer
        if action == "clarify" and session_history:
            print("[ORCHESTRATOR] Override: có lịch sử nhưng LLM trả clarify → đổi thành answer")
            action = "answer"
            # Đảm bảo search_query không rỗng khi override
            if not result.get("search_query"):
                result["search_query"] = query
            # Nếu fast_chunks có hdsd source → dùng search_manual (FAQ không có nội dung HDSD)
            if fast_chunks and result.get("tool", "search_faq") == "search_faq":
                has_hdsd = any(
                    getattr(c, "metadata", {}).get("source", "").startswith("hdsd")
                    for c in fast_chunks
                )
                if has_hdsd:
                    print("[ORCHESTRATOR] Override tool: hdsd chunk in fast_chunks + history → search_manual")
                    result["tool"] = "search_manual"

        result["action"] = action
        # Dùng 'or' thay setdefault để fallback khi LLM trả "" thay vì bỏ key
        result["search_query"] = result.get("search_query") or query
        result.setdefault("clarify_message", "")
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
        "clarify_message": "",
        "tool": "search_faq",
        "knowledge_topic": "",
    }


if __name__ == "__main__":
    print("=== Orchestrator standalone test ===\n")

    # Simulate a RetrievedChunk-like object for testing
    class FakeChunk:
        def __init__(self, text, subject, source="faq", score=0.5):
            self.text = text
            self.metadata = {"subject": subject, "source": source}
            self.score = score

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

    # Test 4: follow-up with HDSD history
    print("--- Test 4: Follow-up with Minipacs history ---")
    minipacs_history = [
        {"role": "user", "text": "cách kết nối minipacs"},
        {"role": "bot", "text": "Để kết nối Minipacs, vào module PACS Server..."},
    ]
    minipacs_chunks = [
        FakeChunk("Những nội dung cần chuẩn bị...", "Những nội dung cần chuẩn bị", source="hdsd_minipacs", score=0.29),
        FakeChunk("Dự trù vật tư...", "Làm thế nào để dự trù vật tư", source="faq", score=0.36),
    ]
    result = orchestrate("cần chuẩn bị gì không", minipacs_chunks, minipacs_history)
    print(f"Result: {json.dumps(result, ensure_ascii=False, indent=2)}\n")

    print("✓ Orchestrator tests completed.")
