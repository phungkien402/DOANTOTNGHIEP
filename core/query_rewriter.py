"""
Query Rewriter — extracts the core technical problem from colloquial user messages.

Uses the LLM (via vLLM OpenAI-compatible API) to bridge the gap between how
doctors ask questions (short, informal) and how the FAQ is written (formal).

Run standalone: python -m core.query_rewriter
"""

import json as _json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI, APIConnectionError

from config import VLLM_BASE_URL, VLLM_MODEL
from core.generator import LLMUnavailableError
from core.abbreviations import expand_abbreviations

# Module-level client — created once when module is first imported
_client = OpenAI(base_url=f"{VLLM_BASE_URL}/v1", api_key="not-needed")


def _safe_content(response) -> str | None:
    """Return message content or None if missing (Qwen3 thinking overflow)."""
    try:
        return response.choices[0].message.content
    except Exception:
        return None


# Thinking budget per task type — limits Qwen3 reasoning tokens
_BUDGET_FAST = 256    # rewrite, intent (simple classification)
_BUDGET_MEDIUM = 512  # analyze_and_rewrite, intent_analyzer (structured JSON)

SYSTEM_PROMPT = (
    "You are a query understanding assistant for EHC electronic medical record software support. "
    "Your job is to read a user's message and extract the core technical problem they are experiencing, "
    "expressed as a short, clear issue statement that matches how FAQ entries are written.\n\n"
    "Rules:\n"
    "1. Focus on WHAT the problem is, not HOW the user described it.\n"
    "2. Output a short statement (1 sentence), not a question.\n"
    "3. Use technical terms that would appear in a FAQ title.\n"
    "4. If the message is already a clear technical question, keep it as-is.\n"
    "5. Output in Vietnamese. Return only the rewritten query — no explanation.\n"
    "6. Prefix selection:\n"
    "   - Use \"Lỗi...\" ONLY when the input describes a system error, crash, malfunction, or "
    "unexpected behavior (e.g. \"bị lỗi\", \"không lên\", \"xoay hoài\", \"bắt update\").\n"
    "   - Use \"Cách...\" when the input is a how-to or navigation question "
    "(e.g. \"bấm vào đâu\", \"làm sao\", \"ở đâu\", \"muốn làm\").\n"
    "   - Use \"Vị trí...\" when asking where something is located in the UI."
)

FEW_SHOT_EXAMPLES = [
    ("merge patient records how?", "Cách gộp mã bệnh nhân"),
    ("xử trí cứ xoay hoài không dừng", "Lỗi màn hình xoay mãi khi xử trí bệnh nhân"),
    ("phần mềm bắt update mới vô được", "Lỗi phần mềm bắt buộc cập nhật mới đăng nhập được"),
    ("in phiếu không lên form view làm sao", "Lỗi in phiếu không hiển thị form view"),
    ("in giấy ra viện lại ở đâu", "Vị trí in lại giấy ra viện trong hệ thống"),
    ("xem giúp tôi hẹn bệnh nhân nhưng hệ thống tự nhảy vào thứ 7 chủ nhật", "Lỗi module đặt hẹn tự chọn sai ngày trong tuần"),
    ("BN ra viện rồi muốn sửa thông tin", "Cách sửa thông tin bệnh nhân sau khi đã xử trí ra viện"),
    ("muốn hủy nhập viện thì bấm vào đâu", "Cách hủy nhập viện trong hệ thống EHC"),
]


def _build_messages(text: str) -> list[dict]:
    """Build the message list with system prompt and few-shot examples."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_ex, assistant_ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": user_ex})
        messages.append({"role": "assistant", "content": assistant_ex})
    messages.append({"role": "user", "content": text})
    return messages


INTENT_PROMPT = (
    "Bạn là trợ lý phân tích ý định người dùng cho phần mềm bệnh án điện tử EHC.\n"
    "Đọc tin nhắn của người dùng và mô tả ngắn gọn (1 câu) vấn đề họ đang gặp phải.\n"
    "Viết ở ngôi thứ 3, ví dụ: \"Bác sĩ đang gặp lỗi màn hình xoay liên tục khi xử trí bệnh nhân.\"\n"
    "Chỉ trả về 1 câu mô tả — không giải thích thêm."
)

CONTEXTUAL_INTENT_PROMPT = (
    "Bạn là trợ lý hiểu phần mềm EHC. Dựa vào tài liệu tham khảo, "
    "hãy mô tả ngắn gọn vấn đề người dùng đang gặp (1 câu).\n"
    "Viết ở ngôi thứ 3, ví dụ: \"Bác sĩ đang gặp lỗi màn hình xoay liên tục khi xử trí bệnh nhân.\"\n"
    "Chỉ trả về 1 câu mô tả — không giải thích thêm."
)


def analyze_intent(query: str, chunks: list = None) -> str | None:
    """
    Analyze user intent — returns a 1-sentence Vietnamese description of the
    user's problem for internal use (injected into generator prompt).

    If chunks are provided, uses them as context for grounded intent analysis.
    If not, falls back to blind analysis (no context).
    Returns None if vLLM is unavailable.
    """
    if chunks:
        print(f"[INTENT] Contextual analysis with {len(chunks)} chunks: \"{query}\"")
        # Build context from chunks
        chunk_texts = []
        for i, c in enumerate(chunks, 1):
            subject = c.metadata.get("subject", "")
            text = c.text or c.metadata.get("description", "")
            chunk_texts.append(f"{i}. {subject}: {text}")
        context_block = "\n".join(chunk_texts)

        user_content = f"Tài liệu tham khảo:\n{context_block}\n\nCâu hỏi: {query}"
        system_prompt = CONTEXTUAL_INTENT_PROMPT
    else:
        print(f"[INTENT] Blind analysis (no chunks): \"{query}\"")
        user_content = query
        system_prompt = INTENT_PROMPT

    try:
        response = _client.chat.completions.create(
            model=VLLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=600,
            temperature=0.1,
            extra_body={"chat_template_kwargs": {"enable_thinking": True, "thinking_budget": _BUDGET_FAST}},
        )

        raw = _safe_content(response)
        if raw is None:
            return None
        intent = raw.strip()
        print(f"[INTENT] Result: \"{intent}\"")
        return intent

    except APIConnectionError as e:
        # Retry once after 1s — vLLM may be busy
        print(f"[INTENT] Connection error, retrying in 1s...")
        time.sleep(1)
        try:
            response = _client.chat.completions.create(
                model=VLLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=600,
                temperature=0.1,
                extra_body={"chat_template_kwargs": {"enable_thinking": True, "thinking_budget": _BUDGET_FAST}},
            )
            raw = _safe_content(response)
            if raw is None:
                return None
            intent = raw.strip()
            print(f"[INTENT] Result (retry): \"{intent}\"")
            return intent
        except Exception:
            print(f"[INTENT] Retry failed, skipping intent analysis")
            return None

    except Exception as e:
        print(f"[INTENT] vLLM unavailable ({type(e).__name__}), skipping intent analysis")
        return None


def rewrite(text: str) -> str:
    """
    Rewrite a colloquial question into a clear intent statement.
    Returns the rewritten query string.
    If vLLM is unavailable, returns the original text as-is.
    """
    text = expand_abbreviations(text)
    print(f"[REWRITER] Original : \"{text}\"")

    try:
        response = _client.chat.completions.create(
            model=VLLM_MODEL,
            messages=_build_messages(text),
            max_tokens=700,
            temperature=0.1,
            extra_body={"chat_template_kwargs": {"enable_thinking": True, "thinking_budget": _BUDGET_FAST}},
        )

        raw = _safe_content(response)
        if raw is None:
            raise LLMUnavailableError("content=None (thinking overflow)")
        rewritten = raw.strip()
        print(f"[REWRITER] Rewritten: \"{rewritten}\"")
        return rewritten

    except APIConnectionError as e:
        # Retry once after 1s — vLLM may be busy
        print(f"[REWRITER] Connection error, retrying in 1s...")
        time.sleep(1)
        try:
            response = _client.chat.completions.create(
                model=VLLM_MODEL,
                messages=_build_messages(text),
                max_tokens=700,
                temperature=0.1,
                extra_body={"chat_template_kwargs": {"enable_thinking": True, "thinking_budget": _BUDGET_FAST}},
            )
            raw = _safe_content(response)
            if raw is None:
                raise LLMUnavailableError("content=None (thinking overflow)")
            rewritten = raw.strip()
            print(f"[REWRITER] Rewritten (retry): \"{rewritten}\"")
            return rewritten
        except LLMUnavailableError:
            raise
        except Exception as retry_e:
            print(f"[REWRITER] Retry failed ({type(retry_e).__name__}), raising LLMUnavailableError")
            raise LLMUnavailableError(str(retry_e)) from retry_e

    except LLMUnavailableError:
        raise

    except Exception as e:
        print(f"[REWRITER] vLLM unavailable ({type(e).__name__}), raising LLMUnavailableError")
        raise LLMUnavailableError(str(e)) from e


ANALYZE_AND_REWRITE_PROMPT = (
    "Bạn là trợ lý EHC. Dựa vào tài liệu tham khảo (nếu có), hãy thực hiện 3 việc:\n"
    "1. Mô tả ngắn gọn vấn đề người dùng đang gặp (1 câu, ngôi thứ 3)\n"
    "2. Viết lại câu hỏi thành query tìm kiếm ngắn gọn, formal tiếng Việt\n"
    "3. Đánh giá xem tài liệu tham khảo có đủ để trả lời câu hỏi không\n"
    "Trả về đúng 3 dòng theo format:\n"
    "INTENT: <mô tả vấn đề>\n"
    "QUERY: <query tìm kiếm>\n"
    "ANSWERABLE: <yes | no | unclear>\n\n"
    "Hướng dẫn cho ANSWERABLE:\n"
    "- yes: tài liệu tham khảo trả lời trực tiếp câu hỏi\n"
    "- unclear: câu hỏi quá mơ hồ, thiếu thông tin (module nào? lỗi gì?), hoặc tài liệu chỉ liên quan một phần\n"
    "- no: tài liệu hoàn toàn không liên quan, hoặc không có tài liệu tham khảo\n\n"
    "Nếu không có tài liệu tham khảo nhưng lịch sử hội thoại đã có danh sách đánh số, "
    "hãy dùng lịch sử đó để xác định intent. ANSWERABLE=yes nếu intent đã rõ từ lịch sử."
)


def _parse_analyze_response(response_text: str, original_query: str) -> tuple[str | None, str, str]:
    """Parse INTENT/QUERY/ANSWERABLE response. Returns (intent, rewritten_query, answerable)."""
    intent = None
    rewritten = original_query
    answerable = "unclear"  # safe default

    for line in response_text.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("INTENT:"):
            intent = line[len("INTENT:"):].strip()
        elif line.upper().startswith("QUERY:"):
            rewritten = line[len("QUERY:"):].strip()
        elif line.upper().startswith("ANSWERABLE:"):
            val = line[len("ANSWERABLE:"):].strip().lower()
            if val in ("yes", "no", "unclear"):
                answerable = val

    if not rewritten:
        rewritten = original_query

    return intent, rewritten, answerable


def analyze_and_rewrite(query: str, chunks: list = None, session_history: list = None) -> tuple[str | None, str, str]:
    """
    Combined intent analysis + query rewrite + answerability check in a single vLLM call.
    Returns (intent, rewritten_query, answerable).

    answerable is one of: "yes", "no", "unclear"
    If chunks are provided, injects them as context for grounded analysis.
    If session_history is provided, injects recent turns so LLM can resolve
    short follow-up replies (e.g. "2") in context.
    Graceful degradation: returns (None, original_query, "unclear") if vLLM is unavailable.
    """
    query = expand_abbreviations(query)
    print(f"[ANALYZE+REWRITE] Query: \"{query}\"")

    # Build history block from recent conversation turns (last 2 exchanges = 4 messages)
    history_block = ""
    if session_history:
        recent = session_history[-4:]
        lines = []
        for turn in recent:
            role = "Người dùng" if turn["role"] == "user" else "Trợ lý"
            lines.append(f"{role}: {turn['text']}")
        history_block = "Lịch sử hội thoại:\n" + "\n".join(lines) + "\n\n"
        print(f"[ANALYZE+REWRITE] Injecting {len(recent)} history turns")

    # Build user content with optional chunk context
    if chunks:
        print(f"[ANALYZE+REWRITE] Using {len(chunks)} chunks as context")
        chunk_texts = []
        for i, c in enumerate(chunks, 1):
            subject = c.metadata.get("subject", "")
            text = c.text or c.metadata.get("description", "")
            chunk_texts.append(f"{i}. {subject}: {text}")
        context_block = "\n".join(chunk_texts)
        user_content = f"{history_block}Tài liệu tham khảo:\n{context_block}\n\nCâu hỏi: {query}"
    else:
        print(f"[ANALYZE+REWRITE] No chunks (blind mode)")
        user_content = f"{history_block}Câu hỏi: {query}"

    messages = [
        {"role": "system", "content": ANALYZE_AND_REWRITE_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        response = _client.chat.completions.create(
            model=VLLM_MODEL,
            messages=messages,
            max_tokens=900,
            temperature=0.1,
            extra_body={"chat_template_kwargs": {"enable_thinking": True, "thinking_budget": _BUDGET_MEDIUM}},
        )

        raw = _safe_content(response)
        if raw is None:
            raise LLMUnavailableError("content=None (thinking overflow)")
        raw = raw.strip()
        intent, rewritten, answerable = _parse_analyze_response(raw, query)
        print(f"[ANALYZE+REWRITE] Intent: \"{intent}\"")
        print(f"[ANALYZE+REWRITE] Rewritten: \"{rewritten}\"")
        print(f"[ANALYZE+REWRITE] Answerable: \"{answerable}\"")
        return intent, rewritten, answerable

    except APIConnectionError:
        # Retry once after 1s — vLLM may be busy
        print(f"[ANALYZE+REWRITE] Connection error, retrying in 1s...")
        time.sleep(1)
        try:
            response = _client.chat.completions.create(
                model=VLLM_MODEL,
                messages=messages,
                max_tokens=900,
                temperature=0.1,
                extra_body={"chat_template_kwargs": {"enable_thinking": True, "thinking_budget": _BUDGET_MEDIUM}},
            )
            raw = _safe_content(response)
            if raw is None:
                raise LLMUnavailableError("content=None (thinking overflow)")
            raw = raw.strip()
            intent, rewritten, answerable = _parse_analyze_response(raw, query)
            print(f"[ANALYZE+REWRITE] Intent (retry): \"{intent}\"")
            print(f"[ANALYZE+REWRITE] Rewritten (retry): \"{rewritten}\"")
            print(f"[ANALYZE+REWRITE] Answerable (retry): \"{answerable}\"")
            return intent, rewritten, answerable
        except LLMUnavailableError:
            raise
        except Exception as retry_e:
            print(f"[ANALYZE+REWRITE] Retry failed ({type(retry_e).__name__}), raising LLMUnavailableError")
            raise LLMUnavailableError(str(retry_e)) from retry_e

    except LLMUnavailableError:
        raise

    except Exception as e:
        print(f"[ANALYZE+REWRITE] vLLM unavailable ({type(e).__name__}), raising LLMUnavailableError")
        raise LLMUnavailableError(str(e)) from e


# ---------------------------------------------------------------------------
# Pre-retrieval intent analyzer
# ---------------------------------------------------------------------------

INTENT_ANALYZER_PROMPT = (
    "Bạn là trợ lý phân tích ý định cho phần mềm bệnh án điện tử EHC.\n\n"
    "Đọc tin nhắn của người dùng (và lịch sử hội thoại nếu có), rồi quyết định:\n"
    "Mặc định dùng action=clarify TRỪ KHI tin nhắn có ít nhất MỘT trong:\n"
    "- Tên module hoặc màn hình cụ thể (vd: module khám bệnh, màn hình xử trí)\n"
    "- Thông báo lỗi cụ thể (vd: báo lỗi kết nối, hiện thông báo đỏ)\n"
    "- Thao tác cụ thể đã làm (vd: ấn in vỏ bệnh án, bấm lưu xong thì...)\n"
    "- Loại tài liệu cụ thể (vd: bảng kê 6556, giấy ra viện, phiếu khám)\n"
    "Nếu chỉ mô tả chung chung như 'không in được', 'bị lỗi', 'không vào được' "
    "mà không kèm thông tin nào trên → action=clarify\n"
    "không thêm các ví dụ vào prompt vì đó chỉ là ví dụ.\n"
    "Trả về JSON hợp lệ (không markdown, không giải thích thêm):\n"
    "{\n"
    '  "action": "proceed" | "clarify",\n'
    '  "rewritten_query": "<câu truy vấn tiếng Việt, ngắn gọn, tối ưu cho vector search>",\n'
    '  "clarify_message": "<câu hỏi ngắn — CHỈ điền khi action=clarify, để trống nếu proceed>"\n'
    "}\n\n"
    "Quy tắc rewritten_query:\n"
    "- Encode những gì bạn hiểu từ tin nhắn của user\n"
    "- Ngắn gọn, formal, bỏ từ thừa (mình, ạ, nhỉ, vậy)\n"
    '- Prefix "Lỗi..." nếu là lỗi hệ thống, "Cách..." nếu là hướng dẫn thao tác\n'
    "- Bắt buộc điền kể cả khi action=clarify (dự đoán tốt nhất có thể)\n\n"
    "Quy tắc clarify_message:\n"
    "- Chỉ hỏi dựa trên những gì user đã nói — KHÔNG suy diễn từ nguồn khác\n"
    "- Hỏi cụ thể: lỗi gì? module nào? thao tác nào? thấy thông báo gì?\n"
    "- KHÔNG đề cập 'hướng dẫn trước đó' nếu lịch sử không có bước nào\n"
    "- KHÔNG giả định user đang dùng module cụ thể nếu họ chưa nói\n"
    "- Tối đa 2 câu"
)


def analyze_intent_pre_retrieval(query: str, session_history: list = None) -> dict:
    """
    Pre-retrieval intent analysis — runs BEFORE FastRetriever.

    Analyzes the user's query (+ recent history) to decide:
    - action="proceed": intent is clear enough → continue to retrieval
    - action="clarify": query is too vague → ask user for more details

    Returns:
        {
            "action": "proceed" | "clarify",
            "rewritten_query": str,    # always filled (best guess even when clarify)
            "clarify_message": str,    # only when action=clarify
        }

    Falls back to {"action": "proceed", "rewritten_query": query} on any LLM error.
    """
    expanded = expand_abbreviations(query)
    print(f"[INTENT_ANALYZER] Query: \"{expanded}\"")

    # Build history block from recent turns (last 2 exchanges = 4 entries)
    history_block = ""
    if session_history:
        recent = session_history[-4:]
        lines = []
        for turn in recent:
            role = "Người dùng" if turn.get("role") == "user" else "Trợ lý"
            text = turn.get("text", turn.get("content", ""))[:150]
            lines.append(f"{role}: {text}")
        history_block = "Lịch sử hội thoại gần đây:\n" + "\n".join(lines) + "\n\n"

    user_content = f"{history_block}Tin nhắn: {expanded}"

    messages = [
        {"role": "system", "content": INTENT_ANALYZER_PROMPT},
        {"role": "user", "content": user_content},
    ]

    def _call():
        return _client.chat.completions.create(
            model=VLLM_MODEL,
            messages=messages,
            max_tokens=900,
            temperature=0.1,
            extra_body={"chat_template_kwargs": {"enable_thinking": True, "thinking_budget": _BUDGET_MEDIUM}},
        )

    try:
        resp = _call()
        raw = _safe_content(resp)
        if raw is None:
            raise ValueError("content=None (thinking overflow)")
        raw = raw.strip()
        print(f"[INTENT_ANALYZER] Raw: {raw}")
        data = _json.loads(raw)
        action = data.get("action", "proceed")
        rewritten = data.get("rewritten_query") or expanded
        clarify_msg = data.get("clarify_message", "")
        print(f"[INTENT_ANALYZER] action={action} | rewritten=\"{rewritten}\"")
        return {"action": action, "rewritten_query": rewritten, "clarify_message": clarify_msg}

    except APIConnectionError:
        print("[INTENT_ANALYZER] Connection error, retrying in 1s...")
        time.sleep(1)
        try:
            resp = _call()
            raw = _safe_content(resp)
            if raw is None:
                raise ValueError("content=None (thinking overflow)")
            raw = raw.strip()
            data = _json.loads(raw)
            action = data.get("action", "proceed")
            rewritten = data.get("rewritten_query") or expanded
            clarify_msg = data.get("clarify_message", "")
            print(f"[INTENT_ANALYZER] action={action} (retry) | rewritten=\"{rewritten}\"")
            return {"action": action, "rewritten_query": rewritten, "clarify_message": clarify_msg}
        except Exception:
            pass

    except Exception as e:
        print(f"[INTENT_ANALYZER] Error: {e}")

    # Fallback: always proceed so the pipeline never gets stuck
    print("[INTENT_ANALYZER] Fallback → proceed")
    return {"action": "proceed", "rewritten_query": expanded, "clarify_message": ""}


if __name__ == "__main__":
    test_queries = [
        "merge patient records how?",
        "in phiếu khám ở đâu",
        "thuốc hết tồn kho làm sao",
        "xử trí cứ xoay hoài không dừng",
        "phần mềm bắt update mới vô được",
    ]
    for q in test_queries:
        result = rewrite(q)
        print()
