# FIX: Generator nói "Mình chưa tìm thấy" dù chunks score cao

## Root cause

`_build_user_prompt()` ở `core/generator.py` lines 71-77 đưa `history[-4:]` vào prompt dưới dạng CONVERSATION HISTORY. Khi flow là:

1. User hỏi → Orchestrator clarify → Bot trả về *"1. Bạn không in được... 2. Lỗi không in..."*
2. User follow-up → Query enriched → Reranker tìm chunk score=0.9487

Thì CONVERSATION HISTORY trong prompt có dạng:
```
User: in bảng kê không được
Assistant: Bạn gặp vấn đề nào sau? 1. Không in được... 2. Lỗi máy in...
User: in bảng kê không được — bảng kê
```

LLM (Qwen2.5-7B) thấy "Assistant" trước đó đưa ra câu hỏi lại (không có câu trả lời), nên suy diễn rằng "bot chưa tìm được" → dùng hedge phrase dù [PRIMARY REFERENCE] score=0.9487.

## Fix — 2 thay đổi trong `~/DOANTN/core/generator.py`

### Fix 1: Thêm helper `_is_clarify_turn()` và filter history

Thêm hàm sau trước `_build_user_prompt`:

```python
def _is_clarify_turn(turn: dict) -> bool:
    """True nếu đây là bot turn clarify (có numbered options), filter trước khi đưa vào prompt."""
    if turn.get("role") != "bot":
        return False
    text = turn.get("text", "")
    # Clarify messages có format "1." hoặc "1)" ít nhất 2 lần
    count = sum(1 for i in range(1, 5) if f"{i}." in text or f"{i})" in text)
    return count >= 2
```

Thay đoạn history trong `_build_user_prompt` (lines 71-77):

Trước:
```python
    # Add last 2 turns of history if available
    if history:
        recent = history[-4:]  # last 2 turns = 4 entries (user+bot x2)
        history_lines = []
        for turn in recent:
            role = "User" if turn["role"] == "user" else "Assistant"
            history_lines.append(f"{role}: {turn['text']}")
        parts.append("\nCONVERSATION HISTORY (for context only):\n" + "\n".join(history_lines))
```

Sau:
```python
    # Add last 2 turns of history if available — filter clarify bot turns
    if history:
        recent = history[-4:]  # last 2 turns = 4 entries (user+bot x2)
        # Bỏ clarify bot turns: chứa numbered options, làm LLM nghĩ bot chưa có câu trả lời
        filtered = [t for t in recent if not _is_clarify_turn(t)]
        if filtered:
            history_lines = []
            for turn in filtered:
                role = "User" if turn["role"] == "user" else "Assistant"
                history_lines.append(f"{role}: {turn['text']}")
            parts.append("\nCONVERSATION HISTORY (for context only):\n" + "\n".join(history_lines))
```

### Fix 2: Strengthen SYSTEM_PROMPT rule 2

Thay rule 2 trong `SYSTEM_PROMPT`:

Trước:
```python
    "2. CONTEXT có thể chứa đường dẫn ngắn hoặc hướng dẫn tóm tắt — hãy diễn giải "
    "thành lời hướng dẫn tự nhiên, dễ hiểu. "
    "Chỉ nói \"Mình chưa tìm thấy hướng dẫn cho vấn đề này.\" nếu CONTEXT hoàn toàn "
    "không liên quan đến câu hỏi.\n"
```

Sau:
```python
    "2. CONTEXT có thể chứa đường dẫn ngắn hoặc hướng dẫn tóm tắt — hãy diễn giải "
    "thành lời hướng dẫn tự nhiên, dễ hiểu. "
    "KHÔNG BAO GIỜ nói \"Mình chưa tìm thấy hướng dẫn cho vấn đề này.\" khi [PRIMARY REFERENCE] "
    "có nội dung liên quan đến câu hỏi. "
    "Chỉ dùng câu đó khi TẤT CẢ chunks trong CONTEXT đều hoàn toàn không liên quan.\n"
```

---

## Steps

```bash
# 1. Edit generator.py
nano ~/DOANTN/core/generator.py

# 2. Restart worker
sudo systemctl restart ehc-worker

# 3. Test: gửi câu hỏi clarify flow qua Telegram
#    TEST 3 trong TELEGRAM_TEST_CASES.md

# 4. Check log
sudo journalctl -u ehc-worker -n 30 | grep -E "GENERATOR|conf="
```

Expected: Generator không còn prefix "Mình chưa tìm thấy" khi conf > 0.8.

---

## Claude Code instruction (copy-paste)

```
In ~/DOANTN/core/generator.py:

1. Add this helper function right before `_build_user_prompt`:

def _is_clarify_turn(turn: dict) -> bool:
    """True if this is a clarify bot turn (has numbered options like 1. / 2.) — filter before injecting into LLM prompt."""
    if turn.get("role") != "bot":
        return False
    text = turn.get("text", "")
    count = sum(1 for i in range(1, 5) if f"{i}." in text or f"{i})" in text)
    return count >= 2

2. In `_build_user_prompt`, replace the history block (lines ~71-77):

OLD:
    # Add last 2 turns of history if available
    if history:
        recent = history[-4:]  # last 2 turns = 4 entries (user+bot x2)
        history_lines = []
        for turn in recent:
            role = "User" if turn["role"] == "user" else "Assistant"
            history_lines.append(f"{role}: {turn['text']}")
        parts.append("\nCONVERSATION HISTORY (for context only):\n" + "\n".join(history_lines))

NEW:
    # Add last 2 turns of history if available — filter clarify bot turns
    if history:
        recent = history[-4:]  # last 2 turns = 4 entries (user+bot x2)
        # Filter out clarify bot turns: they have numbered options and make LLM think bot couldn't answer
        filtered = [t for t in recent if not _is_clarify_turn(t)]
        if filtered:
            history_lines = []
            for turn in filtered:
                role = "User" if turn["role"] == "user" else "Assistant"
                history_lines.append(f"{role}: {turn['text']}")
            parts.append("\nCONVERSATION HISTORY (for context only):\n" + "\n".join(history_lines))

3. In SYSTEM_PROMPT, replace rule 2:

OLD:
    "2. CONTEXT có thể chứa đường dẫn ngắn hoặc hướng dẫn tóm tắt — hãy diễn giải "
    "thành lời hướng dẫn tự nhiên, dễ hiểu. "
    "Chỉ nói \"Mình chưa tìm thấy hướng dẫn cho vấn đề này.\" nếu CONTEXT hoàn toàn "
    "không liên quan đến câu hỏi.\n"

NEW:
    "2. CONTEXT có thể chứa đường dẫn ngắn hoặc hướng dẫn tóm tắt — hãy diễn giải "
    "thành lời hướng dẫn tự nhiên, dễ hiểu. "
    "KHÔNG BAO GIỜ nói \"Mình chưa tìm thấy hướng dẫn cho vấn đề này.\" khi [PRIMARY REFERENCE] "
    "có nội dung liên quan đến câu hỏi. "
    "Chỉ dùng câu đó khi TẤT CẢ chunks trong CONTEXT đều hoàn toàn không liên quan.\n"

Then: sudo systemctl restart ehc-worker
```
