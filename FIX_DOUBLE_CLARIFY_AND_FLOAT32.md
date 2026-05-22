# FIX: Orchestrator clarify 2 lần + Float32 trên ticket path

## Bug 1 — Float32 trên ticket/fallback path

### Root cause
`round(numpy.float32(x), n)` trong Python 3 trả về `numpy.float32`, KHÔNG phải Python float.
Vì vậy dù dùng `round()` trong tl_event, giá trị vẫn là float32 khi tl_finish() gọi json.dumps.

`_NumpySafeEncoder` đã được add vào trace_logger.py nhưng cần xác nhận lại — ticket path là path đầu tiên test sau khi fix.

### Fix — xác nhận trace_logger.py đúng

Kiểm tra file `~/DOANTN/core/trace_logger.py`:

```bash
grep -n "NumpySafe\|import numpy\|cls=" ~/DOANTN/core/trace_logger.py
```

Nếu output trống → fix chưa được apply. Apply lại:

```python
# Thêm ở đầu file (sau import json):
import numpy as np

class _NumpySafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
```

Thay **TẤT CẢ** `json.dumps(...)` trong trace_logger.py thành:
```python
json.dumps(..., cls=_NumpySafeEncoder)
```

---

## Bug 2 — Orchestrator clarify 2 lần

### Root cause
Sau khi enriched query (`original — follow-up`), fast_chunks vẫn không liên quan (score 0.38-0.39).
Orchestrator thấy chunks tệ + không nhận ra "đã clarify 1 lần" vì rule trong prompt yếu → clarify lại.

### Fix A — Prompt Orchestrator (core/orchestrator.py)

Thêm rule mạnh hơn vào `ORCHESTRATOR_PROMPT`, ngay sau dòng "KHÔNG clarify nếu lịch sử...":

Tìm đoạn:
```
   → KHÔNG clarify nếu lịch sử cho thấy đã hỏi lại 1 lần → dùng action="answer" hoặc "ticket"
```

Thay bằng:
```
   → KHÔNG clarify nếu lịch sử cho thấy đã hỏi lại 1 lần → dùng action="answer" hoặc "ticket"
   → NẾU query chứa " — " ở giữa (ví dụ: "câu hỏi gốc — câu trả lời của user"), đây là follow-up
     sau clarification. TUYỆT ĐỐI KHÔNG action="clarify". Chỉ được "answer" hoặc "ticket".
```

### Fix B — Hard guard trong node_orchestrator (core/langgraph_agent.py)

Trong hàm `node_orchestrator`, sau khi lấy `action` từ orchestrate(), thêm guard:

Tìm đoạn:
```python
    action = result["action"]
    search_query = result.get("search_query", query)
    clarify_msg = result.get("clarify_message", "")
    reasoning = result.get("reasoning", "")
```

Thêm ngay sau:
```python
    # Guard: nếu session đang awaiting (đã clarify 1 lần) mà orchestrator vẫn muốn clarify lại
    # → override sang ticket (không có đủ thông tin để trả lời)
    session_id = state.get("session_id", "")
    if action == "clarify" and _session_mgr and _session_mgr.is_awaiting_clarification(session_id):
        print(f"[ORCHESTRATOR] Guard: overriding clarify→ticket (already awaiting)")
        action = "ticket"
        result["action"] = "ticket"
```

Fix B là hard guard, đảm bảo bot không bao giờ clarify 2 lần dù prompt LLM ra sao.

---

## Steps

```bash
# 1. Kiểm tra float32 fix
grep -n "NumpySafe\|cls=" ~/DOANTN/core/trace_logger.py

# 2. Apply fix nếu cần
nano ~/DOANTN/core/trace_logger.py

# 3. Apply Fix A + B
nano ~/DOANTN/core/orchestrator.py
nano ~/DOANTN/core/langgraph_agent.py

# 4. Restart
sudo systemctl restart ehc-worker

# 5. Test lại: gửi câu hỏi không có trong FAQ → bot nên hỏi lại 1 lần → turn 2 → tạo ticket
```

Expected behavior sau fix:
```
Turn 1: "câu hỏi không có trong FAQ" → Bot: hỏi lại (clarify)
Turn 2: "trả lời của user"            → Bot: tạo ticket (không clarify lần 2)
```

---

## Claude Code instruction (copy-paste)

```
1. In ~/DOANTN/core/trace_logger.py:
   - Check if _NumpySafeEncoder class exists. If not, add:
     import numpy as np (after import json)
     class _NumpySafeEncoder(json.JSONEncoder):
         def default(self, obj):
             if isinstance(obj, np.floating): return float(obj)
             if isinstance(obj, np.integer): return int(obj)
             if isinstance(obj, np.ndarray): return obj.tolist()
             return super().default(obj)
   - Replace ALL json.dumps(...) with json.dumps(..., cls=_NumpySafeEncoder)

2. In ~/DOANTN/core/orchestrator.py ORCHESTRATOR_PROMPT:
   Find: "KHÔNG clarify nếu lịch sử cho thấy đã hỏi lại 1 lần → dùng action="answer" hoặc "ticket""
   Replace with:
   "   → KHÔNG clarify nếu lịch sử cho thấy đã hỏi lại 1 lần → dùng action="answer" hoặc "ticket"\n   → NẾU query chứa " — " ở giữa, đây là follow-up sau clarification. TUYỆT ĐỐI KHÔNG action="clarify"."

3. In ~/DOANTN/core/langgraph_agent.py, in node_orchestrator(), after:
     action = result["action"]
     search_query = result.get("search_query", query)
     clarify_msg = result.get("clarify_message", "")
     reasoning = result.get("reasoning", "")
   Add:
     session_id = state.get("session_id", "")
     if action == "clarify" and _session_mgr and _session_mgr.is_awaiting_clarification(session_id):
         print(f"[ORCHESTRATOR] Guard: overriding clarify→ticket (already awaiting)")
         action = "ticket"
         result["action"] = "ticket"

4. sudo systemctl restart ehc-worker
```
