# REFACTOR: Add pre-retrieval IntentAnalyzer node

_Generated: 2026-05-23_

## Goal

Move the clarify decision **before** FastRetriever.
Currently the Orchestrator decides clarify/answer/ticket **after** seeing chunks — causing the
clarify message to leak chunk titles (e.g. bot assumes "bệnh án" when user only said "không in được").

### New graph flow

```
query_analyzer → chat_fallback         (off-topic)
             → intent_analyzer         (EHC-related)
                 → END                 (action=clarify — stop before retrieval)
                 → fast_retriever      (action=proceed — intent is clear)
                     → orchestrator    (answer | ticket — no more clarify)
                         → full_retriever → synthesizer → generator → END
                         → ticket_creator → END
```

### LLM call count: unchanged (4 total)
| Old | New |
|-----|-----|
| IntentGuard | IntentGuard |
| QueryAnalyzer (rewrite only) | IntentAnalyzer (understand + rewrite + clarify decision) |
| Orchestrator (clarify/answer/ticket) | Orchestrator (answer/ticket only) |
| Generator | Generator |

---

## Change 1 — Add `analyze_intent_pre_retrieval()` to `core/query_rewriter.py`

Add the following **after the existing imports**, before `SYSTEM_PROMPT`:

```python
import json as _json
```

Add the following **at the end of the file**, before `if __name__ == "__main__":`:

```python
# ---------------------------------------------------------------------------
# Pre-retrieval intent analyzer
# ---------------------------------------------------------------------------

INTENT_ANALYZER_PROMPT = (
    "Bạn là trợ lý phân tích ý định cho phần mềm bệnh án điện tử EHC.\n\n"
    "Đọc tin nhắn của người dùng (và lịch sử hội thoại nếu có), rồi quyết định:\n"
    "- Nếu tin nhắn có ít nhất MỘT trong: thông báo lỗi cụ thể / tên module hoặc màn hình / "
    "thao tác đã thực hiện / loại tài liệu liên quan → action=proceed\n"
    "- Nếu tin nhắn quá mơ hồ, không đủ để tìm kiếm → action=clarify\n\n"
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
            max_tokens=200,
            temperature=0.1,
        )

    try:
        resp = _call()
        raw = resp.choices[0].message.content.strip()
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
            raw = resp.choices[0].message.content.strip()
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
```

---

## Change 2 — Add `node_intent_analyzer` to `core/langgraph_agent.py`

### 2a. Add import at top of file (with other core imports)

```python
from core.query_rewriter import analyze_intent_pre_retrieval
```

### 2b. Add new node function

Insert after `node_query_analyzer` (around line 143), before `node_tool_router`:

```python
def node_intent_analyzer(state: AgentState) -> dict:
    """
    Pre-retrieval intent analysis.
    Understands what the user wants and rewrites the query.
    If the query is too vague → clarify immediately (before any retrieval).
    If intent is clear → proceed to FastRetriever.
    """
    query = state["query"]
    session_id = state.get("session_id", "")
    session_history = state.get("session_history", [])
    trace_id = state.get("trace_id", "")
    print(f"[AGENT] Node: IntentAnalyzer | query=\"{query}\"")

    t_start = time.time()
    result = analyze_intent_pre_retrieval(query, session_history)
    elapsed = (time.time() - t_start) * 1000

    action = result["action"]
    rewritten_query = result.get("rewritten_query") or query
    clarify_msg = result.get("clarify_message", "")

    tl_event(trace_id, "IntentAnalyzer", "decision", {
        "action": action,
        "rewritten_query": rewritten_query,
        "duration_ms": round(elapsed, 1),
    }, duration_ms=round(elapsed, 1))

    if action == "clarify":
        # Guard: max 2 clarifications — force proceed after that
        if _session_mgr:
            clarify_count = _session_mgr.get_clarify_count(session_id)
            if clarify_count >= 2:
                print(f"[INTENT_ANALYZER] Guard: max clarify reached ({clarify_count}) → proceed")
                action = "proceed"
            else:
                _session_mgr.increment_clarify_count(session_id)
                _session_mgr.set_awaiting_clarification(session_id, True)
                print(f"[INTENT_ANALYZER] Clarify #{clarify_count + 1}: \"{clarify_msg}\"")

    if action == "clarify":
        return {
            "answer": clarify_msg,
            "intent": "clarify",
            "tool_called": "clarifier",
            "rewritten_query": rewritten_query,
        }
    else:
        # Clear awaiting state if we are now proceeding
        if _session_mgr and _session_mgr.is_awaiting_clarification(session_id):
            _session_mgr.set_awaiting_clarification(session_id, False)
        print(f"[INTENT_ANALYZER] Proceed | rewritten=\"{rewritten_query}\"")
        return {
            "rewritten_query": rewritten_query,
            "tool_called": "fast_retriever",
        }
```

### 2c. Simplify `node_orchestrator` — remove clarify handling

In `node_orchestrator`, find and **remove** this entire block:

```python
    if action == "clarify" and _session_mgr:
        clarify_count = _session_mgr.get_clarify_count(session_id)
        if clarify_count >= 2:
            print(f"[ORCHESTRATOR] Guard: overriding clarify→ticket (clarify_count={clarify_count})")
            action = "ticket"
            result["action"] = "ticket"
        else:
            _session_mgr.increment_clarify_count(session_id)
            print(f"[ORCHESTRATOR] Allowing clarify #{clarify_count + 1}")
```

Also remove the `clarify` branch in the return block:

```python
    if action == "clarify":
        if _session_mgr:
            _session_mgr.set_awaiting_clarification(session_id, True)
            _session_mgr.set_fast_chunks(session_id, fast_chunks)
        return {
            "answer": clarify_msg,
            "intent": "clarify",
            "tool_called": "clarifier",
            "rewritten_query": search_query,
        }
```

After removing, the `node_orchestrator` only handles `ticket` and `answer` (default).
The `set_awaiting_clarification(False)` and `set_fast_chunks([])` calls in the `ticket` and `answer`
branches can stay as cleanup.

### 2d. Update graph wiring

Find the graph definition section (around line 489) and replace it with:

```python
graph.add_node("query_analyzer",   node_query_analyzer)
graph.add_node("intent_analyzer",  node_intent_analyzer)   # NEW
graph.add_node("fast_retriever",   node_fast_retriever)
graph.add_node("orchestrator",     node_orchestrator)
graph.add_node("full_retriever",   node_full_retriever)
graph.add_node("synthesizer",      node_synthesizer)
graph.add_node("generator",        node_generator)
graph.add_node("ticket_creator",   node_ticket_creator)
graph.add_node("chat_fallback",    node_chat_fallback)

graph.set_entry_point("query_analyzer")

# query_analyzer → chat_fallback (off-topic) or intent_analyzer (EHC-related)
graph.add_conditional_edges(
    "query_analyzer",
    lambda s: "chat_fallback" if not s["is_ehc_related"] else "intent_analyzer"
)

# intent_analyzer → END (clarify) or fast_retriever (proceed)
graph.add_conditional_edges(
    "intent_analyzer",
    lambda s: END if s["intent"] == "clarify" else "fast_retriever"
)

# fast_retriever always goes to orchestrator
graph.add_edge("fast_retriever", "orchestrator")

# orchestrator → full_retriever (answer) or ticket_creator (ticket)
graph.add_conditional_edges(
    "orchestrator",
    lambda s: "ticket_creator" if s["intent"] == "create_ticket" else "full_retriever"
)

# full_retriever → synthesizer → generator or ticket_creator
graph.add_edge("full_retriever", "synthesizer")
graph.add_conditional_edges(
    "synthesizer",
    lambda s: "generator" if s["intent"] == "search_faq" else "ticket_creator"
)

# Terminal edges
graph.add_edge("generator",      END)
graph.add_edge("ticket_creator", END)
graph.add_edge("chat_fallback",  END)

app = graph.compile()
```

> **Note:** `node_tool_router` is no longer needed and can be removed from the file.

---

## Change 3 — Simplify `ORCHESTRATOR_PROMPT` in `core/orchestrator.py`

Remove rule 2 (`action="clarify"`) entirely. The prompt should only have 2 actions: `answer` and `ticket`.

Find and **delete** the entire rule 2 block. The remaining structure should be:

```
HƯỚNG DẪN QUYẾT ĐỊNH:

1. action = "answer" — ...  (keep existing rule 1 unchanged)

2. action = "ticket" — CHỈ khi đã có đủ thông tin về vấn đề nhưng không tìm được chunk phù hợp.
   KHÔNG tạo ticket nếu câu hỏi vẫn còn mơ hồ — lúc đó IntentAnalyzer đã xử lý rồi.
   ...  (keep existing ticket rule, update the description)
```

Also remove `clarify` from the JSON output format example in the prompt if it's listed there.

---

## Verify

```bash
# 1. Syntax check
rtk python3 -m py_compile core/query_rewriter.py
rtk python3 -m py_compile core/langgraph_agent.py

# 2. Standalone test
rtk python3 -c "
from core.query_rewriter import analyze_intent_pre_retrieval
tests = [
    'mình không in được',
    'ấn in vỏ bệnh án thì trắng xóa',
    'bị lỗi rồi',
    'quét thẻ BHYT báo không tìm thấy bệnh nhân',
]
for q in tests:
    r = analyze_intent_pre_retrieval(q)
    print(f'[{r[\"action\"]}] {q!r} → {r[\"rewritten_query\"]!r}')
    if r['clarify_message']:
        print(f'  clarify: {r[\"clarify_message\"]}')
"

# 3. Restart
sudo systemctl restart doantn ehc-worker

# 4. Check logs
sudo journalctl -u doantn -n 30
```

Expected log for vague query:
```
[AGENT] Node: IntentAnalyzer | query="mình không in được"
[INTENT_ANALYZER] action=clarify | rewritten="Lỗi không in được tài liệu"
[INTENT_ANALYZER] Clarify #1: "Bạn đang gặp lỗi ở bước nào? Thấy thông báo gì không?"
[AGENT] Done | tool=clarifier confidence=0.0000
```

Expected log for specific query:
```
[AGENT] Node: IntentAnalyzer | query="ấn in vỏ bệnh án thì trắng xóa"
[INTENT_ANALYZER] action=proceed | rewritten="Lỗi in vỏ bệnh án hiển thị trắng xóa"
[AGENT] Node: FastRetriever | query="Lỗi in vỏ bệnh án hiển thị trắng xóa"
```

---

## Git

```bash
cd ~/DOANTN
git checkout -b feature/intent-analyzer
git add core/query_rewriter.py core/langgraph_agent.py core/orchestrator.py
git commit -m "refactor: add pre-retrieval IntentAnalyzer node, move clarify decision before FastRetriever"
git push origin feature/intent-analyzer
```

## Notes
- Use `rtk` prefix for all shell and Python commands
- `analyze_intent_pre_retrieval` falls back to `action=proceed` on any LLM error — pipeline never stalls
- The `tool_router` node is no longer used and can be deleted from `langgraph_agent.py`
- `node_query_analyzer` bypass (`is_awaiting_clarification`) can be removed — `intent_analyzer` handles this now via history context
- Orchestrator `clarify_count` guard is now fully in `node_intent_analyzer`
