# TASK: System-wide improvements — prompts, routing, Slack worker, session

_Generated: 2026-05-23_

Apply all fixes below in order. Each fix is independent, but the order makes testing easier.

---

## Fix 1 — Float32 crash in ticket path (`core/tools/create_ticket.py`)

### Root cause
`round(numpy.float32(x), n)` returns `numpy.float32`, not a Python `float`.
When `json.dumps` encounters this value → `TypeError: Object of type float32 is not JSON serializable`.

### Fix

File: `~/DOANTN/core/tools/create_ticket.py`

Find (around line 82):
```python
"confidence": round(confidence, 4)
```

Replace with:
```python
"confidence": float(round(confidence, 4))
```

### Verify
```bash
grep -n '"confidence"' ~/DOANTN/core/tools/create_ticket.py
```
Must show `float(round(confidence, 4))`.

---

## Fix 2 — Intent Guard: block destructive commands (`core/intent_guard.py`)

### Root cause
Query `"xoá database server đi"` is classified as YES (EHC-related) because:
1. The NO definition in the prompt is too narrow (only greeting/small talk)
2. The bias `"When uncertain → YES"` pushes the LLM toward YES

### Fix A — Add regex pre-filter before LLM call

File: `~/DOANTN/core/intent_guard.py`

Add after imports (before `CLASSIFY_PROMPT`):

```python
import re

_DANGEROUS_PATTERNS = re.compile(
    r'\b(xoá|xóa|xoa|delete|drop|rm\s*-rf|format|wipe|destroy|truncate)\b'
    r'.{0,40}'
    r'\b(database|server|data|db|disk|system|table|ổ\s*cứng|máy\s*chủ)\b',
    re.IGNORECASE | re.UNICODE,
)
```

Add at the top of `classify()`, before the LLM call:

```python
def classify(query: str) -> bool:
    # Pre-filter: block destructive/infrastructure commands without calling LLM
    if _DANGEROUS_PATTERNS.search(query):
        print(f"[INTENT_GUARD] Blocked destructive pattern → OFF-TOPIC: \"{query}\"")
        return True  # off-topic

    # ... rest of existing code unchanged
```

### Fix B — Update CLASSIFY_PROMPT: expand NO definition

In `CLASSIFY_PROMPT`, find the section describing off-topic queries (NO).
Add these two new categories to the NO list:

```
Off-topic queries (NO):
- Pure greetings / small talk (hello, thank you, goodbye, ...)
- Completely unrelated to the hospital or EHC software (weather, cooking, entertainment, ...)
- Destructive IT commands: delete database, drop table, format disk, rm -rf, destroy server
- Infrastructure administration not related to EHC software operations: OS-level server restart,
  firewall changes, OS installation
```

### Verify
```bash
rtk python3 -c "
from core.intent_guard import classify
tests = [
    ('xoá database server đi', True),
    ('drop table patients', True),
    ('rm -rf /var/data', True),
    ('bấm lưu không được', False),
    ('màn hình quay không dừng', False),
]
for q, expected_offtopic in tests:
    result = classify(q)
    status = '✅' if result == expected_offtopic else '❌'
    print(f'{status} [OFF={result}] {q}')
"
```

---

## Fix 3 — Orchestrator: require clarify before ticket (`core/orchestrator.py`)

### Root cause
`ORCHESTRATOR_PROMPT` rule for `action="ticket"` currently says "ticket when chunks are irrelevant" —
no requirement to clarify first. LLM picks ticket immediately on first failure.

### Fix — Update `ORCHESTRATOR_PROMPT`

File: `~/DOANTN/core/orchestrator.py`

Find rule 3 (or the section describing when to use `action="ticket"`). Replace/augment with:

```
3. action="ticket":
   - ONLY use after at least 1 clarify attempt has been made and the user still could not provide enough info
   - ONLY use when the query is clearly a specific technical issue not found in the knowledge base
   - Do NOT use on the first turn if the query is still ambiguous — clarify first
   - IF all chunk scores < 0.35 AND no clarify has been done yet → prefer action="clarify"
   - IF all chunk scores < 0.35 AND at least 1 clarify has been done → action="ticket"
```

Find the rule for `action="clarify"` and append:
```
   → IF the query contains " — " in the middle (e.g., "original question — user's follow-up"),
     this is an enriched query after clarification. NEVER use action="clarify". Only "answer" or "ticket".
```

### Verify
Log must show correct behavior:
```
Turn 1 (ambiguous query, low score):
  [ORCHESTRATOR] action=clarify
  [BOT] → asks user for more info

Turn 2 (enriched query):
  [ORCHESTRATOR] action=answer or ticket
  [BOT] → does NOT clarify again
```

---

## Fix 4 — Generator: more natural output (`core/generator.py`)

### Root cause
Generator sometimes copies raw FAQ text verbatim (including `→` arrows and awkward formatting)
instead of rephrasing it into natural language.

### Fix — Update `SYSTEM_PROMPT`

File: `~/DOANTN/core/generator.py`

**4a. Add rule about `→` arrow notation**

Find the section describing output format rules. Add:

```
- If the source document uses → (arrow) to list steps, REWRITE them as complete sentences
  or use numbered format (1), (2), (3)... Do NOT let → appear in the response.
```

**4b. Add rule about numbered notation in FAQ content**

```
- When the source uses (1)(2)(3)(4) or 1. 2. 3. to enumerate steps,
  keep the numbering but write each step as a full, readable sentence.
```

**4c. Remove the "ask the user back" rule** (rule 9 or equivalent)

Find and delete any rule with content like:
```
- If you don't have enough information, ask the user to clarify
```
The Generator must NOT ask follow-up questions — that is the Orchestrator's responsibility.

**4d. Make the "admit limitations" rule optional**

Find: "if information is not found, admit it directly".
Replace with:

```
- If the provided documents are not sufficient for a complete answer, answer what you can
  and add a brief note at the end only if necessary. Do not say "I don't have enough information"
  if you have already answered most of the question.
```

**4e. Check temperature and max_tokens**

```bash
grep -n "temperature\|max_tokens" ~/DOANTN/core/generator.py
```

Ensure:
- `temperature`: `0.25` (not `0.1` — needs slight creativity for natural phrasing)
- `max_tokens`: `400` (enough for detailed answers)

Update if not already set.

### Verify
Send a question whose FAQ answer contains `→`. The bot must reply in complete sentences, no `→` in output.

---

## Fix 5 — Chunk filter before Generator (`core/langgraph_agent.py`)

### Root cause
`node_full_retriever` currently passes ALL chunks after reranking (including chunks with score 0.06, 0.09)
to the Generator. Low-score chunks introduce noise — Generator may blend information from unrelated FAQs.

### Fix — Add filter in `node_full_retriever`

File: `~/DOANTN/core/langgraph_agent.py`

In `node_full_retriever`, find where `ranked_chunks` is built and the function is about to return.
Add the filter just before `return`:

```python
# Filter: only pass chunks that meet the confidence threshold to the Generator
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.4"))
filtered_chunks = [c for c in ranked_chunks if c.score >= CONFIDENCE_THRESHOLD]

# Fallback: if no chunk meets the threshold, keep the best one (prevent empty generator input)
if not filtered_chunks and ranked_chunks:
    filtered_chunks = ranked_chunks[:1]
    print(f"[FULL_RETRIEVER] No chunk >= {CONFIDENCE_THRESHOLD}, keeping top-1 (score={ranked_chunks[0].score:.4f})")
else:
    print(f"[FULL_RETRIEVER] Filtered {len(ranked_chunks)} → {len(filtered_chunks)} chunks (threshold={CONFIDENCE_THRESHOLD})")

return {"chunks": filtered_chunks, "confidence": top_score, "knowledge_content": knowledge_content}
```

### Verify
Log must show:
```
[FULL_RETRIEVER] Filtered 3 → 1 chunks (threshold=0.4)
```
Generator must no longer receive chunks with score 0.06 or 0.09.

---

## Fix 6 — FastRetriever: use rewritten query (`core/langgraph_agent.py`)

### Root cause
`node_fast_retriever` currently uses `state["query"]` (the raw user query).
`node_query_analyzer` already rewrites the query into `state["rewritten_query"]` for better search —
but FastRetriever ignores it.

### Fix

File: `~/DOANTN/core/langgraph_agent.py`

In `node_fast_retriever`, find the line that fetches the query for search:

```python
query = state["query"]
```

Replace with:

```python
query = state.get("rewritten_query") or state["query"]
```

If the rewriter fails (LLMUnavailableError), `rewritten_query` will be `None` or `""` →
falls back to `state["query"]` automatically. Zero regression risk.

### Verify
```bash
grep -n "rewritten_query\|node_fast_retriever" ~/DOANTN/core/langgraph_agent.py | head -20
```
After the fix, FastRetriever log must show the same rewritten query as the QueryAnalyzer log.

---

## Fix 7 — Session clarify_count (`api/session.py`)

### Context
The clarify-count guard in `node_orchestrator` (commit e09f3eb) calls
`_session_mgr.get_clarify_count()` and `_session_mgr.increment_clarify_count()`,
but `SessionManager` does not have these methods yet → `AttributeError` at runtime.

### Fix

File: `~/DOANTN/api/session.py`

**7a. Add storage dict in `__init__`**

```python
def __init__(self, ttl: int = SESSION_TTL):
    # ... existing fields ...
    self._clarify_count: dict[str, int] = {}   # add this line
```

**7b. Add 2 methods after `set_fast_chunks`**

```python
def get_clarify_count(self, session_id: str) -> int:
    self._check_ttl(session_id)
    return self._clarify_count.get(session_id, 0)

def increment_clarify_count(self, session_id: str) -> None:
    self._clarify_count[session_id] = self._clarify_count.get(session_id, 0) + 1
```

**7c. Reset count when clearing awaiting state**

In `set_awaiting_clarification`:

```python
def set_awaiting_clarification(self, session_id: str, value: bool) -> None:
    self._awaiting_clarification[session_id] = value
    if not value:
        self._clarify_count.pop(session_id, None)   # reset when no longer awaiting
```

**7d. Clean up in `clear()`**

```python
def clear(self, session_id: str) -> None:
    # ... existing pops ...
    self._clarify_count.pop(session_id, None)   # add this line
```

### Verify
```bash
grep -n "clarify_count\|get_clarify_count\|increment_clarify" ~/DOANTN/api/session.py
```
Must show all 4 changes above.

---

## Fix 8 — Slack worker + `_RedisSessionMgr` clarify_count (`workers/pipeline_worker.py`)

### Context
`_RedisSessionMgr` is the shim used by async workers (Telegram, Slack).
Needs `clarify_count` methods mirroring `api/session.py`.
Also needs `process_slack_query` and `_send_slack`.

### Fix A — Add clarify_count to `_RedisSessionMgr`

File: `~/DOANTN/workers/pipeline_worker.py`

Inside class `_RedisSessionMgr`, add the following methods:

```python
def set_awaiting_clarification(self, session_id: str, value: bool) -> None:
    self._data["awaiting_clarification"] = value
    if not value:
        self._data["clarify_count"] = 0  # reset when no longer awaiting

def get_clarify_count(self, session_id: str) -> int:
    return self._data.get("clarify_count", 0)

def increment_clarify_count(self, session_id: str) -> None:
    self._data["clarify_count"] = self._data.get("clarify_count", 0) + 1

def reset_clarify_count(self, session_id: str) -> None:
    self._data["clarify_count"] = 0
```

### Fix B — Add `_send_slack` helper

Add at module level (after `_send_telegram`):

```python
def _send_slack(channel_id: str, text: str, thread_ts: str = None):
    """Send a message to a Slack channel, optionally in a thread."""
    token = os.getenv("SLACK_BOT_TOKEN", "")
    if not token:
        print("[WORKER-SLACK] SLACK_BOT_TOKEN not set, skipping send")
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
    except Exception as e:
        print(f"[WORKER-SLACK] Send error: {e}")
```

### Fix C — Add `process_slack_query` function

Add after `process_telegram_query` (mirrors the same pattern):

```python
def process_slack_query(
    session_id: str,
    channel_id: str,
    text: str,
    thread_ts: str,
    history: list,
):
    """RQ worker: process a Slack message and reply in thread."""
    from core.models import Message
    from core.langgraph_agent import run, set_session_manager

    # Load session state from Redis
    session_data = _load_tg_session(session_id)
    awaiting = session_data.get("awaiting_clarification", False)
    original_query = session_data.get("original_query", "")

    # Build shim and inject into agent
    shim = _RedisSessionMgr(session_id, session_data)
    set_session_manager(shim)

    # Enrich query if we are in clarification mode
    effective_text = text
    if awaiting and original_query:
        effective_text = f"{original_query} — {text}"

    msg = Message(
        user_id=session_id,
        session_id=session_id,
        text=effective_text,
        timestamp=time.time(),
        platform="slack",
    )

    answer_text = None
    confidence = 0.0
    try:
        answer = run(msg, session_data.get("history", []))
        answer_text = answer.text
        confidence = answer.confidence
    except Exception as e:
        print(f"[WORKER-SLACK] Agent error: {e}")
        import traceback; traceback.print_exc()

    # Build reply with confidence indicator
    if answer_text:
        reply_text = answer_text
        if confidence >= 0.4:
            reply_text += f"\n\n📊 Độ tin cậy: {confidence * 100:.0f}%"
        else:
            reply_text += "\n\n🔴 Độ tin cậy: thấp"
    else:
        reply_text = "⚠️ Hệ thống đang bận, vui lòng thử lại sau."

    _send_slack(channel_id, reply_text, thread_ts)

    # Persist session state back to Redis
    if answer_text:
        session_data.setdefault("history", [])
        session_data["history"].append({"role": "user", "text": text})
        session_data["history"].append({"role": "bot", "text": answer_text})

        new_awaiting = session_data.get("awaiting_clarification", False)
        if new_awaiting and not awaiting:
            session_data["original_query"] = text
        elif not new_awaiting:
            session_data["original_query"] = ""

    _save_tg_session(session_id, session_data)
```

### Verify
```bash
grep -n "process_slack_query\|_send_slack\|clarify_count" ~/DOANTN/workers/pipeline_worker.py
```

---

## Execution order & restart

After applying all fixes:

```bash
# 1. Check syntax
rtk python3 -m py_compile core/tools/create_ticket.py
rtk python3 -m py_compile core/intent_guard.py
rtk python3 -m py_compile core/orchestrator.py
rtk python3 -m py_compile core/generator.py
rtk python3 -m py_compile core/langgraph_agent.py
rtk python3 -m py_compile api/session.py
rtk python3 -m py_compile workers/pipeline_worker.py

# 2. Restart services
sudo systemctl restart ehc-api ehc-worker

# 3. Check logs are clean
sudo journalctl -u ehc-api -n 30
sudo journalctl -u ehc-worker -n 30
```

---

## Smoke tests

### Test 1 — Float32 (Fix 1)
Send a question not in the FAQ 3 times via Slack.
Expected: turn 1 → clarify, turn 2 → clarify (2nd time), turn 3 → ticket. No float32 crash.

### Test 2 — Intent guard (Fix 2)
Send via Slack: `"xoá database server đi"`
Expected: bot replies off-topic. No EHC pipeline triggered, no ticket created.

### Test 3 — Clarify flow (Fix 3 + 7 + 8)
Turn 1: ambiguous question (e.g., "hệ thống bị lỗi") → bot asks for more info (clarify #1)
Turn 2: vague follow-up → bot may clarify again (clarify #2, if `clarify_count` limit allows)
Turn 3: still not enough info → bot creates ticket (no third clarify)

### Test 4 — Generator output (Fix 4)
Send a question whose FAQ contains `→`. Bot must reply in complete sentences, no `→` in output.

### Test 5 — Chunk filter (Fix 5)
Check server log:
```
[FULL_RETRIEVER] Filtered 3 → 1 chunks (threshold=0.4)
```
Generator must no longer receive chunks with score below 0.4 (except the fallback top-1).

---

## Git

```bash
cd ~/DOANTN
git checkout -b feature/system-improvements
git add core/tools/create_ticket.py core/intent_guard.py core/orchestrator.py \
        core/generator.py core/langgraph_agent.py api/session.py workers/pipeline_worker.py
git commit -m "fix: float32, intent guard, orchestrator prompt, generator output, chunk filter, Slack worker"
git push origin feature/system-improvements
```

## Notes
- Use `rtk` prefix for all shell commands
- Do not modify any file outside the list above
- Fix 6 (FastRetriever rewritten_query) is zero-risk — falls back to original query if rewrite fails
- Fix 5 (chunk filter) — if bot stops answering, try lowering CONFIDENCE_THRESHOLD to 0.35 in `.env`
