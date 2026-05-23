# FIX: Qwen3-8B compatibility — thinking budget + None guard + max_tokens

_Generated: 2026-05-23_

## Problems

1. **`content=None` crash** — `response.choices[0].message.content.strip()` crashes when
   Qwen3-8B uses all allocated tokens for thinking and produces no answer text.
   Every `.content.strip()` call needs a None guard.

2. **`max_tokens` too small** — all LLM calls were sized for Qwen2.5-7B (no thinking overhead).
   With Qwen3-8B, thinking tokens consume the budget before the answer is produced.

3. **No `thinking_budget`** — model overthinks simple classification tasks.
   Add `extra_body={"chat_template_kwargs": {"enable_thinking": True, "thinking_budget": N}}`
   to cap thinking per task type.

4. **`awaiting_clarification` bypass in `node_query_analyzer`** — causes new messages
   in the same thread to skip IntentGuard. IntentAnalyzer handles this via session_history
   context — the bypass is no longer needed and should be removed.

---

## Change 1 — `core/query_rewriter.py`

### 1a. Add helper at top of file (after imports, before SYSTEM_PROMPT)

```python
def _safe_content(response) -> str | None:
    """Return message content or None if missing (Qwen3 thinking overflow)."""
    try:
        return response.choices[0].message.content
    except Exception:
        return None
```

### 1b. Add `THINKING_BUDGET` constant (after imports)

```python
# Thinking budget per task type — limits Qwen3 reasoning tokens
_BUDGET_FAST = 256    # rewrite, intent (simple classification)
_BUDGET_MEDIUM = 512  # analyze_and_rewrite, intent_analyzer (structured JSON)
```

### 1c. Update `analyze_intent()` — both call sites

Replace:
```python
            max_tokens=100,
            temperature=0.1,
        )

        intent = response.choices[0].message.content.strip()
```
With:
```python
            max_tokens=600,
            temperature=0.1,
            extra_body={"chat_template_kwargs": {"enable_thinking": True, "thinking_budget": _BUDGET_FAST}},
        )

        raw = _safe_content(response)
        if raw is None:
            return None
        intent = raw.strip()
```

Apply this change to BOTH call sites in `analyze_intent()` (primary call and APIConnectionError retry).

### 1d. Update `rewrite()` — both call sites

Replace:
```python
            max_tokens=150,
            temperature=0.1,
        )

        rewritten = response.choices[0].message.content.strip()
```
With:
```python
            max_tokens=700,
            temperature=0.1,
            extra_body={"chat_template_kwargs": {"enable_thinking": True, "thinking_budget": _BUDGET_FAST}},
        )

        raw = _safe_content(response)
        if raw is None:
            raise LLMUnavailableError("content=None (thinking overflow)")
        rewritten = raw.strip()
```

Apply to BOTH call sites (primary call and APIConnectionError retry).

### 1e. Update `analyze_and_rewrite()` — both call sites

Replace:
```python
            max_tokens=200,
            temperature=0.1,
        )

        raw = response.choices[0].message.content.strip()
```
With:
```python
            max_tokens=900,
            temperature=0.1,
            extra_body={"chat_template_kwargs": {"enable_thinking": True, "thinking_budget": _BUDGET_MEDIUM}},
        )

        raw = _safe_content(response)
        if raw is None:
            raise LLMUnavailableError("content=None (thinking overflow)")
        raw = raw.strip()
```

Apply to BOTH call sites.

### 1f. Update `analyze_intent_pre_retrieval()` inner `_call()`

Replace:
```python
    def _call():
        return _client.chat.completions.create(
            model=VLLM_MODEL,
            messages=messages,
            max_tokens=200,
            temperature=0.1,
        )
```
With:
```python
    def _call():
        return _client.chat.completions.create(
            model=VLLM_MODEL,
            messages=messages,
            max_tokens=900,
            temperature=0.1,
            extra_body={"chat_template_kwargs": {"enable_thinking": True, "thinking_budget": _BUDGET_MEDIUM}},
        )
```

And in the `try` block that calls `_call()`, add None guard:
```python
        resp = _call()
        raw = _safe_content(resp)
        if raw is None:
            raise ValueError("content=None (thinking overflow)")
        raw = raw.strip()
        data = _json.loads(raw)
```

Apply the same None guard pattern to the APIConnectionError retry block inside `analyze_intent_pre_retrieval()`.

---

## Change 2 — `core/orchestrator.py`

### 2a. Add helper at top of file (after imports, before ORCHESTRATOR_PROMPT)

```python
def _safe_content(response) -> str | None:
    """Return message content or None if missing (Qwen3 thinking overflow)."""
    try:
        return response.choices[0].message.content
    except Exception:
        return None

_BUDGET_ORCHESTRATOR = 512  # enough reasoning for routing decisions
```

### 2b. Fix messages format — use system role

In `orchestrate()`, replace:
```python
    messages = [{"role": "user", "content": prompt}]
```
With:
```python
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": query},
    ]
```

> Note: `prompt` already contains `{query}` interpolated, so the user turn is redundant context
> but keeps the conversation format correct for Qwen3's chat template.
> Alternatively, split ORCHESTRATOR_PROMPT into a static system part and a dynamic user part —
> but the simplest fix is to move the full formatted prompt to system role.
> Do NOT include the query twice if it's already in the system prompt.
> Use this instead:
> ```python
>     messages = [{"role": "system", "content": prompt}]
> ```

### 2c. Update `orchestrate()` LLM calls — both call sites

Replace:
```python
            max_tokens=300,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
```
With:
```python
            max_tokens=1000,
            temperature=0.1,
            extra_body={"chat_template_kwargs": {"enable_thinking": True, "thinking_budget": _BUDGET_ORCHESTRATOR}},
        )
        raw = _safe_content(response)
        if raw is None:
            print("[ORCHESTRATOR] content=None (thinking overflow) → fallback")
            return _fallback_result(query)
        raw = raw.strip()
```

Apply to BOTH call sites (primary call and APIConnectionError retry).

---

## Change 3 — `core/generator.py`

### 3a. Add helper + None guard

Add after imports:
```python
def _safe_content(response) -> str | None:
    try:
        return response.choices[0].message.content
    except Exception:
        return None

_BUDGET_GENERATOR = 1024  # generator needs more reasoning for quality answers
```

Update both call sites in `generate()`:
```python
            max_tokens=1500,
            temperature=0.3,
            extra_body={"chat_template_kwargs": {"enable_thinking": True, "thinking_budget": _BUDGET_GENERATOR}},
        )

        answer = _safe_content(response)
        if answer is None:
            raise LLMUnavailableError("content=None (thinking overflow)")
        answer = answer.strip()
```

---

## Change 4 — `core/langgraph_agent.py`

### 4a. Remove `awaiting_clarification` bypass in `node_query_analyzer`

Find and DELETE this entire block:
```python
    # Bypass classifier if mid-clarification — let Orchestrator handle via history
    if _session_mgr and _session_mgr.is_awaiting_clarification(session_id):
        print(f"[AGENT] Classifier: BYPASS (awaiting_clarification=True)")
        elapsed = (time.time() - t_start) * 1000
        tl_event(trace_id, "IntentGuard", "decision", {
            "result": True, "bypass": True, "duration_ms": round(elapsed, 1),
        }, duration_ms=round(elapsed, 1))
        return {
            "is_ehc_related": True,
            "intent": "search_faq",
        }
```

IntentAnalyzer receives `session_history` and handles mid-clarification context correctly.
The bypass is a leftover from the pre-IntentAnalyzer architecture.

---

## Verify

```bash
# 1. Syntax check
rtk python3 -m py_compile core/query_rewriter.py
rtk python3 -m py_compile core/orchestrator.py
rtk python3 -m py_compile core/generator.py
rtk python3 -m py_compile core/langgraph_agent.py

# 2. Quick smoke test — should return action=clarify with valid JSON
rtk python3 -c "
from core.query_rewriter import analyze_intent_pre_retrieval
r = analyze_intent_pre_retrieval('mình không in được')
print('action:', r['action'])
print('rewritten:', r['rewritten_query'])
print('clarify:', r['clarify_message'])
assert r['action'] == 'clarify', 'Expected clarify for vague query'
print('OK')
"

# 3. Restart
sudo systemctl restart doantn ehc-worker

# 4. Check logs for absence of NoneType errors
sudo journalctl -u doantn -n 40 | grep -E "NoneType|content=None|INTENT_ANALYZER|ORCHESTRATOR"
```

Expected: no `NoneType` errors, `[INTENT_ANALYZER] action=clarify` for vague queries.

---

## Git

```bash
cd ~/DOANTN
git checkout -b fix/qwen3-compat
git add core/query_rewriter.py core/orchestrator.py core/generator.py core/langgraph_agent.py
git commit -m "fix: Qwen3-8B compat — thinking_budget, None guard, max_tokens, remove awaiting bypass"
git push origin fix/qwen3-compat
```

## Notes
- Use `rtk` prefix for all shell and Python commands
- `_safe_content` can be moved to a shared `core/llm_utils.py` later — for now duplicate per file is fine
- `thinking_budget` values: 256 (fast tasks) / 512 (medium) / 1024 (generator) — tune if needed
- Do not change INTENT_ANALYZER_PROMPT or ORCHESTRATOR_PROMPT content
