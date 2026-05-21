# FIX: float32 JSON crash — final root cause + fix

## Why the bug persists

Two float32 cast loops were added in `langgraph_agent.py` at lines 573–577 (duplicate — same loop twice). These fix `chunk.score` values. But the crash is **not** from chunk scores.

The crash comes from `trace_logger.finish_trace()` → `json.dumps(record)` which serializes ALL trace events, including data from the Generator node's `tl_event` call around line 399. That event likely passes raw `c.score` values (numpy.float32) as part of the event data, bypassing the `round()` calls.

**The simplest, most bulletproof fix: add a custom JSON encoder to `trace_logger.py` that converts numpy types automatically.**

---

## Fix 1 — trace_logger.py (REQUIRED, covers all json.dumps calls)

File: `~/DOANTN/core/trace_logger.py`

Find the `finish_trace` function and replace `json.dumps(record)` with a numpy-safe version:

```python
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

Then change every `json.dumps(record)` in `trace_logger.py` to:
```python
json.dumps(record, cls=_NumpySafeEncoder)
```

---

## Fix 2 — langgraph_agent.py (CLEANUP)

Remove the **duplicate** float cast loop. There are currently TWO identical loops at lines 573 and 577. Keep only ONE.

Before (broken — duplicate):
```python
chunks = result.get("chunks", [])
conf = result.get("confidence", 0.0)

for chunk in chunks:
    chunk.score = float(chunk.score)

for chunk in chunks:           # ← DUPLICATE, remove this
    chunk.score = float(chunk.score)
```

After (clean):
```python
chunks = result.get("chunks", [])
conf = result.get("confidence", 0.0)

# Cast float32 → float for JSON serialization
for chunk in chunks:
    chunk.score = float(chunk.score)
```

The `confidence=float(conf)` in the `Answer(...)` constructor is already correct — keep it.

---

## Steps

```bash
# 1. Edit trace_logger.py — add _NumpySafeEncoder class and update json.dumps calls
nano ~/DOANTN/core/trace_logger.py

# 2. Edit langgraph_agent.py — remove the duplicate float cast loop
nano ~/DOANTN/core/langgraph_agent.py

# 3. Restart worker
sudo systemctl restart ehc-worker

# 4. Send a Telegram test message

# 5. Check logs — should see Done, NOT float32 error
sudo journalctl -u ehc-worker -n 20
```

Expected in logs after fix:
```
[WORKER] Done | chat_id=... | conf=0.9784
```

Should NOT see:
```
Agent error: Object of type float32 is not JSON serializable
```

---

## Why trace_logger is the culprit

- `pipeline_worker.py` catches ALL exceptions from `core.langgraph_agent.run()` → sends "Hệ thống đang bận"
- The exception must be raised INSIDE `run()`
- Inside `run()`, `json.dumps` is called by:
  1. `tl_finish(trace_id, ...)` → `trace_logger.finish_trace()` → `json.dumps(record)` — **this is it**
  2. `end_trace(lf_trace, ...)` → `langfuse_tracer.end_trace()` → uses only `round()`, safe
- `finish_trace()` serializes ALL tl_event records including the Generator node's event at line ~399
- That Generator tl_event may pass chunk scores as raw float32 (not via `round()`)
- The numpy-safe encoder fixes this globally — no more float32 crash regardless of where it comes from

---

## Claude Code instruction (copy-paste)

```
In ~/DOANTN/core/trace_logger.py:
1. Add import: import numpy as np
2. Add class before finish_trace():
   class _NumpySafeEncoder(json.JSONEncoder):
       def default(self, obj):
           if isinstance(obj, np.floating):
               return float(obj)
           if isinstance(obj, np.integer):
               return int(obj)
           if isinstance(obj, np.ndarray):
               return obj.tolist()
           return super().default(obj)
3. Replace all json.dumps(record) with json.dumps(record, cls=_NumpySafeEncoder)

In ~/DOANTN/core/langgraph_agent.py:
4. Find the two identical loops "for chunk in chunks: chunk.score = float(chunk.score)" around lines 573-577
5. Remove the duplicate (keep only one)

Then: sudo systemctl restart ehc-worker
```
