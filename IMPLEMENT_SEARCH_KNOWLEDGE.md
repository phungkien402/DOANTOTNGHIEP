# TASK: Implement `search_knowledge` Tool

## Goal

Add a `search_knowledge` LangGraph tool that lets the Orchestrator load relevant
context files from `data/knowledge/` at runtime — without hardcoding any examples
or mappings in prompts.

Each file in `data/knowledge/` covers a broad operational topic (printing, network,
medications, discharge workflow, etc.). The LLM selects which file to read based on
the combination of (a) the user's question and (b) the fast_chunks already retrieved
— so it reasons about root cause, not just surface keywords.

**Design constraint:** no hardcoded keyword→file mappings anywhere.
All routing is done by LLM reasoning from live context.

---

## Architecture

```
User question
     ↓
FastRetriever  →  fast_chunks (quick semantic scan, top-5)
     ↓
Orchestrator node
  - sees: question + fast_chunks + list of available .md files in data/knowledge/
  - decides: does the fast_chunks context suggest a specific operational domain?
  - if yes: calls search_knowledge(topic="<filename_stem>")
  - if the retrieved chunks already fully cover the answer: skip search_knowledge
     ↓
search_knowledge tool  →  reads file, returns content string
     ↓
Orchestrator (second pass or parallel) → may also call rag_search
     ↓
AnswerGenerator  →  uses fast_chunks + knowledge_content + rag_results
```

### Why fast_chunks matter for tool selection

Example: query = "Không in được bảng kê"

**Case A** — fast_chunks contain: *"bệnh nhân chưa có y lệnh xử trí"*
→ Root cause is workflow (doctor hasn't approved orders yet), NOT print config.
→ Orchestrator should NOT call search_knowledge("printing") — that would mislead.
→ Orchestrator answers from the retrieved chunk directly.

**Case B** — fast_chunks contain: *"lỗi kết nối máy in", "cấu hình printer"*
→ Root cause is print configuration.
→ Orchestrator SHOULD call search_knowledge("printing") for supplementary guidance.

The Orchestrator prompt must instruct the LLM to read fast_chunks BEFORE deciding
which knowledge file (if any) to load.

---

## 1. Folder structure

```
data/
  knowledge/
    _index.json          ← auto-generated manifest (see Step 2)
    printing.md
    network.md
    medications.md
    discharge.md
    lab_orders.md
    (add more as needed)
```

Each `.md` file starts with a YAML-style frontmatter block:

```markdown
---
title: Hướng dẫn in ấn trong EHC
covers: in bảng kê, in toa thuốc, in phiếu xét nghiệm, cấu hình máy in, lỗi in
---

## Các bước kiểm tra khi không in được

...
```

The `covers` line is a comma-separated natural-language description used by the
Orchestrator to decide relevance. It is NOT a keyword list — just a short human
description of what the file is about.

---

## 2. `core/knowledge_store.py` — new file

```python
"""
Hot-reloadable knowledge file store.
Files in data/knowledge/ are read from disk at call time — no caching.
This means updates to .md files take effect immediately without restart.
"""

import json
import re
from pathlib import Path

KNOWLEDGE_DIR = Path(__file__).parent.parent / "data" / "knowledge"
INDEX_FILE = KNOWLEDGE_DIR / "_index.json"


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Extract YAML-style frontmatter from a markdown file.
    Returns (metadata_dict, body_text).
    If no frontmatter found, returns ({}, text).
    """
    match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
    if not match:
        return {}, text
    fm_raw, body = match.group(1), match.group(2)
    meta = {}
    for line in fm_raw.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    return meta, body.strip()


def rebuild_index() -> None:
    """Scan data/knowledge/ and regenerate _index.json.
    Called automatically when _index.json is missing or stale.
    Safe to call at startup or after adding new files.
    """
    entries = []
    for md_file in sorted(KNOWLEDGE_DIR.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        meta, _ = _parse_frontmatter(text)
        entries.append({
            "stem": md_file.stem,
            "filename": md_file.name,
            "title": meta.get("title", md_file.stem),
            "covers": meta.get("covers", ""),
        })
    INDEX_FILE.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def list_topics() -> list[dict]:
    """Return the index entries: [{stem, filename, title, covers}, ...].
    Rebuilds _index.json if missing.
    """
    if not INDEX_FILE.exists():
        rebuild_index()
    return json.loads(INDEX_FILE.read_text(encoding="utf-8"))


def load_topic(stem: str) -> str:
    """Read and return the body of data/knowledge/<stem>.md.
    Returns an empty string if the file does not exist.
    """
    path = KNOWLEDGE_DIR / f"{stem}.md"
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    _, body = _parse_frontmatter(text)
    return body
```

---

## 3. `core/tools.py` — add `search_knowledge` tool

Add alongside the existing tools (rag_search, etc.):

```python
from core.knowledge_store import list_topics, load_topic

@tool
def search_knowledge(topic: str) -> str:
    """Load a knowledge guidance file for a specific operational topic.

    Before calling this tool, review the fast_chunks already retrieved.
    If the chunks already explain the root cause, do NOT call this tool.
    Only call this tool when supplementary operational guidance would help
    the user (e.g., step-by-step config, troubleshooting checklist).

    Args:
        topic: The stem name of the knowledge file to load (e.g. "printing",
               "network", "medications"). Use list_knowledge_topics() first
               to see what is available.
    """
    content = load_topic(topic)
    if not content:
        return f"[Knowledge file '{topic}' not found. Available topics: {[t['stem'] for t in list_topics()]}]"
    return content


@tool
def list_knowledge_topics() -> str:
    """List all available knowledge topics with their titles and coverage descriptions.
    Call this BEFORE search_knowledge if you are unsure which topic to load.
    Returns a formatted list of available files.
    """
    topics = list_topics()
    if not topics:
        return "[No knowledge files found in data/knowledge/]"
    lines = []
    for t in topics:
        lines.append(f"- {t['stem']}: {t['title']} — bao gồm: {t['covers']}")
    return "\n".join(lines)
```

Register both tools in the tools list passed to the LangGraph agent.

---

## 4. Orchestrator prompt update

Find the Orchestrator system prompt in `core/langgraph_agent.py` (or wherever it
is defined). Add this section AFTER the existing tool descriptions and BEFORE the
final instruction:

```
Công cụ hỗ trợ ngữ cảnh nghiệp vụ:
- list_knowledge_topics(): liệt kê các file hướng dẫn nghiệp vụ đang có
- search_knowledge(topic): đọc nội dung file hướng dẫn cho một chủ đề cụ thể

Khi nào dùng search_knowledge:
1. Trước tiên, đọc fast_chunks đã truy xuất được.
2. Nếu fast_chunks đã giải thích được nguyên nhân gốc (root cause) của vấn đề
   → KHÔNG cần gọi search_knowledge, trả lời trực tiếp từ chunks.
3. Chỉ gọi search_knowledge khi fast_chunks cho thấy vấn đề thuộc một lĩnh vực
   nghiệp vụ cụ thể (in ấn, mạng, thuốc, xuất viện...) VÀ người dùng cần
   hướng dẫn thao tác chi tiết hơn những gì chunks cung cấp.
4. Nếu chưa chắc chắn topic nào phù hợp → gọi list_knowledge_topics() trước.

Ví dụ lý luận (KHÔNG hardcode vào prompt — đây chỉ là minh hoạ):
  Câu hỏi: "Không in được bảng kê"
  fast_chunks trả về: "bệnh nhân chưa có y lệnh được duyệt"
  → Nguyên nhân là quy trình nghiệp vụ, KHÔNG phải cấu hình in
  → Trả lời từ chunks, không gọi search_knowledge("printing")
```

**Do not include the example block in the actual prompt.** The "Ví dụ lý luận"
section is for your understanding only — it explains the intended reasoning pattern.
Include only the numbered rules (1–4) in the real prompt. The LLM generalizes from
the rules, not from hardcoded examples.

---

## 5. Seed knowledge files

Create these starter files. They will be extended by the helpdesk team over time.

### `data/knowledge/printing.md`

```markdown
---
title: Hướng dẫn xử lý sự cố in ấn
covers: in bảng kê, in toa thuốc, in phiếu xét nghiệm, máy in, cổng in, driver, không in được
---

## Kiểm tra nhanh khi không in được

1. Xác nhận bệnh nhân đã hoàn thành quy trình trước khi in:
   - Nội trú: y lệnh điều trị đã được bác sĩ duyệt chưa?
   - Ngoại trú: đơn thuốc đã được xác nhận chưa?
   (Nếu chưa → lỗi do quy trình, không phải do máy in)

2. Kiểm tra kết nối máy in:
   - Vào System > Cấu hình máy in
   - Xác nhận tên máy in đúng với máy đang kết nối trên mạng nội bộ

3. Kiểm tra cổng in (print port):
   - Thử in thử (test page) từ Windows
   - Nếu Windows in được nhưng EHC không in → kiểm tra cấu hình máy in trong EHC

4. Driver máy in:
   - Đảm bảo driver đúng phiên bản với máy in
   - Thử xóa và cài lại driver nếu lỗi xuất hiện sau khi cập nhật Windows

5. Log lỗi:
   - Vào EHC > Nhật ký hệ thống để xem thông báo lỗi in chi tiết
```

### `data/knowledge/network.md`

```markdown
---
title: Hướng dẫn xử lý sự cố mạng và kết nối
covers: mất mạng, không vào được hệ thống, kết nối chậm, VPN, không truy cập, lỗi kết nối
---

## Sự cố mạng phổ biến trong bệnh viện

Lưu ý: EHC là phần mềm chạy trên mạng nội bộ (LAN) bệnh viện.
Các vấn đề mạng cần liên hệ bộ phận IT bệnh viện — không phải bộ phận hỗ trợ EHC.

### Không vào được phần mềm EHC

1. Kiểm tra cáp mạng hoặc WiFi của máy tính
2. Thử mở trình duyệt và truy cập một trang nội bộ khác
3. Nếu mạng hoạt động bình thường nhưng EHC không vào được:
   - Thông báo cho IT kiểm tra server EHC
   - Hoặc liên hệ hotline hỗ trợ EHC

### Đăng nhập chậm / loading lâu

- Có thể do tải mạng cao trong giờ cao điểm
- Hoặc do server đang được cập nhật — kiểm tra thông báo bảo trì

### VPN / truy cập từ xa

EHC không hỗ trợ truy cập từ ngoài bệnh viện trực tiếp.
Để truy cập từ xa cần VPN do IT bệnh viện cung cấp — liên hệ bộ phận IT.
```

### `data/knowledge/medications.md`

```markdown
---
title: Hướng dẫn nghiệp vụ thuốc và kê đơn
covers: kê đơn, toa thuốc, cấp phát thuốc, y lệnh thuốc, dược, lĩnh thuốc, in toa
---

## Quy trình kê đơn ngoại trú

1. Bác sĩ vào phân hệ Khám bệnh > Kê đơn thuốc
2. Chọn thuốc từ danh mục (chỉ thuốc có trong danh mục mới kê được)
3. Nhập liều dùng, số ngày
4. Bấm Xác nhận đơn thuốc
5. Bệnh nhân mang phiếu đến nhà thuốc để lĩnh thuốc

## Y lệnh thuốc nội trú

1. Bác sĩ tạo y lệnh thuốc trong Điều trị nội trú > Y lệnh
2. Điều dưỡng duyệt và thực hiện y lệnh
3. Dược sĩ cấp phát theo y lệnh đã duyệt

## Thuốc không có trong danh mục

- Không thể kê thuốc ngoài danh mục EHC trực tiếp
- Cần trình dược sĩ hoặc trưởng khoa dược để bổ sung danh mục
- Trường hợp khẩn: lập phiếu lĩnh thuốc thủ công

## In toa thuốc

- Toa chỉ in được sau khi bác sĩ đã Xác nhận đơn
- Nếu nút In bị mờ (disabled): kiểm tra trạng thái đơn thuốc — có thể chưa xác nhận
```

---

## 6. `core/knowledge_store.py` — rebuild index at startup

In the module where tools are initialized (likely `core/langgraph_agent.py` or
`core/tools.py`), add a one-time call:

```python
from core.knowledge_store import rebuild_index
rebuild_index()   # regenerates _index.json from current .md files
```

This ensures `_index.json` is always fresh when the service starts.

---

## 7. Verify

```bash
rtk python3 -c "
from core.knowledge_store import list_topics, load_topic
print('=== Available topics ===')
for t in list_topics():
    print(f'  {t[\"stem\"]}: {t[\"title\"]}')
print()
print('=== printing.md body (first 200 chars) ===')
print(load_topic('printing')[:200])
print()
print('=== nonexistent topic ===')
print(load_topic('doesnotexist'))
"
```

Expected output:
```
=== Available topics ===
  discharge: ...
  lab_orders: ...
  medications: ...
  network: ...
  printing: ...

=== printing.md body (first 200 chars) ===
## Kiểm tra nhanh khi không in được
...

=== nonexistent topic ===

```

Then do a quick agent smoke test:

```bash
rtk python3 -c "
import asyncio
from core.langgraph_agent import run_agent

async def test():
    result = await run_agent('u_test', 'Không in được bảng kê, bảo lỗi máy in')
    print(result)

asyncio.run(test())
"
```

The agent should call `search_knowledge('printing')` and return troubleshooting steps.

---

## 8. Git workflow

```bash
/bin/bash -c "export PATH=/home/phungkien/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin && cd ~/DOANTN && git checkout -b feature/search-knowledge-tool"
```

Files to add:
```
core/knowledge_store.py        ← new file
core/tools.py                  ← add search_knowledge + list_knowledge_topics
core/langgraph_agent.py        ← update Orchestrator prompt + register tools + rebuild_index()
data/knowledge/printing.md     ← new file
data/knowledge/network.md      ← new file
data/knowledge/medications.md  ← new file
```

```bash
/bin/bash -c "export PATH=/home/phungkien/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin && cd ~/DOANTN && git add core/knowledge_store.py core/tools.py core/langgraph_agent.py data/knowledge/ && git commit -m 'feat: add search_knowledge tool with hot-reloadable MD knowledge base' && git push origin feature/search-knowledge-tool"
```

---

## Done Criteria

- [ ] `data/knowledge/` folder exists with at least 3 seed `.md` files
- [ ] `_index.json` is generated at startup and lists all `.md` files
- [ ] `list_knowledge_topics()` returns correct topic list
- [ ] `search_knowledge("printing")` returns file body (not frontmatter)
- [ ] `search_knowledge("nonexistent")` returns a graceful error string
- [ ] Orchestrator prompt includes the 4-rule guidance section (no hardcoded examples)
- [ ] Agent correctly handles "Không in được bảng kê":
  - When fast_chunks suggest workflow issue → answers from chunks, does NOT load printing.md
  - When fast_chunks suggest print config → loads printing.md and uses content
- [ ] Smoke test passes without exceptions
- [ ] `rebuild_index()` called at service startup
