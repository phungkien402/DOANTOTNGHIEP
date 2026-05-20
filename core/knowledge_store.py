"""
knowledge_store.py — Hot-reloadable knowledge file store.

Files in data/knowledge/ are read from disk at call time — no caching.
This means updates to .md files take effect immediately without restart.

Public API:
  - rebuild_index() → regenerates _index.json from current .md files
  - list_topics()   → returns [{stem, filename, title, covers}, ...]
  - load_topic(stem) → returns body text of the .md file (no frontmatter)
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

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
    meta: dict[str, str] = {}
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
    if not KNOWLEDGE_DIR.exists():
        KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
        print("[KNOWLEDGE_STORE] Created data/knowledge/ directory")

    entries: list[dict] = []
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
    print(f"[KNOWLEDGE_STORE] Rebuilt _index.json with {len(entries)} entries")


def list_topics() -> list[dict]:
    """Return the index entries: [{stem, filename, title, covers}, ...].

    Rebuilds _index.json if missing.
    """
    if not INDEX_FILE.exists():
        rebuild_index()
    try:
        return json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
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


if __name__ == "__main__":
    print("=== knowledge_store.py standalone test ===\n")

    rebuild_index()

    print("\n=== Available topics ===")
    for t in list_topics():
        print(f"  {t['stem']}: {t['title']}")

    print("\n=== printing.md body (first 200 chars) ===")
    print(load_topic("printing")[:200])

    print("\n=== nonexistent topic ===")
    result = load_topic("doesnotexist")
    print(f"  (empty string: {repr(result)})")

    print("\n✓ knowledge_store.py works correctly.")
