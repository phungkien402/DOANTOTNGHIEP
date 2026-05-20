"""
search_knowledge.py — Knowledge base tool for operational guidance.

Loads relevant knowledge files from data/knowledge/ at runtime.
The Orchestrator decides which file to load based on the user's question
and the fast_chunks already retrieved — no hardcoded keyword→file mappings.

Public API:
  - search_knowledge(topic) → returns file body content
  - list_knowledge_topics() → returns formatted list of available topics
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.knowledge_store import list_topics, load_topic


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

    Returns:
        The body content of the knowledge file, or an error message if not found.
    """
    content = load_topic(topic)
    if not content:
        available = [t["stem"] for t in list_topics()]
        return f"[Knowledge file '{topic}' not found. Available topics: {available}]"

    print(f"[SEARCH_KNOWLEDGE] Loaded topic '{topic}' ({len(content)} chars)")
    return content


def list_knowledge_topics() -> str:
    """List all available knowledge topics with their titles and coverage descriptions.

    Call this BEFORE search_knowledge if you are unsure which topic to load.

    Returns:
        A formatted list of available files with their descriptions.
    """
    topics = list_topics()
    if not topics:
        return "[No knowledge files found in data/knowledge/]"
    lines = []
    for t in topics:
        lines.append(f"- {t['stem']}: {t['title']} — bao gồm: {t['covers']}")
    result = "\n".join(lines)
    print(f"[SEARCH_KNOWLEDGE] Listed {len(topics)} topics")
    return result


if __name__ == "__main__":
    print("=== search_knowledge.py standalone test ===\n")

    print("--- list_knowledge_topics() ---")
    print(list_knowledge_topics())

    print("\n--- search_knowledge('printing') ---")
    result = search_knowledge("printing")
    print(result[:300])

    print("\n--- search_knowledge('nonexistent') ---")
    print(search_knowledge("nonexistent"))

    print("\n✓ search_knowledge.py works correctly.")
