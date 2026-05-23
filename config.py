"""
Configuration loader.
Reads all settings from .env file and exports them as module-level constants.
Validates on import — raises RuntimeError if any required variable is missing.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path)


def _require(key: str) -> str:
    """Get an environment variable or raise with a clear message."""
    value = os.getenv(key)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {key}\n"
            f"Copy .env.example to .env and fill in all values."
        )
    return value


def _get(key: str, default: str = "") -> str:
    """Get an environment variable with a default."""
    return os.getenv(key, default)


# --- Redmine ---
REDMINE_URL = _require("REDMINE_URL")
REDMINE_API_KEY = _require("REDMINE_API_KEY")
REDMINE_PROJECT = _get("REDMINE_PROJECT", "ehcfaq")

# --- vLLM ---
VLLM_BASE_URL = _get("VLLM_BASE_URL", "http://localhost:8000")
VLLM_MODEL = _get("VLLM_MODEL", "Qwen/Qwen3-8B")

# --- Embedding & Reranker ---
EMBED_MODEL = _get("EMBED_MODEL", "BAAI/bge-m3")
RERANKER_MODEL = _get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# --- Qdrant ---
QDRANT_URL = _get("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = _get("QDRANT_COLLECTION", "doantn_faq")

# --- RAG settings ---
RETRIEVER_TOP_K = int(_get("RETRIEVER_TOP_K", "10"))
RERANKER_TOP_N = int(_get("RERANKER_TOP_N", "3"))
CONFIDENCE_THRESHOLD = float(_get("CONFIDENCE_THRESHOLD", "0.4"))

# --- Clarification heuristic ---
CLARIFY_SPREAD_THRESHOLD = float(_get("CLARIFY_SPREAD_THRESHOLD", "0.05"))

# --- Session ---
SESSION_MAX_TURNS = int(_get("SESSION_MAX_TURNS", "10"))

# --- Maintenance ---
MAINTENANCE_MODE = _get("MAINTENANCE_MODE", "false").lower() == "true"
ADMIN_TOKEN = _get("ADMIN_TOKEN", "")

# --- Telegram ---
TELEGRAM_BOT_TOKEN = _get("TELEGRAM_BOT_TOKEN", "")

# --- Zalo OA ---
ZALO_OA_SECRET = _get("ZALO_OA_SECRET", "")
ZALO_ACCESS_TOKEN = _get("ZALO_ACCESS_TOKEN", "")

# --- Slack ---
SLACK_BOT_TOKEN = _get("SLACK_BOT_TOKEN", "")
SLACK_SIGNING_SECRET = _get("SLACK_SIGNING_SECRET", "")
SLACK_ADMIN_USERS = [u.strip() for u in _get("SLACK_ADMIN_USERS", "").split(",") if u.strip()]


if __name__ == "__main__":
    print("=== EHC Helpdesk Configuration ===")
    print(f"REDMINE_URL        : {REDMINE_URL}")
    print(f"REDMINE_PROJECT    : {REDMINE_PROJECT}")
    print(f"VLLM_BASE_URL      : {VLLM_BASE_URL}")
    print(f"VLLM_MODEL         : {VLLM_MODEL}")
    print(f"EMBED_MODEL        : {EMBED_MODEL}")
    print(f"RERANKER_MODEL     : {RERANKER_MODEL}")
    print(f"QDRANT_URL         : {QDRANT_URL}")
    print(f"QDRANT_COLLECTION  : {QDRANT_COLLECTION}")
    print(f"RETRIEVER_TOP_K    : {RETRIEVER_TOP_K}")
    print(f"RERANKER_TOP_N     : {RERANKER_TOP_N}")
    print(f"CONFIDENCE_THRESHOLD: {CONFIDENCE_THRESHOLD}")
    print(f"SESSION_MAX_TURNS  : {SESSION_MAX_TURNS}")
    print(f"TELEGRAM_BOT_TOKEN : {'***set***' if TELEGRAM_BOT_TOKEN else '(not set)'}")
    print(f"ZALO_OA_SECRET     : {'***set***' if ZALO_OA_SECRET else '(not set)'}")
    print(f"ZALO_ACCESS_TOKEN  : {'***set***' if ZALO_ACCESS_TOKEN else '(not set)'}")
    print(f"SLACK_BOT_TOKEN    : {'***set***' if SLACK_BOT_TOKEN else '(not set)'}")
    print(f"SLACK_SIGNING_SECRET: {'***set***' if SLACK_SIGNING_SECRET else '(not set)'}")
    print("\n✓ All required variables loaded successfully.")
