"""
RAG Pipeline orchestrator — iterative retrieval.
Accepts a standard Message object, runs it through 6 steps:
  1. Fast Retrieve (top 3, no rerank) — get initial context
  2. Analyze + Rewrite (single LLM call) — intent & formal query
  3. Full Retrieve (top K) — embed rewritten query
  4. Rerank (top N) — cross-encoder rescore
  5. Confidence Check — route to fallback if below threshold
  6. Generate — grounded answer with user intent acknowledgment

Why iterative: LLM needs EHC context to analyze intent accurately —
blind intent analysis is ineffective for domain-specific terminology.

Run standalone: python -m core.pipeline
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RETRIEVER_TOP_K, RERANKER_TOP_N, CONFIDENCE_THRESHOLD, MAINTENANCE_MODE
from core.models import Message, Answer
from core import query_rewriter, retriever, reranker, generator, confidence, fallback
from core.query_rewriter import analyze_and_rewrite
from core.generator import GeneratorError, LLMUnavailableError
from core.intent_guard import classify, chat_fallback

# --- Maintenance mode (toggled at runtime via /admin/maintenance) ---
_maintenance_mode: bool = MAINTENANCE_MODE

MAINTENANCE_MESSAGE = (
    "⚙️ Hệ thống đang bảo trì, vui lòng thử lại sau ít phút. "
    "Xin lỗi vì sự bất tiện này 🙏"
)


def set_maintenance_mode(enabled: bool):
    """Toggle maintenance mode at runtime."""
    global _maintenance_mode
    _maintenance_mode = enabled
    print(f"[PIPELINE] Maintenance mode: {'ON' if enabled else 'OFF'}")


def is_maintenance_mode() -> bool:
    """Check if maintenance mode is active."""
    return _maintenance_mode


def run(message: Message, session_history: list) -> Answer:
    """
    Execute the full RAG pipeline on a message.
    Iterative retrieval flow:
      1. Fast retrieve (top 3, no rerank) for context
      2. Analyze + rewrite (single LLM call)
      3. Full retrieve (top K)
      4. Rerank (top N)
      5. Confidence check
      6. Generate grounded answer
    """
    # Short-circuit if maintenance mode is active
    if _maintenance_mode:
        print(f"[PIPELINE] Maintenance mode active — returning maintenance message")
        return Answer(
            text=MAINTENANCE_MESSAGE,
            confidence=0.0,
            source_chunks=[],
            is_fallback=True,
            rewritten_question="",
        )

    print(f"\n{'='*60}")
    print(f"[PIPELINE] Input: \"{message.text}\"")
    print(f"{'='*60}")

    # Guard: off-topic / small talk (LLM classifier)
    if classify(message.text):
        print(f"[PIPELINE] Off-topic detected: \"{message.text}\" → chat fallback")
        fallback_text = chat_fallback(message.text)
        return Answer(
            text=fallback_text,
            confidence=1.0,
            source_chunks=[],
            is_fallback=False,
            rewritten_question=message.text,
        )

    # Step 1: Fast retrieve — embed original query, get top 3 (no reranking)
    print(f"\n[PIPELINE] Step 1: Fast retrieve (top 3)")
    fast_chunks = retriever.retrieve(message.text, top_k=3)

    # Step 2: Analyze intent + rewrite query in a single LLM call
    print(f"\n[PIPELINE] Step 2: Analyze + Rewrite (single call)")
    try:
        if fast_chunks:
            user_intent, rewritten, answerable = analyze_and_rewrite(message.text, chunks=fast_chunks)

        else:
            user_intent, rewritten, answerable = analyze_and_rewrite(message.text)
    except LLMUnavailableError:
        print("[PIPELINE] vLLM unavailable during analyze+rewrite → LLM unavailable response")
        return Answer(
            text="⚠️ Hệ thống AI đang bận hoặc đang khởi động lại, vui lòng thử lại sau 1–2 phút. Nếu vẫn lỗi, liên hệ bộ phận IT để kiểm tra server.",
            confidence=0.0,
            source_chunks=[],
            is_fallback=True,
            rewritten_question="",
        )
    print(f"[PIPELINE] analyze_and_rewrite: intent=\"{user_intent}\" query=\"{rewritten}\"")

    # Step 3: Full retrieve with rewritten query
    print(f"\n[PIPELINE] Step 3: Full retrieve (top {RETRIEVER_TOP_K})")
    chunks = retriever.retrieve(rewritten, top_k=RETRIEVER_TOP_K)

    if not chunks:
        print("[PIPELINE] No chunks retrieved → fallback")
        return fallback.handle(message, session_history)

    # Step 4: Rerank candidates
    print(f"\n[PIPELINE] Step 4: Rerank (top {RERANKER_TOP_N})")
    ranked_chunks = reranker.rerank(rewritten, chunks, top_n=RERANKER_TOP_N)

    if not ranked_chunks:
        print("[PIPELINE] No chunks after reranking → fallback")
        return fallback.handle(message, session_history)

    # Step 5: Check confidence
    if not confidence.is_confident(ranked_chunks[0], threshold=CONFIDENCE_THRESHOLD):
        print(f"[PIPELINE] Low confidence ({ranked_chunks[0].score:.4f} < {CONFIDENCE_THRESHOLD}) → fallback")
        return fallback.handle(message, session_history)

    # Step 6: Generate grounded answer
    print(f"\n[PIPELINE] Step 6: Generate answer")
    try:
        answer_text = generator.generate(rewritten, ranked_chunks, session_history, user_intent=user_intent)
    except LLMUnavailableError:
        print("[PIPELINE] vLLM unavailable during generation → LLM unavailable response")
        return Answer(
            text="⚠️ Hệ thống AI đang bận hoặc đang khởi động lại, vui lòng thử lại sau 1–2 phút. Nếu vẫn lỗi, liên hệ bộ phận IT để kiểm tra server.",
            confidence=0.0,
            source_chunks=[],
            is_fallback=True,
            rewritten_question=rewritten,
        )
    except GeneratorError:
        print("[PIPELINE] Generator failed (vLLM unavailable) → fallback")
        return fallback.handle(message, session_history)

    # Cast float32 → float for JSON serialization (RQ job result)
    for chunk in ranked_chunks:
        chunk.score = float(chunk.score)

    answer = Answer(
        text=answer_text,
        confidence=float(ranked_chunks[0].score),
        source_chunks=ranked_chunks,
        is_fallback=False,
        rewritten_question=rewritten,
    )

    print(f"\n[PIPELINE] Done. confidence={answer.confidence:.4f} fallback={answer.is_fallback}")
    return answer


if __name__ == "__main__":
    print("=== EHC RAG Pipeline — Interactive Test ===")
    print("Type a question (or 'quit' to exit):\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q == "quit":
            break
        msg = Message(
            user_id="test", session_id="s1",
            text=q, timestamp=time.time(), platform="web"
        )
        answer = run(msg, [])
        print(f"\nBot: {answer.text}")
        print(f"     [confidence={answer.confidence:.2f}  fallback={answer.is_fallback}]\n")
