"""
Reranker — uses a cross-encoder model (bge-reranker-v2-m3) to rescore
retrieved chunks by reading each (question, chunk) pair directly.

This is the primary fix for "chaotic" outputs — vector similarity alone
returns loosely related chunks; the reranker produces much more accurate
relevance scores.

Run standalone: python -m core.reranker
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from FlagEmbedding import FlagReranker
from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL, RERANKER_TOP_N, CONFIDENCE_THRESHOLD
from core.models import RetrievedChunk

# Module-level singleton — loaded once when module is first imported
print(f"[RERANKER] Loading model: {RERANKER_MODEL}")
from sentence_transformers import CrossEncoder
_reranker = CrossEncoder(RERANKER_MODEL, device='cpu')


def rerank(query: str, chunks: list[RetrievedChunk], top_n: int = None) -> list[RetrievedChunk]:
    """
    Rescore chunks using the cross-encoder reranker.
    Returns the top_n chunks sorted by reranker score (descending).
    """
    if top_n is None:
        top_n = RERANKER_TOP_N

    if not chunks:
        print("[RERANKER] No chunks to rerank.")
        return []

    print(f"[RERANKER] Input: {len(chunks)} chunks")

    # Build (query, chunk_text) pairs for cross-encoder
    pairs = [[query, chunk.text] for chunk in chunks]

    # Compute reranker scores
    scores = _reranker.predict([(query, chunk.text) for chunk in chunks])

    # Handle single result (returns float instead of list)
    if isinstance(scores, float):
        scores = [scores]

    # Update chunk scores with reranker scores and sort
    scored_chunks = []
    for chunk, score in zip(chunks, scores):
        new_chunk = RetrievedChunk(
            text=chunk.text,
            score=score,
            metadata=chunk.metadata,
        )
        scored_chunks.append(new_chunk)

    # Sort by score descending, keep top_n
    scored_chunks.sort(key=lambda c: c.score, reverse=True)
    top_chunks = scored_chunks[:top_n]

    # Log results
    print(f"[RERANKER] After reranking (top {top_n}):")
    for i, c in enumerate(top_chunks, 1):
        print(f"  #{i}  score={c.score:.4f} | {c.metadata.get('subject', 'N/A')}")

    top_score = top_chunks[0].score if top_chunks else 0.0
    status = "CONFIDENT" if top_score >= CONFIDENCE_THRESHOLD else "FALLBACK"
    print(f"[RERANKER] Top score: {top_score:.4f}  (threshold: {CONFIDENCE_THRESHOLD}) → {status}")

    return top_chunks


if __name__ == "__main__":
    from core.retriever import retrieve

    test_queries = [
        "in bảng kê khám bệnh ở đâu",
        "cách gộp hồ sơ bệnh nhân trùng",
    ]
    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: \"{q}\"")
        chunks = retrieve(q, top_k=10)
        ranked = rerank(q, chunks, top_n=3)
        print()
