"""
Retriever — embeds the rewritten query with bge-m3 and fetches Top-K
most similar chunks from Qdrant.

Uses the SAME embedding model as data/embedder.py — mismatched models
will silently produce poor results.

Run standalone: python -m core.retriever
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from config import QDRANT_URL, QDRANT_COLLECTION, EMBED_MODEL, RETRIEVER_TOP_K
from core.models import RetrievedChunk

# Module-level singletons — loaded once when module is first imported
print(f"[RETRIEVER] Loading embedding model: {EMBED_MODEL}")
_model = SentenceTransformer(EMBED_MODEL, device='cpu')
_client = QdrantClient(url=QDRANT_URL)


def retrieve(query: str, top_k: int = None) -> list[RetrievedChunk]:
    """
    Embed the query and search Qdrant for the top_k most similar chunks.
    Returns a list of RetrievedChunk with raw cosine similarity scores.
    """
    if top_k is None:
        top_k = RETRIEVER_TOP_K

    print(f"[RETRIEVER] Query: \"{query}\"")

    # Embed the query
    query_vector = _model.encode(query).tolist()

    # Search Qdrant
    results = _client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_vector,
        limit=top_k,
    )

    # Convert to RetrievedChunk objects
    chunks = []
    for r in results:
        chunk = RetrievedChunk(
            text=r.payload.get("chunk_text", ""),
            score=r.score,
            metadata={
                "issue_id": r.payload.get("issue_id"),
                "subject": r.payload.get("subject"),
                "description": r.payload.get("description"),
                "project": r.payload.get("project"),
                "url": r.payload.get("url"),
                "image_urls": r.payload.get("image_urls", []),
            },
        )
        chunks.append(chunk)

    # Log results
    print(f"[RETRIEVER] Top {len(chunks)} chunks retrieved:")
    for i, c in enumerate(chunks, 1):
        print(f"  #{i}  score={c.score:.3f} | {c.metadata['subject']}")

    return chunks


if __name__ == "__main__":
    test_queries = [
        "in bảng kê khám bệnh ở đâu",
        "cách gộp hồ sơ bệnh nhân trùng",
        "xem tồn kho thuốc",
    ]
    for q in test_queries:
        print(f"\n{'='*60}")
        results = retrieve(q, top_k=5)
        print()
