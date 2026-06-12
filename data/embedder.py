"""
Embedder — Takes a list of Documents, embeds them with bge-m3, stores in Qdrant.

Each chunk is stored with payload: {issue_id, subject, description, project, url, chunk_text}
Embedding dimension: 1024 (bge-m3 output size)
Uses upsert so re-running is always safe (no duplicates).

Run standalone: python -m data.embedder
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from config import QDRANT_URL, QDRANT_COLLECTION, EMBED_MODEL
from data.ingestor import Document


def build_chunk_text(doc: Document) -> str:
    """
    Combine subject + description into the text to embed.
    Subject carries most semantic meaning since descriptions are very short.
    """
    return f"Câu hỏi: {doc.subject}\nHướng dẫn: {doc.description}"


def embed_and_store(docs: list[Document], recreate: bool = False) -> int:
    """
    Embed all documents and upsert into Qdrant.
    Args:
        docs: list of Document objects to embed
        recreate: if True, drop and recreate collection (used by full reindex).
                  if False (default), create only if not exists, then upsert safely.
    Returns the number of chunks stored.
    """
    if not docs:
        print("[EMBEDDER] No documents to embed.")
        return 0

    # Build chunk texts
    chunk_texts = [build_chunk_text(doc) for doc in docs]
    print(f"[EMBEDDER] Built {len(chunk_texts)} chunk texts")
    print(f"[EMBEDDER] Example chunk:\n  {chunk_texts[0][:120]}...")

    # Load embedding model
    print(f"[EMBEDDER] Loading model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL, device='cpu')

    # Generate embeddings in batches
    print(f"[EMBEDDER] Encoding {len(chunk_texts)} texts (batch_size=32)...")
    embeddings = model.encode(chunk_texts, batch_size=32, show_progress_bar=True)
    print(f"[EMBEDDER] Embeddings shape: {embeddings.shape}")

    # Connect to Qdrant
    client = QdrantClient(url=QDRANT_URL)

    # Collection management
    if recreate:
        print(f"[EMBEDDER] Recreating collection '{QDRANT_COLLECTION}' (dim=1024, cosine)")
        if client.collection_exists(QDRANT_COLLECTION):
            client.delete_collection(QDRANT_COLLECTION)
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
    else:
        if not client.collection_exists(QDRANT_COLLECTION):
            print(f"[EMBEDDER] Creating collection '{QDRANT_COLLECTION}' (dim=1024, cosine)")
            client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )
        else:
            print(f"[EMBEDDER] Collection '{QDRANT_COLLECTION}' exists, upserting...")

    # Upsert in batches of 100
    batch_size = 100
    total_upserted = 0

    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]
        batch_texts = chunk_texts[i:i + batch_size]

        points = [
            PointStruct(
                id=doc.issue_id,
                vector=embedding.tolist(),
                payload={
                    "issue_id": doc.issue_id,
                    "subject": doc.subject,
                    "description": doc.description,
                    "project": doc.project,
                    "url": doc.url,
                    "chunk_text": chunk_text,
                    "image_urls": doc.image_urls or [], 
                },
            )
            for doc, embedding, chunk_text in zip(batch_docs, batch_embeddings, batch_texts)
        ]

        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
        total_upserted += len(points)
        print(f"[EMBEDDER] Upserted batch {i // batch_size + 1}: {len(points)} points (total: {total_upserted})")

    print(f"[EMBEDDER] Done. {total_upserted} chunks stored in '{QDRANT_COLLECTION}'")
    return total_upserted


if __name__ == "__main__":
    from data.ingestor import fetch_all_documents

    docs = fetch_all_documents()
    count = embed_and_store(docs)
    print(f"\n{'='*50}")
    print(f"Stored {count} chunks in Qdrant collection '{QDRANT_COLLECTION}'")

    # Sanity check: run a test query
    print(f"\n--- Sanity Check ---")
    print("Test query: 'in bảng kê khám bệnh ở đâu'")

    model = SentenceTransformer(EMBED_MODEL, device='cpu')
    query_vector = model.encode("in bảng kê khám bệnh ở đâu").tolist()

    client = QdrantClient(url=QDRANT_URL)
    results = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_vector,
        limit=3,
    )

    for i, r in enumerate(results, 1):
        print(f"  #{i} score={r.score:.3f} | {r.payload['subject']}")
