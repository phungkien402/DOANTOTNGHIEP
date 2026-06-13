"""
ingest_manual.py — Read HDSD markdown files, chunk by ## headings, embed and store in Qdrant.

Collection: ehc_manual
Embedding: same model as embedder.py (BAAI/bge-m3, dim=1024)

Run standalone: python3 -m data.ingest_manual
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from config import QDRANT_URL, EMBED_MODEL

MANUAL_COLLECTION = "ehc_manual"
HDSD_DIR = Path(__file__).parent / "hdsd"


def _chunk_markdown(filepath: Path) -> list[dict]:
    """
    Split a markdown file by ## headings.
    Returns list of dicts: {source, subject, body, chunk_text}
    """
    text = filepath.read_text(encoding="utf-8")
    source = filepath.stem  # e.g. "hdsd_minipacs"

    # Split by ## headings (keep the heading text)
    sections = re.split(r"^## ", text, flags=re.MULTILINE)

    chunks = []
    for section in sections[1:]:  # skip content before first ##
        lines = section.strip().split("\n", 1)
        title = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""

        # Skip empty sections
        if not body or len(body) < 20:
            continue

        chunk_text = f"Tiêu đề: {title}\nNội dung: {body}"
        chunks.append({
            "source": source,
            "subject": title,
            "body": body,
            "chunk_text": chunk_text,
        })

    return chunks


def ingest_manuals() -> int:
    """
    Read all .md files from HDSD_DIR, chunk, embed, and store in Qdrant.
    Returns number of chunks stored.
    """
    # Collect all chunks from all files
    all_chunks = []
    md_files = sorted(HDSD_DIR.glob("*.md"))

    if not md_files:
        print(f"[INGEST_MANUAL] No .md files found in {HDSD_DIR}")
        return 0

    for filepath in md_files:
        chunks = _chunk_markdown(filepath)
        print(f"[INGEST_MANUAL] {filepath.name}: {len(chunks)} sections")
        all_chunks.extend(chunks)

    print(f"[INGEST_MANUAL] Total chunks: {len(all_chunks)}")

    if not all_chunks:
        return 0

    # Embed
    print(f"[INGEST_MANUAL] Loading model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL, device="cpu")

    chunk_texts = [c["chunk_text"] for c in all_chunks]
    print(f"[INGEST_MANUAL] Encoding {len(chunk_texts)} chunks...")
    embeddings = model.encode(chunk_texts, batch_size=32, show_progress_bar=True)
    print(f"[INGEST_MANUAL] Embeddings shape: {embeddings.shape}")

    # Store in Qdrant
    client = QdrantClient(url=QDRANT_URL)

    print(f"[INGEST_MANUAL] Recreating collection '{MANUAL_COLLECTION}' (dim=1024, cosine)")
    if client.collection_exists(MANUAL_COLLECTION):
        client.delete_collection(MANUAL_COLLECTION)
    client.create_collection(
        collection_name=MANUAL_COLLECTION,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

    # Upsert all chunks
    points = [
        PointStruct(
            id=i + 1,
            vector=embeddings[i].tolist(),
            payload={
                "chunk_id": i + 1,
                "source": chunk["source"],
                "subject": chunk["subject"],
                "description": chunk["body"],
                "chunk_text": chunk["chunk_text"],
            },
        )
        for i, chunk in enumerate(all_chunks)
    ]

    client.upsert(collection_name=MANUAL_COLLECTION, points=points)
    print(f"[INGEST_MANUAL] Done. {len(points)} chunks stored in '{MANUAL_COLLECTION}'")
    return len(points)


if __name__ == "__main__":
    print("=== ingest_manual.py — Ingest HDSD manuals ===\n")

    count = ingest_manuals()
    print(f"\nStored {count} chunks in collection '{MANUAL_COLLECTION}'")

    # Test query
    print(f"\n--- Test query: 'kết nối PACS' ---")
    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    query_vector = model.encode("kết nối PACS").tolist()

    client = QdrantClient(url=QDRANT_URL)
    results = client.search(
        collection_name=MANUAL_COLLECTION,
        query_vector=query_vector,
        limit=3,
    )

    for i, r in enumerate(results, 1):
        print(f"  #{i} score={r.score:.3f} | {r.payload['subject']}")

    print("\n✓ ingest_manual.py completed.")
