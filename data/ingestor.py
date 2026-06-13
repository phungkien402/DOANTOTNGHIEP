"""
Ingestor — Fetches FAQ issues from the Redmine API.

Output: list of Document objects {issue_id, subject, description, url, project}
Skips issues where description is empty or shorter than 20 characters.
Normalizes arrow separators (-->, =>, ==>) to → for consistency.

Run standalone: python -m data.ingestor
"""

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import REDMINE_URL, REDMINE_API_KEY, REDMINE_PROJECT


@dataclass
class Document:
    """A single FAQ document from Redmine."""
    issue_id: int
    subject: str
    description: str
    project: str
    url: str
    image_urls: list = None


def normalize(text: str) -> str:
    """Normalize arrow separators and collapse whitespace."""
    # Order matters: longest arrows first to avoid partial replacements
    text = text.replace("==>", "→")
    text = text.replace("-->", "→")
    text = text.replace("=>", "→")
    # Collapse multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def fetch_all_documents() -> list[Document]:
    docs = []
    offset = 0
    limit = 100
    skipped = 0

    print(f"[INGESTOR] Fetching from {REDMINE_URL}/issues.json (project={REDMINE_PROJECT})")

    with httpx.Client() as client:  # ← dùng 1 client cho cả list + attachment requests
        while True:
            params = {
                "project_id": REDMINE_PROJECT,
                "limit": limit,
                "offset": offset,
                "key": REDMINE_API_KEY,
                "status_id": "*",
            }
            response = client.get(f"{REDMINE_URL}/issues.json", params=params, timeout=30.0)
            response.raise_for_status()
            issues = response.json().get("issues", [])
            if not issues:
                break

            for issue in issues:
                issue_id = issue["id"]
                subject = issue.get("subject", "").strip()
                description = issue.get("description", "").strip()

                if not description or len(description) < 20:
                    skipped += 1
                    continue

                subject = normalize(subject)
                description = normalize(description)

                # Fetch attachments riêng nếu có
                image_urls = []
                try:
                    r = client.get(
                        f"{REDMINE_URL}/issues/{issue_id}.json",
                        params={"key": REDMINE_API_KEY, "include": "attachments"},
                        timeout=10.0,
                    )
                    attachments = r.json().get("issue", {}).get("attachments", [])
                    image_urls = [
                        a["content_url"] for a in attachments
                        if a.get("content_type", "").startswith("image/")
                    ]
                    if image_urls:
                        print(f"  [IMAGE] id={issue_id} → {len(image_urls)} ảnh")
                except Exception as e:
                    print(f"  [WARN] id={issue_id} attachments failed: {e}")
                    image_urls = []

                docs.append(Document(
                    issue_id=issue_id,
                    subject=subject,
                    description=description,
                    project=REDMINE_PROJECT,
                    url=f"{REDMINE_URL}/issues/{issue_id}",
                    image_urls=image_urls,
                ))

            offset += limit

    print(f"\n[INGESTOR] Done. Total: {len(docs)}, Skipped: {skipped}")
    return docs


if __name__ == "__main__":
    docs = fetch_all_documents()
    print(f"\nTotal fetched : {len(docs)} documents")
    print(f"\nFirst 3 examples:")
    for doc in docs[:3]:
        print(f"  ---")
        print(f"  ID         : {doc.issue_id}")
        print(f"  Subject    : {doc.subject}")
        print(f"  Description: {doc.description[:100]}")
        print(f"  URL        : {doc.url}")
        print(f"  Image URLs : {doc.image_urls}")