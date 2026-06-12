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
    """
    Fetch all FAQ issues from Redmine, paginating through all pages.
    Skips issues with empty or too-short descriptions.
    Returns a list of Document objects.
    """
    docs: list[Document] = []
    offset = 0
    limit = 100
    skipped = 0

    print(f"[INGESTOR] Fetching from {REDMINE_URL}/issues.json (project={REDMINE_PROJECT})")

    while True:
        params = {
            "project_id": REDMINE_PROJECT,
            "limit": limit,
            "offset": offset,
            "key": REDMINE_API_KEY,
            "status_id": "*",  # all statuses
            "include": "attachments",
        }

        response = httpx.get(
            f"{REDMINE_URL}/issues.json",
            params=params,
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        issues = data.get("issues", [])

        if not issues:
            break

        page_num = (offset // limit) + 1
        print(f"[INGESTOR] Page {page_num}: fetched {len(issues)} issues (offset={offset})")

        for issue in issues:
            issue_id = issue["id"]
            subject = issue.get("subject", "").strip()
            description = issue.get("description", "").strip()

            # Skip condition: empty or too short description
            if not description or len(description) < 20:
                reason = "empty description" if not description else f"too short ({len(description)} chars)"
                print(f"  [SKIP] id={issue_id} subject=\"{subject[:50]}\" reason=\"{reason}\"")
                skipped += 1
                continue

            # Normalize text
            subject = normalize(subject)
            description = normalize(description)

            attachments = issue.get("attachments", [])
            image_urls = [
                a["content_url"]
                for a in attachments
                if a.get("content_type", "").startswith("image/")
            ]

            doc = Document(
                issue_id=issue_id,
                subject=subject,
                description=description,
                project=REDMINE_PROJECT,
                url=f"{REDMINE_URL}/issues/{issue_id}",
                image_urls=image_urls,
            )
            docs.append(doc)

        offset += limit

    print(f"\n[INGESTOR] Done. Total usable: {len(docs)}, Skipped: {skipped}")
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