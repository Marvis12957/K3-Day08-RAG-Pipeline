"""Task 8 - PageIndex vectorless fallback."""

import os
import time

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
MAX_PARALLEL_DOCS = 11
POLL_TIMEOUT_S = 45


def _client():
    if not PAGEINDEX_API_KEY:
        return None
    try:
        from pageindex.client import PageIndexClient
    except ImportError:
        return None
    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def upload_documents() -> dict[str, str]:
    client = _client()
    if client is None:
        return {}

    try:
        docs = client.list_documents(limit=100).get("documents", [])
    except Exception:
        return {}
    return {
        (doc.get("name") or doc.get("filename") or doc.get("title") or doc["id"]): doc.get("id") or doc.get("doc_id")
        for doc in docs
        if doc.get("id") or doc.get("doc_id")
    }


def _ready_documents(client) -> list[tuple[str, str]]:
    try:
        docs = client.list_documents(limit=100).get("documents", [])
    except Exception:
        return []
    ready = []
    for doc in docs:
        doc_id = doc.get("id") or doc.get("doc_id")
        name = doc.get("name") or doc.get("filename") or doc.get("title") or doc_id
        if doc_id and doc.get("status") == "completed":
            ready.append((doc_id, name))
    return ready


def _query_one(client, doc_id: str, filename: str, query: str) -> list[dict]:
    if client is None:
        return []
    try:
        submitted = client.submit_query(doc_id=doc_id, query=query)
        retrieval_id = submitted.get("retrieval_id") or submitted.get("id")
    except Exception:
        return []

    deadline = time.time() + POLL_TIMEOUT_S
    retrieval = {}
    while time.time() < deadline:
        retrieval = client.get_retrieval(retrieval_id)
        if retrieval.get("status") in {"completed", "failed", "error"}:
            break
        time.sleep(1.5)

    hits = []
    for node in retrieval.get("retrieved_nodes", []):
        for group in node.get("relevant_contents", []):
            for item in group:
                text = (item.get("relevant_content") or "").strip()
                if text:
                    hits.append({
                        "content": text,
                        "metadata": {
                            "source": filename,
                            "type": "legal",
                            "section": item.get("section_title") or node.get("title"),
                        },
                    })
    return hits


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    client = _client()
    if client is None or top_k <= 0:
        return []

    docs = _ready_documents(client)[:MAX_PARALLEL_DOCS]
    if not docs:
        return []

    hits = []
    for doc_id, filename in docs:
        hits.extend(_query_one(client, doc_id, filename, query))
        if len(hits) >= top_k:
            break
    return [
        {**hit, "score": round(max(0.1, 0.9 - rank * 0.1), 2), "source": "pageindex"}
        for rank, hit in enumerate(hits[:top_k])
    ]


if __name__ == "__main__":
    print(upload_documents())
    for result in pageindex_search("tuition fee payment methods", top_k=3):
        print(f"[{result['score']:.2f}] {result['metadata']['source']} | {result['content'][:90]}...")
