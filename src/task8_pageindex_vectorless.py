"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]]. In response thật ra
(json.dumps(...)) trước khi viết logic parse, đừng đoán schema từ ví dụ code cũ.

SCHEMA THẬT (đo được ngày 2026-08-04, khác code mẫu ở 2 chỗ):
    retrieved_nodes[i] = {
        "id": "0001",
        "title": str,
        "metadata": [doc_id, filename, "", description, "File", ...],   <-- LIST, không phải dict
        "relevant_contents": [[{section_title, physical_index, relevant_content}]]
    }
    - metadata[1] mới là tên file. Code mẫu ghi metadata là dict là sai.
    - Query 1 tài liệu không khớp -> trả về 0 node (status vẫn "completed").
      Nên phải hỏi NHIỀU tài liệu rồi gộp, chứ hỏi 1 file là hay ra rỗng.

PageIndex nhận PDF, không nhận .md -> upload từ data/landing/legal/.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
LEGAL_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
DOC_ID_CACHE = Path(__file__).parent.parent / "pageindex_doc_ids.json"

# Số tài liệu hỏi song song. PageIndex mất ~4-10s mỗi query nên chạy tuần tự
# 11 file sẽ tốn cả phút — quá chậm cho một nhánh fallback.
MAX_PARALLEL_DOCS = 8
POLL_TIMEOUT_S = 45


def _client():
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("Thiếu PAGEINDEX_API_KEY trong .env — đăng ký tại pageindex.ai")
    from pageindex.client import PageIndexClient

    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def upload_documents():
    """
    Upload toàn bộ documents lên PageIndex.

    Upload PDF gốc trong data/landing/legal/ chứ không phải markdown:
    PageIndex phân tích cấu trúc trang (physical_index) nên cần file PDF.
    """
    client = _client()

    existing = {
        d["name"]: d["id"] for d in client.list_documents(limit=100).get("documents", [])
    }
    doc_ids = dict(existing)

    for pdf in sorted(LEGAL_DIR.glob("*.pdf")):
        if pdf.name in existing:
            print(f"  · đã có, bỏ qua: {pdf.name}")
            continue
        resp = client.submit_document(str(pdf))
        doc_id = resp.get("doc_id") or resp.get("id")
        doc_ids[pdf.name] = doc_id
        print(f"  ✓ uploaded: {pdf.name} -> {doc_id}")

    DOC_ID_CACHE.write_text(json.dumps(doc_ids, ensure_ascii=False, indent=2))
    print(f"\n✓ {len(doc_ids)} tài liệu, doc_id lưu tại {DOC_ID_CACHE.name}")
    return doc_ids


def _ready_documents(client) -> list[tuple[str, str]]:
    """Trả về [(doc_id, filename)] của các tài liệu đã xử lý xong."""
    docs = client.list_documents(limit=100).get("documents", [])
    return [(d["id"], d["name"]) for d in docs if d.get("status") == "completed"]


def _query_one(client, doc_id: str, filename: str, query: str) -> list[dict]:
    """Hỏi 1 tài liệu, poll tới khi xong, trả về các đoạn liên quan."""
    import time

    try:
        rid = client.submit_query(doc_id=doc_id, query=query)["retrieval_id"]
    except Exception:
        return []

    deadline = time.time() + POLL_TIMEOUT_S
    result = {}
    while time.time() < deadline:
        result = client.get_retrieval(rid)
        if result.get("status") in ("completed", "failed", "error"):
            break
        time.sleep(1.5)

    out = []
    for node in result.get("retrieved_nodes", []):
        # relevant_contents lồng 2 tầng list.
        for group in node.get("relevant_contents", []):
            for item in group:
                text = (item.get("relevant_content") or "").strip()
                if not text:
                    continue
                out.append({
                    "content": text,
                    "metadata": {
                        "source": filename,
                        "type": "legal",
                        "section": item.get("section_title") or node.get("title"),
                    },
                })
    return out


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    client = _client()
    docs = _ready_documents(client)[:MAX_PARALLEL_DOCS]
    if not docs:
        return []

    # Hỏi song song: mỗi query mất ~4-10s, tuần tự 8 file là gần một phút.
    with ThreadPoolExecutor(max_workers=len(docs)) as pool:
        batches = pool.map(lambda d: _query_one(client, d[0], d[1], query), docs)

    hits = [h for batch in batches for h in batch]

    # PageIndex KHÔNG trả điểm số. Tự gán theo thứ hạng để Task 9 còn so sánh
    # được, thang giảm dần 0.9, 0.8, ... cho khớp với thang [0,1] của cosine.
    results = []
    for rank, hit in enumerate(hits[:top_k]):
        results.append({
            **hit,
            "score": round(max(0.1, 0.9 - rank * 0.1), 2),
            "source": "pageindex",
        })
    return results


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("tuition fee payment methods", top_k=3)
        for r in results:
            print(f"[{r['score']:.2f}] {r['metadata']['source']} | {r['content'][:90]}...")
