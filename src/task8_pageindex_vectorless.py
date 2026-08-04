"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex
Docs: https://docs.pageindex.ai/quickstart (import đúng là `from pageindex import
PageIndexClient` — README/ví dụ cũ ghi `pageindex.client` là sai).

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

PageIndex chỉ nhận PDF (không nhận .md) — upload trực tiếp từ
data/landing/legal/ (bản gốc Task 1), không convert markdown.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai, lấy API key dạng pix_...
    2. Điền PAGEINDEX_API_KEY vào .env (xem .env.example)
    3. Chạy `python -m src.task8_pageindex_vectorless` để upload + poll tới khi
       document status == "completed", rồi lưu doc_id vào data/pageindex_doc_ids.json
    4. pageindex_search() đọc lại manifest đó, không upload lại mỗi lần query

Lưu ý: API `/retrieval` của PageIndex hiện là legacy (docs khuyên dùng chat API
cho use case mới, nhưng legacy vẫn hoạt động và đúng schema retrieval-style
"content" cần cho hàm này). Response trả "retrieved_nodes": mỗi node có "title"/
"node_id" + "relevant_contents". Schema "relevant_contents" từng đổi giữa các bản
(có lúc là list phẳng [{page_index, relevant_content}], có lúc list lồng list
[{section_title, relevant_content}]) — _parse_retrieved_nodes() xử lý cả 2 dạng.
Nếu chạy thật mà parse ra rỗng, in json.dumps(retrieval) ra xem schema thật trước
khi sửa parser, đừng đoán.
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
LEGAL_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
DOC_MANIFEST_PATH = Path(__file__).parent.parent / "data" / "pageindex_doc_ids.json"

POLL_INTERVAL_SECONDS = 5
POLL_TIMEOUT_SECONDS = 300


def _get_client():
    """Khởi tạo PageIndexClient. Raise rõ ràng nếu thiếu key (Task 9 catch được)."""
    if not PAGEINDEX_API_KEY:
        raise RuntimeError(
            "Thiếu PAGEINDEX_API_KEY trong .env — đăng ký tại https://pageindex.ai/ "
            "(key dạng pix_...), xem .env.example."
        )
    from pageindex import PageIndexClient

    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def _poll_until_completed(check_fn, label: str) -> dict:
    """Gọi check_fn() lặp lại tới khi result['status'] == 'completed' hoặc hết timeout."""
    start = time.time()
    while True:
        result = check_fn()
        status = result.get("status")
        if status == "completed":
            return result
        if status in ("failed", "error"):
            raise RuntimeError(f"{label} thất bại (status={status}): {result}")
        if time.time() - start > POLL_TIMEOUT_SECONDS:
            raise TimeoutError(f"{label}: quá {POLL_TIMEOUT_SECONDS}s vẫn '{status}', bỏ cuộc")
        time.sleep(POLL_INTERVAL_SECONDS)


def upload_documents() -> dict:
    """
    Upload toàn bộ PDF trong data/landing/legal/ lên PageIndex, chờ xử lý xong,
    rồi lưu mapping {filename: doc_id} vào data/pageindex_doc_ids.json để
    pageindex_search() dùng lại — không upload lại mỗi lần query.

    Returns:
        dict {filename: doc_id}
    """
    client = _get_client()

    pdf_files = sorted(LEGAL_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"Không tìm thấy PDF nào trong {LEGAL_DIR}")

    doc_ids = {}
    for pdf_path in pdf_files:
        print(f"  Uploading {pdf_path.name}...")
        submitted = client.submit_document(str(pdf_path))
        doc_id = submitted["doc_id"]

        _poll_until_completed(lambda: client.get_document(doc_id), f"Xử lý {pdf_path.name}")

        doc_ids[pdf_path.name] = doc_id
        print(f"  ✓ {pdf_path.name} -> {doc_id}")

    DOC_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_MANIFEST_PATH.write_text(json.dumps(doc_ids, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ Đã lưu manifest: {DOC_MANIFEST_PATH}")
    return doc_ids


def _load_doc_ids() -> dict:
    if not DOC_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Chưa có {DOC_MANIFEST_PATH.name} — chạy upload_documents() "
            "(hoặc `python -m src.task8_pageindex_vectorless`) trước."
        )
    return json.loads(DOC_MANIFEST_PATH.read_text(encoding="utf-8"))


def _parse_retrieved_nodes(retrieved_nodes: list, max_nodes: int = 2) -> list[dict]:
    """
    Parse retrieved_nodes -> list content item {content, section_title, physical_index}.
    Xử lý cả 2 dạng relevant_contents đã thấy trong thực tế (xem note đầu file):
    list phẳng các dict, hoặc list lồng list các dict. Field vị trí trong trang
    thật ra tên là "physical_index" (dạng "<physical_index_N>"), không phải
    "page_index" — xác nhận bằng cách in raw response thật, đúng như note trên.
    """
    items = []
    for node in retrieved_nodes[:max_nodes]:
        node_title = node.get("title") or node.get("section_title")
        relevant_contents = node.get("relevant_contents", [])

        flat = []
        for entry in relevant_contents:
            if isinstance(entry, list):
                flat.extend(entry)
            else:
                flat.append(entry)

        for item in flat:
            items.append({
                "content": item.get("relevant_content", ""),
                "section_title": item.get("section_title") or node_title,
                "physical_index": item.get("physical_index"),
            })
    return items


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

    Lưu ý hiệu năng: legacy /retrieval endpoint là per-document (không nhận
    list doc_id), nên hàm này loop qua toàn bộ doc đã upload rồi gộp kết quả
    — với 9 PDF, mỗi query có thể mất vài chục giây do phải poll từng doc.
    Đây là fallback (chỉ chạy khi Cosine < 0.48 ở Task 9), không phải đường
    chính, nên đánh đổi độ trễ lấy độ phủ là chấp nhận được.
    """
    client = _get_client()
    doc_ids = _load_doc_ids()

    all_items = []
    for filename, doc_id in doc_ids.items():
        submitted = client.submit_query(doc_id=doc_id, query=query)
        retrieval_id = submitted.get("retrieval_id") or submitted.get("id")

        retrieval = _poll_until_completed(
            lambda: client.get_retrieval(retrieval_id),
            f"Retrieval query trên {filename}",
        )

        for item in _parse_retrieved_nodes(retrieval.get("retrieved_nodes", [])):
            item["source_file"] = filename
            all_items.append(item)

        if len(all_items) >= top_k:
            break  # đủ ứng viên rồi, khỏi hỏi tiếp các doc còn lại

    # PageIndex không trả score số — gán theo rank giảm dần (item trả về trước
    # theo thứ tự PageIndex xếp = liên quan hơn).
    results = []
    for i, item in enumerate(all_items[:top_k]):
        results.append({
            "content": item["content"],
            "score": round(max(0.0, 1.0 - i / max(top_k, 1)), 4),
            "metadata": {
                "section": item.get("section_title"),
                "source": item.get("source_file"),
                "physical_index": item.get("physical_index"),
            },
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
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
