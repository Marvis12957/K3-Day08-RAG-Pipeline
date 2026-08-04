"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)

Vì sao BM25 index CHUNK chứ không index nguyên file:
    Task 9 gộp kết quả semantic + lexical bằng RRF. RRF chỉ gộp được thứ hạng
    của những thứ CÙNG ĐƠN VỊ. Nếu semantic trả về chunk ~800 ký tự còn BM25
    trả về nguyên file 20.000 ký tự thì hai danh sách không so được với nhau,
    và file dài sẽ luôn thắng vì chứa nhiều từ khoá hơn một cách máy móc.
    Nên ở đây tái dùng thẳng chunk_documents() của Task 4 — một nguồn chunk
    duy nhất cho cả dense lẫn sparse.
"""

from .task4_chunking_indexing import chunk_documents, load_documents

# Corpus và index được dựng lazy ở lần search đầu tiên, sau đó cache lại.
CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}
_bm25 = None


def _tokenize(text: str) -> list[str]:
    """
    Tokenize đơn giản: hạ chữ thường rồi tách theo khoảng trắng.

    Đủ dùng vì corpus là tiếng Anh. Với tiếng Việt cần tách từ ghép
    ("học phí" là 1 từ, không phải 2) thì nên thay bằng underthesea.
    """
    return text.lower().split()


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def _ensure_index():
    """Nạp corpus + dựng BM25 index nếu chưa có."""
    global CORPUS, _bm25
    if _bm25 is None:
        CORPUS = chunk_documents(load_documents())
        _bm25 = build_bm25_index(CORPUS)
    return _bm25


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    import numpy as np

    bm25 = _ensure_index()
    scores = bm25.get_scores(_tokenize(query))

    results = []
    for idx in np.argsort(scores)[::-1][:top_k]:
        # Bỏ chunk không khớp từ khoá nào. Giữ lại chỉ làm nhiễu bước gộp RRF
        # ở Task 9 vì chúng vẫn chiếm thứ hạng.
        if scores[idx] <= 0:
            continue
        results.append({
            "content": CORPUS[idx]["content"],
            "score": float(scores[idx]),
            "metadata": CORPUS[idx]["metadata"],
        })
    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("tuition fee payment methods", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['metadata']['source']} | {r['content'][:80]}...")
