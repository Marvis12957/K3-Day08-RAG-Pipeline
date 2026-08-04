"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4

Điểm trả về ở đây là COSINE SIMILARITY GỐC, thang [0,1] có ý nghĩa thật.
Task 9 sẽ dùng chính điểm này (KHÔNG phải điểm RRF sau khi fuse) để quyết
định có fallback sang PageIndex hay không — xem ghi chú trong task9.
"""

from .task4_chunking_indexing import get_collection, get_embedding_model


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    # Query PHẢI được embed bằng đúng model đã dùng lúc index (Task 4).
    # Dùng model khác là vector rơi vào không gian khác, cosine vô nghĩa.
    model = get_embedding_model()
    query_vector = model.encode(query).tolist()

    collection = get_collection()
    if collection.count() == 0:
        # Chưa index -> trả list rỗng thay vì nổ, để Task 9 còn fallback được.
        return []

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        # Chroma trả về cosine DISTANCE (0 = giống hệt). Đổi sang similarity.
        # Kẹp sàn 0.0 vì với vector chưa chuẩn hoá, distance có thể > 1.
        score = max(0.0, 1.0 - dist)
        output.append({
            "content": doc,
            "score": round(score, 4),
            "metadata": meta,
        })

    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]


if __name__ == "__main__":
    # Test
    results = semantic_search("what is the tuition fee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['metadata'].get('source')} | {r['content'][:80]}...")
