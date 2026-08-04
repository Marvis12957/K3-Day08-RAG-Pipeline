"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results

⚠️ BẪY THƯỜNG GẶP — đọc kỹ trước khi code:
    Nếu bạn dùng điểm RRF đã fuse (Task 7) để so với score_threshold, bạn sẽ gặp bug
    thật: RRF max score luôn ≈ 1/(k+1) ≈ 0.0164 (k=60) BẤT KỂ nội dung có liên quan
    hay không. Nếu đặt threshold thấp (như 0.005) để "hợp" với thang điểm RRF, thực
    chất KHÔNG câu hỏi nào đủ thấp để trigger fallback nữa — kể cả query hoàn toàn vô
    nghĩa vẫn trả về kết quả "hybrid" (rác) thay vì fallback đúng như thiết kế.

    Cách sửa đúng: giữ điểm cosine similarity GỐC của semantic_search (trước khi qua
    RRF) làm căn cứ quyết định fallback, tách biệt khỏi điểm RRF dùng để sắp xếp kết
    quả cuối cùng. Calibrate threshold bằng cách tự đo: chạy vài câu hỏi chắc chắn
    liên quan và vài câu chắc chắn lạc đề/rác qua semantic_search, xem khoảng cách
    điểm số giữa hai nhóm rồi chọn ngưỡng nằm giữa.
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

# Ngưỡng ĐO ĐƯỢC trên corpus này, không phải copy giá trị mẫu.
# Chạy semantic_search với 6 câu đúng chủ đề và 5 câu lạc đề trên index
# 24 documents / 180 chunks (bge-m3):
#       đúng chủ đề : thấp nhất 0.5936   (cao nhất 0.6850)
#       lạc đề      : cao nhất  0.4518   (thấp nhất 0.3309)
#       khoảng trống: 0.1418  -> điểm giữa 0.5227
# Chọn 0.52 = điểm giữa, cách đều cả hai nhóm ~0.07.
# LAB_GUIDE đề xuất 0.48 nhưng con số đó chỉ cách nhóm lạc đề 0.026 — quá sát,
# một câu lạc đề hơi may mắn là lọt qua mà không kích hoạt fallback.
SCORE_THRESHOLD = 0.52  # Nếu best score (cosine gốc) < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "rrf"  # "cross_encoder" | "mmr" | "rrf"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → dense_results (giữ điểm cosine gốc)
          ├→ Lexical Search  → sparse_results
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If dense_results[0]["score"] < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm cosine gốc tối thiểu (KHÔNG phải điểm RRF)
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # Lấy dư gấp đôi ở mỗi nhánh: RRF cần đủ chiều sâu để một chunk bị ranker
    # này xếp thấp vẫn có cơ hội được ranker kia kéo lên.
    dense_results = semantic_search(query, top_k=top_k * 2)
    sparse_results = lexical_search(query, top_k=top_k * 2)

    # QUYẾT ĐỊNH FALLBACK PHẢI DÙNG ĐIỂM COSINE GỐC, TRƯỚC KHI QUA RRF.
    # Điểm RRF sau khi fuse luôn xấp xỉ 1/(60+1) = 0.0164 cho top-1 bất kể nội
    # dung có liên quan hay không — so nó với threshold thì fallback không bao
    # giờ kích hoạt được, kể cả với query hoàn toàn vô nghĩa.
    best_dense_score = dense_results[0]["score"] if dense_results else 0.0

    if best_dense_score < score_threshold:
        try:
            fallback = pageindex_search(query, top_k=top_k)
        except Exception:
            fallback = []  # Hết quota/mất mạng thì không được làm sập pipeline
        if fallback:
            return fallback[:top_k]

        # Cả hybrid lẫn PageIndex đều không đạt ngưỡng -> TRẢ RỖNG.
        # Đừng rơi xuống nhánh hybrid: với query lạc đề, chunk điểm cao nhất là
        # rác (đo thật: "xyzabc123nonsense" trả về dòng "Copyright © 2026 RMIT
        # University, ABN 49 781 030 034"). Đưa rác đó vào prompt là mời LLM bịa
        # ra câu trả lời nghe có nguồn. Trả rỗng để Task 10 nói thẳng
        # "không xác minh được thông tin này".
        return []

    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)
    for item in merged:
        item["source"] = "hybrid"

    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        for item in final_results:
            item.setdefault("source", "hybrid")
    else:
        final_results = merged[:top_k]

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "What is the tuition fee at RMIT Vietnam?",
        "How do I book a library study room?",
        "What scholarships are available for international students?",
        "xyzabc123nonsense",  # Query không có kết quả → test fallback
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
