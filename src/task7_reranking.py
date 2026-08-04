"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement — khuyến nghị vì không cần API key

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.

Lưu ý quan trọng về RRF (sẽ dùng lại ở Task 9): điểm RRF fused CHỈ phụ thuộc thứ hạng,
không phải độ tương đồng thật. Top-1 sau khi fuse luôn xấp xỉ 1/(k+1) ≈ 0.0164 (k=60),
bất kể nội dung đó có thật sự liên quan đến câu hỏi hay không. Đừng dùng điểm RRF để
quyết định fallback ở Task 9 — xem ghi chú ở đó.

PHƯƠNG PHÁP CHỌN: RRF.
    Lý do: không cần API key (Jina/Qwen đều cần), và bài toán ở đây đúng là bài toán
    RRF sinh ra để giải — gộp 2 ranker có thang điểm KHÔNG so sánh được với nhau.
    Cosine của Task 5 nằm trong [0,1]; BM25 của Task 6 là số dương không chặn trên
    (đo trên corpus này: 3.6 - 13.8). Cộng hay lấy trung bình hai thang đó đều vô
    nghĩa. RRF bỏ qua giá trị điểm, chỉ dùng THỨ HẠNG nên miễn nhiễm với việc hai
    ranker có thang khác nhau.
"""

import math
from typing import Optional


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    import os

    import requests

    api_key = os.getenv("JINA_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "Cross-encoder cần JINA_API_KEY trong .env. "
            "Chưa có key thì dùng method='rrf' (không cần API)."
        )

    response = requests.post(
        "https://api.jina.ai/v1/rerank",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "documents": [c["content"] for c in candidates],
            "top_n": top_k,
        },
        timeout=30,
    )
    response.raise_for_status()
    return [
        {**candidates[r["index"]], "score": r["relevance_score"]}
        for r in response.json()["results"]
    ]


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity giữa 2 vector, không cần numpy."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    selected: list[int] = []
    remaining = list(range(len(candidates)))

    while remaining and len(selected) < top_k:
        best_idx, best_score = None, float("-inf")

        for idx in remaining:
            emb = candidates[idx].get("embedding")
            if emb is None:
                continue
            relevance = _cosine_sim(query_embedding, emb)

            # Phạt theo mức giống nhất với những chunk ĐÃ chọn — đây là chỗ
            # MMR khác rerank thường: chunk thứ 2 gần trùng chunk thứ 1 sẽ bị
            # đẩy xuống dù nó rất liên quan tới query.
            max_sim_selected = max(
                (_cosine_sim(emb, candidates[s]["embedding"]) for s in selected),
                default=0.0,
            )
            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim_selected

            if mmr > best_score:
                best_score, best_idx = mmr, idx

        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    k=60 (Cormack et al. 2009) làm hằng số giảm chấn: nó khiến chênh lệch giữa
    hạng 1 và hạng 2 (1/61 vs 1/62) nhỏ hơn nhiều so với không có k (1/1 vs 1/2).
    Nhờ vậy một ranker "chắc chắn sai nhưng tự tin" không thể áp đảo ranker kia —
    tài liệu được CẢ HAI ranker xếp hạng khá sẽ thắng tài liệu chỉ được MỘT ranker
    xếp hạng nhất. Đó chính là cân bằng mà hybrid search cần.

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            # Giữ bản đầu tiên gặp: các ranker trả về cùng chunk nên metadata
            # giống nhau, chỉ khác trường score gốc (cosine vs BM25).
            content_map.setdefault(key, item)

    ordered = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)

    results = []
    for content, score in ordered[:top_k]:
        item = dict(content_map[content])
        # Giữ lại điểm gốc để Task 9 còn dùng làm căn cứ fallback — điểm RRF
        # ghi đè lên 'score' KHÔNG phản ánh độ liên quan thật.
        item["original_score"] = item.get("score")
        item["score"] = score
        results.append(item)
    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval. Chấp nhận 2 dạng:
            - list[dict]        : một ranked list
            - list[list[dict]]  : nhiều ranked list (dùng cho RRF)
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if not candidates:
        return []

    # Cho phép truyền thẳng nhiều ranked list vào đây thay vì phải gọi
    # rerank_rrf() riêng — Task 9 gọi kiểu này.
    is_nested = isinstance(candidates[0], list)

    if method == "cross_encoder":
        flat = [item for lst in candidates for item in lst] if is_nested else candidates
        return rerank_cross_encoder(query, flat, top_k)

    if method == "mmr":
        flat = [item for lst in candidates for item in lst] if is_nested else candidates
        if not all("embedding" in c for c in flat):
            raise ValueError(
                "MMR cần mỗi candidate có key 'embedding'. "
                "Dùng rerank_mmr(query_embedding, ...) trực tiếp."
            )
        raise NotImplementedError("Call rerank_mmr with query_embedding")

    if method == "rrf":
        # Một list đơn vẫn gộp được: coi như hybrid chỉ có 1 ranker. Thứ tự giữ
        # nguyên nhưng score được chuẩn hoá về thang RRF, nhất quán với lúc có
        # đủ 2 ranker.
        ranked_lists = candidates if is_nested else [candidates]
        return rerank_rrf(ranked_lists, top_k=top_k)

    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Tuition fee payment schedule", "score": 0.8, "metadata": {}},
        {"content": "Scholarship eligibility requirements", "score": 0.6, "metadata": {}},
        {"content": "Library study room booking guide", "score": 0.5, "metadata": {}},
    ]
    results = rerank("tuition fee payment", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.4f}] {r['content']}")
