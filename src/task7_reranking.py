"""Task 7 - reranking helpers."""


def rerank_cross_encoder(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    raise NotImplementedError("Implement rerank_cross_encoder")


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    raise NotImplementedError("Implement rerank_mmr")


def rerank_rrf(ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            key = item.get("content", "")
            if not key:
                continue
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in items:
                saved = item.copy()
                saved["original_score"] = item.get("score")
                items[key] = saved

    ordered = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    results = []
    for key, score in ordered[:top_k]:
        item = items[key].copy()
        item["score"] = score
        results.append(item)
    return results


def _as_ranked_lists(candidates: list[dict] | list[list[dict]]) -> list[list[dict]]:
    if not candidates:
        return []
    first = candidates[0]
    return candidates if isinstance(first, list) else [candidates]


def rerank(
    query: str,
    candidates: list[dict] | list[list[dict]],
    top_k: int = 5,
    method: str = "rrf",
) -> list[dict]:
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "mmr":
        raise NotImplementedError("Call rerank_mmr with query_embedding")
    if method == "rrf":
        return rerank_rrf(_as_ranked_lists(candidates), top_k=top_k)
    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy_candidates = [
        {"content": "Tuition fee payment schedule", "score": 0.8, "metadata": {}},
        {"content": "Scholarship eligibility requirements", "score": 0.6, "metadata": {}},
        {"content": "Library study room booking guide", "score": 0.5, "metadata": {}},
    ]
    for r in rerank("tuition fee payment", dummy_candidates, top_k=2):
        print(f"[{r['score']:.5f}] {r['content']}")
