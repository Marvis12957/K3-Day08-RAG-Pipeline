"""Task 9 - retrieval pipeline with cosine-threshold fallback."""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search

SCORE_THRESHOLD = 0.52
DEFAULT_TOP_K = 5
RERANK_METHOD = "rrf"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    dense_results = semantic_search(query, top_k=top_k * 2)
    sparse_results = lexical_search(query, top_k=top_k * 2)

    best_cosine = dense_results[0]["score"] if dense_results else 0.0
    if best_cosine < score_threshold:
        try:
            return pageindex_search(query, top_k=top_k)[:top_k]
        except Exception:
            return []

    if use_reranking and RERANK_METHOD == "rrf":
        final_results = rerank_rrf([dense_results, sparse_results], top_k=top_k)
    elif use_reranking:
        final_results = rerank(query, dense_results + sparse_results, top_k=top_k, method=RERANK_METHOD)
    else:
        final_results = (dense_results + sparse_results)[:top_k]

    for item in final_results:
        item["source"] = "hybrid"
    return final_results[:top_k]


if __name__ == "__main__":
    for question in [
        "What is the tuition fee at RMIT Vietnam?",
        "xyzabc123nonsense",
    ]:
        print(f"\nQuery: {question}")
        for result in retrieve(question, top_k=3):
            print(f"[{result['score']:.3f}] [{result['source']}] {result['content'][:80]}...")
