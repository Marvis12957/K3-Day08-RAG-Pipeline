"""Task 5 - semantic search over the ChromaDB index."""


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    from .task4_chunking_indexing import get_collection, get_embedding_model

    if top_k <= 0:
        return []

    query_vector = get_embedding_model().encode(query).tolist()
    results = get_collection().query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for doc, meta, dist in zip(
        results.get("documents", [[]])[0],
        results.get("metadatas", [[]])[0],
        results.get("distances", [[]])[0],
    ):
        output.append({
            "content": doc,
            "score": max(0.0, 1.0 - float(dist)),
            "metadata": meta or {},
        })
    return sorted(output, key=lambda x: x["score"], reverse=True)[:top_k]


if __name__ == "__main__":
    for result in semantic_search("what is the tuition fee", top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
