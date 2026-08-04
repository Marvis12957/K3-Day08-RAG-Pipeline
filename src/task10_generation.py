"""Task 10 - generation helpers with citations."""

from .task9_retrieval_pipeline import retrieve

TOP_K = 5


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]


def format_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", f"Source {i}")
        doc_type = metadata.get("type", "unknown")
        parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk.get('content', '')}"
        )
    return "\n---\n".join(parts)


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    chunks = retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)

    if not reordered:
        answer = "Toi khong the xac minh thong tin nay tu nguon hien co."
    else:
        first = reordered[0]
        source = first.get("metadata", {}).get("source", "unknown-source")
        snippet = " ".join(first.get("content", "").split())[:500]
        answer = f"{snippet} [{source}, 2026]"

    return {
        "answer": answer,
        "sources": reordered,
        "retrieval_source": reordered[0].get("source", "none") if reordered else "none",
        "context": format_context(reordered),
    }


if __name__ == "__main__":
    result = generate_with_citation("What is the tuition fee at RMIT Vietnam?")
    print(result["answer"])
