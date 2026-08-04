"""Task 6 - lexical BM25 search over markdown documents."""

import math
from collections import Counter
from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CORPUS: list[dict] = []
BM25_INDEX = None


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _load_corpus() -> list[dict]:
    if CORPUS:
        return CORPUS

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if content:
            CORPUS.append({
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "path": md_file.relative_to(STANDARDIZED_DIR).as_posix(),
                    "type": md_file.parent.name,
                },
            })
    return CORPUS


def build_bm25_index(corpus: list[dict]):
    tokenized = [_tokenize(doc["content"]) for doc in corpus]
    doc_freq = Counter(term for tokens in tokenized for term in set(tokens))
    avgdl = sum(len(tokens) for tokens in tokenized) / max(len(tokenized), 1)
    return {"tokenized": tokenized, "doc_freq": doc_freq, "avgdl": avgdl, "n": len(tokenized)}


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    global BM25_INDEX

    if top_k <= 0:
        return []

    corpus = _load_corpus()
    if BM25_INDEX is None:
        BM25_INDEX = build_bm25_index(corpus)

    k1, b = 1.5, 0.75
    query_terms = _tokenize(query)
    results = []

    for idx, tokens in enumerate(BM25_INDEX["tokenized"]):
        counts = Counter(tokens)
        doc_len = len(tokens) or 1
        score = 0.0
        for term in query_terms:
            df = BM25_INDEX["doc_freq"].get(term, 0)
            if not df:
                continue
            idf = math.log(1 + (BM25_INDEX["n"] - df + 0.5) / (df + 0.5))
            tf = counts[term]
            denom = tf + k1 * (1 - b + b * doc_len / BM25_INDEX["avgdl"])
            score += idf * (tf * (k1 + 1)) / denom
        if score > 0:
            results.append({
                "content": corpus[idx]["content"],
                "score": float(score),
                "metadata": corpus[idx]["metadata"],
            })

    return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]


if __name__ == "__main__":
    for result in lexical_search("tuition fee payment methods", top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
