"""Task 4 - chunk markdown corpus and index it into ChromaDB."""

import shutil
from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

VECTOR_STORE = "chromadb"
COLLECTION_NAME = "university_services_docs"

_EMBEDDING_MODEL = None


def load_documents() -> list[dict]:
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "path": md_file.relative_to(STANDARDIZED_DIR).as_posix(),
                "type": md_file.parent.name,
            },
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []

    for doc in documents:
        for chunk_index, chunk_text in enumerate(splitter.split_text(doc["content"])):
            if chunk_text:
                chunks.append({
                    "content": chunk_text,
                    "metadata": {**doc["metadata"], "chunk_index": chunk_index},
                })

    return chunks


def get_embedding_model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL)
    return _EMBEDDING_MODEL


def embed_chunks(chunks: list[dict]) -> list[dict]:
    model = get_embedding_model()
    embeddings = model.encode([c["content"] for c in chunks], show_progress_bar=True)
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()
    return chunks


def get_collection():
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def index_to_vectorstore(chunks: list[dict]):
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    collection = get_collection()
    collection.upsert(
        ids=[f"{c['metadata']['path']}::chunk-{c['metadata']['chunk_index']}" for c in chunks],
        documents=[c["content"] for c in chunks],
        embeddings=[c["embedding"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )


def run_pipeline():
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
