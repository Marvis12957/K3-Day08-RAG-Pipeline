"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (ChromaDB khuyến cáo — đơn giản, local, không cần Docker)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho cả tiếng Việt lẫn tiếng Anh)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - ChromaDB (khuyến cáo: đơn giản, local persistent, không cần Docker)
    - Weaviate (hỗ trợ hybrid search built-in, cần Docker/Cloud)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers chromadb

Lưu ý quan trọng: nếu sau này đổi corpus (đổi chủ đề, thêm/bớt tài liệu), phải XÓA
chroma_db/ cũ trước khi reindex — nếu không, chunk cũ và mới sẽ tồn tại lẫn lộn
trong cùng collection, retrieval sẽ trả về kết quả rác từ dữ liệu cũ.
"""

from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# Chunking: RecursiveCharacterTextSplitter.
#   Corpus gồm 2 loại rất khác nhau — PDF chính sách (đã qua MarkItDown nên
#   heading không còn đáng tin) và bài crawl (heading ## rõ ràng). Splitter
#   theo heading sẽ hỏng với nhóm thứ nhất, nên chọn recursive: cắt theo
#   đoạn văn trước, xuống dòng, rồi mới tới câu — an toàn cho cả hai.
CHUNK_SIZE = 800        # Đủ trọn một mục chính sách (học phí/điều kiện học bổng)
                        # mà không nuốt sang mục kế. File ngắn nhất trong corpus
                        # là 1.613 ký tự nên vẫn tách được ≥2 chunk.
CHUNK_OVERLAP = 100     # 12.5% — đủ để một câu bị cắt ngang vẫn còn nguyên vẹn
                        # ở chunk kế tiếp, không phình số chunk quá nhiều.
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# Embedding: BAAI/bge-m3.
#   Corpus là tiếng Anh (trang RMIT) nhưng câu hỏi người dùng là tiếng Việt —
#   bắt buộc phải dùng model multilingual, cùng không gian vector cho cả hai
#   ngôn ngữ. all-MiniLM-L6-v2 nhẹ hơn nhiều nhưng chỉ mạnh tiếng Anh, hỏi
#   tiếng Việt sẽ không khớp được với chunk tiếng Anh.
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# Vector store: ChromaDB — local persistent, không cần Docker, đủ cho corpus
# cỡ này và hỗ trợ sẵn cosine similarity cần cho Task 5.
VECTOR_STORE = "chromadb"  # "chromadb" | "weaviate" | "faiss"
COLLECTION_NAME = "university_services_docs"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        if not content.strip():
            continue
        # type lấy theo thư mục cha (legal/ hoặc news/) chứ không dò tên file,
        # vì tên file không chứa thông tin này.
        doc_type = md_file.parent.name
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type},
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Ưu tiên cắt ở ranh giới ngữ nghĩa lớn trước: hết đoạn -> hết dòng ->
        # hết câu -> hết từ. "" là phương án cuối, cắt giữa từ.
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        for i, chunk_text in enumerate(splitter.split_text(doc["content"])):
            if not chunk_text.strip():
                continue
            chunks.append({
                "content": chunk_text,
                # Giữ nguyên source + type: Task 10 cần chúng để sinh citation.
                # Thiếu metadata ở bước này là phải index lại toàn bộ.
                "metadata": {**doc["metadata"], "chunk_index": i},
            })
    return chunks


_embedding_model = None


def get_embedding_model():
    """
    Trả về SentenceTransformer đã load (singleton).

    Task 5 import hàm này để embed query bằng ĐÚNG model đã dùng lúc index —
    khác model là vector nằm ở không gian khác, cosine vô nghĩa.
    Cache lại vì bge-m3 nặng ~2GB, load nhiều lần rất chậm.
    """
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def get_collection():
    """
    Trả về Chroma collection đang dùng (Task 5 import hàm này).

    metadata={"hnsw:space": "cosine"} phải khớp với lúc tạo, nếu không Chroma
    dùng L2 mặc định và công thức score = 1 - distance ở Task 5 sẽ sai thang.
    """
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    model = get_embedding_model()
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=8)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    collection = get_collection()

    # id phải duy nhất theo (file, vị trí chunk). Dùng upsert nên chạy lại
    # trên cùng corpus sẽ ghi đè chứ không nhân đôi dữ liệu.
    ids = [
        f"{c['metadata']['source']}::chunk_{c['metadata']['chunk_index']}"
        for c in chunks
    ]
    collection.upsert(
        ids=ids,
        documents=[c["content"] for c in chunks],
        embeddings=[c["embedding"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )
    return collection.count()


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
