"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"

Gợi ý LLM: OpenRouter có nhiều model gắn hậu tố ":free" không tính phí — xem
https://openrouter.ai/models?max_price=0 — phù hợp nếu chưa có credit trả phí.
Base URL: "https://openrouter.ai/api/v1", dùng chung interface với OpenAI SDK.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3

# Nhóm hiện có OPENAI_API_KEY nên gọi thẳng OpenAI. Nếu chỉ có
# OPENROUTER_API_KEY thì tự chuyển sang OpenRouter (cùng interface OpenAI SDK,
# chỉ khác base_url và tên model có tiền tố provider).
LLM_MODEL_OPENAI = "gpt-4o-mini"
LLM_MODEL_OPENROUTER = "openai/gpt-4o-mini"  # hoặc model ":free" nếu chưa có credit
LLM_MODEL = LLM_MODEL_OPENAI


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Bạn là trợ lý trả lời câu hỏi về dịch vụ và chính sách đại học
(học phí, học bổng, ký túc xá, thư viện, đăng ký học phần).

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin từ context được cung cấp — KHÔNG bịa đặt
2. Mỗi khẳng định phải có trích dẫn ngay sau, ví dụ: [Tuition Fees, 2026]
3. Nếu context không đủ thông tin → trả lời: "Tôi không thể xác minh thông tin này từ nguồn hiện có"
4. Trả lời bằng tiếng Việt, có cấu trúc rõ ràng theo đoạn văn
5. Không suy luận hay mở rộng ngoài những gì được nêu trong context"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if len(chunks) <= 2:
        return list(chunks)

    # chunks[::2] lấy hạng 1,3,5... đặt lên đầu; chunks[1::2] lấy hạng 2,4...
    # đảo ngược rồi ghép vào cuối => hạng 2 nằm ở vị trí CUỐI CÙNG, nơi LLM
    # chú ý thứ nhì sau vị trí đầu. Hạng thấp nhất bị đẩy vào giữa.
    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata") or {}
        source = meta.get("source", f"Source {i}")
        doc_type = meta.get("type", "unknown")
        # Tên file phải nằm nguyên văn trong context: LLM chỉ cite lại được
        # những gì nó nhìn thấy, và người đọc cần chuỗi này để truy ngược
        # về data/standardized/ mà đối chiếu.
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk['content']}\n"
        )
    return "\n---\n".join(context_parts)


# =============================================================================
# GENERATION
# =============================================================================

def _build_client():
    """
    Tạo OpenAI client. Ưu tiên OPENAI_API_KEY, không có thì thử OpenRouter.

    Returns:
        (client, model_name)
    """
    from openai import OpenAI

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key and not openai_key.endswith("..."):
        return OpenAI(api_key=openai_key), LLM_MODEL_OPENAI

    router_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if router_key and not router_key.endswith("..."):
        return (
            OpenAI(api_key=router_key, base_url="https://openrouter.ai/api/v1"),
            LLM_MODEL_OPENROUTER,
        )

    raise RuntimeError(
        "Chưa có API key. Đặt OPENAI_API_KEY hoặc OPENROUTER_API_KEY trong .env"
    )


def _resolve_followup(query: str, history: list[dict]) -> str:
    """
    Viết lại câu hỏi nối tiếp thành câu độc lập, dùng cho retrieval.

    Vì sao cần: người dùng hỏi "Học phí bao nhiêu?" rồi hỏi tiếp "còn học bổng
    thì sao?". Đưa nguyên chuỗi "còn học bổng thì sao?" vào semantic_search thì
    embedding của nó gần như vô nghĩa — mất hết chủ ngữ. Phải ghép ngữ cảnh
    TRƯỚC khi retrieve, chứ không phải chỉ nhét history vào prompt của LLM.

    Args:
        query: Câu hỏi hiện tại
        history: [{'role': 'user'|'assistant', 'content': str}, ...]

    Returns:
        Câu hỏi đã tự chứa ngữ cảnh (hoặc giữ nguyên nếu không có lịch sử).
    """
    prev_users = [m["content"] for m in history if m.get("role") == "user"]
    if not prev_users:
        return query

    try:
        client, model = _build_client()
        # Chỉ lấy 3 lượt gần nhất: xa hơn thường là chủ đề khác, đưa vào chỉ
        # làm nhiễu câu viết lại.
        recent = "\n".join(f"- {q}" for q in prev_users[-3:])
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Viết lại câu hỏi cuối thành một câu độc lập, tự chứa đủ ngữ "
                        "cảnh để tìm kiếm tài liệu. Giữ nguyên ngôn ngữ gốc. "
                        "Nếu câu hỏi đã đầy đủ, trả lại y nguyên. "
                        "CHỈ trả về câu hỏi, không giải thích."
                    ),
                },
                {"role": "user", "content": f"Các câu hỏi trước:\n{recent}\n\nCâu hỏi cuối: {query}"},
            ],
            temperature=0.0,  # Viết lại phải ổn định, không sáng tạo
        )
        rewritten = (resp.choices[0].message.content or "").strip()
        return rewritten or query
    except Exception:
        # Lỗi mạng/quota thì dùng câu gốc, đừng để chatbot chết vì bước phụ này.
        return query


def generate_with_citation(
    query: str, top_k: int = TOP_K, history: list[dict] | None = None
) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # Câu hỏi nối tiếp phải được ghép ngữ cảnh TRƯỚC khi retrieve.
    search_query = _resolve_followup(query, history or [])
    chunks = retrieve(search_query, top_k=top_k)

    # Task 9 trả rỗng khi cả hybrid lẫn PageIndex đều dưới ngưỡng. Không gọi
    # LLM với context rỗng: nó sẽ trả lời bằng kiến thức có sẵn, tức là bịa
    # đúng thứ RAG sinh ra để tránh.
    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    user_message = f"""Context:\n{context}\n\n---\n\nQuestion: {query}"""

    client, model = _build_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    answer = response.choices[0].message.content

    return {
        "answer": answer,
        # Trả CHUNKS GỐC theo thứ tự điểm, không phải bản đã reorder: reorder
        # chỉ phục vụ sự chú ý của LLM, còn người đọc cần thấy nguồn xếp theo
        # độ liên quan.
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid"),
        # Trả về để UI hiển thị được câu đã viết lại — người dùng nhìn thấy hệ
        # thống hiểu câu nối tiếp của mình thành gì.
        "search_query": search_query,
    }


if __name__ == "__main__":
    test_queries = [
        "Học phí tại RMIT Vietnam là bao nhiêu?",
        "Làm sao để đặt phòng học nhóm ở thư viện?",
        "Sinh viên quốc tế có những học bổng nào?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
