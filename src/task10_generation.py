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

# Model ID khác nhau tùy provider: OpenRouter cần prefix "openai/", gọi thẳng
# OpenAI thì không — xem _get_llm_client_and_model().
LLM_MODEL_OPENROUTER = "openai/gpt-4o-mini"  # hoặc model ":free" nếu chưa có credit
LLM_MODEL_OPENAI = "gpt-4o-mini"


def _is_real_key(value: str) -> bool:
    """Loại placeholder kiểu 'sk-or-v1-...' còn sót lại từ .env.example."""
    return bool(value) and "..." not in value


def _get_llm_client_and_model():
    """
    Chọn client + model theo key nào THẬT SỰ có trong .env.

    Quan trọng: OPENAI_API_KEY (key gốc OpenAI) không xác thực được trên
    base_url của OpenRouter và ngược lại — không thể vô tư lấy key nào có
    rồi gán cứng base_url OpenRouter như gợi ý cũ, phải chọn base_url khớp
    với key đang dùng.
    """
    from openai import OpenAI

    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    if _is_real_key(openrouter_key):
        return (
            OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1"),
            LLM_MODEL_OPENROUTER,
        )
    if _is_real_key(openai_key):
        return OpenAI(api_key=openai_key), LLM_MODEL_OPENAI

    raise RuntimeError(
        "Thiếu OPENROUTER_API_KEY hoặc OPENAI_API_KEY hợp lệ trong .env — "
        "xem .env.example."
    )


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

# Lưu ý (QA rà soát Task 10): ví dụ trích dẫn PHẢI khớp với field thật trong
# context mà format_context() tạo ra — mỗi block context có dòng
# "[Document i | Source: <filename> | Type: ...]". Bản prompt cũ chỉ cho ví dụ
# "[Tuition Fees, 2026]" — không phải filename thật và không có field "năm"
# nào được cấp cho model, nên model tự bịa mỗi lần một kiểu ([Document N],
# tên rút gọn, năm đoán mò). Cố định về đúng 1 field luôn có sẵn và
# deterministic: filename trong "Source:".
SYSTEM_PROMPT = """Bạn là trợ lý trả lời câu hỏi về dịch vụ và chính sách đại học
(học phí, học bổng, ký túc xá, thư viện, đăng ký học phần).

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin từ context được cung cấp — KHÔNG bịa đặt
2. MỖI khẳng định/số liệu phải có trích dẫn NGAY SAU khẳng định đó — không
   được gộp nhiều khẳng định lại rồi trích dẫn một lần ở cuối đoạn
3. Trích dẫn LUÔN theo định dạng [tên-file-nguồn], lấy CHÍNH XÁC giá trị sau
   "Source:" của block context tương ứng (ví dụ [tuition-fees-rmit.pdf]).
   TUYỆT ĐỐI KHÔNG dùng "[Document 1]", "[Document N]" hay bất kỳ số thứ tự
   nào; cũng KHÔNG tự đặt tên gọn/năm nếu không có sẵn trong context.
4. Nếu context không đủ thông tin → trả lời: "Tôi không thể xác minh thông tin này từ nguồn hiện có"
5. Trả lời bằng tiếng Việt, có cấu trúc rõ ràng theo đoạn văn
6. Không suy luận hay mở rộng ngoài những gì được nêu trong context

Ví dụ ĐÚNG (mỗi câu một trích dẫn riêng, dùng đúng tên file):
"Học phí ngành Business là 375.840.000 VND mỗi năm [tuition-fees-rmit.pdf]. Sinh viên có thể được giảm 25-50% học phí qua học bổng Academic Achievement [study-at-rmit-scholarships.pdf]. Thư viện RMIT Saigon South mở cửa từ 8h đến 20h các ngày trong tuần [library-hours-locations-rmit.pdf]."

Ví dụ SAI (không được làm theo):
"Học phí là 375.840.000 VND, được giảm 25-50% qua học bổng, và thư viện mở cửa 8h-20h [Document 1]." — gộp nhiều khẳng định vào một trích dẫn duy nhất, và dùng "Document 1" thay vì tên file thật."""


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
    front = chunks[::2]   # index 0, 2, 4 -> đặt ở đầu
    back = chunks[1::2]   # index 1, 3    -> đặt ở cuối (reversed)
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
        source = chunk.get("metadata", {}).get("source", f"Source {i}")
        doc_type = chunk.get("metadata", {}).get("type", "unknown")
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk['content']}\n"
        )
    return "\n---\n".join(context_parts)


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
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
    # Step 1: Retrieve
    chunks = retrieve(query, top_k=top_k)

    # Step 2: Reorder (tránh lost in the middle)
    reordered = reorder_for_llm(chunks)

    # Step 3: Format context
    context = format_context(reordered)

    # Step 4: Build prompt
    user_message = f"""Context:\n{context}\n\n---\n\nQuestion: {query}"""

    # Step 5: Call LLM
    client, model = _get_llm_client_and_model()

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

    # Step 6: Return
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
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
