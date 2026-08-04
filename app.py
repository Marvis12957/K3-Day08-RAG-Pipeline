"""
RAG Chatbot — University Services (Starter Template)
Streamlit app kết nối RAG Retrieval (Task 9) và Generation (Task 10).

Chạy:
    streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Thêm project root vào sys.path để import các task từ src/
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="University Services RAG Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# THEME / CUSTOM CSS  (RMIT-red academic theme, light + dark aware)
# =============================================================================

st.markdown(
    """
    <style>
    :root {
        --brand: #C8102E;
        --brand-dark: #8C0C20;
        --card-border: #e8e8ec;
        --text-muted: #6b7280;
        --badge-high-bg: rgba(46, 160, 67, 0.14);
        --badge-high-fg: #1e7e34;
        --badge-med-bg: rgba(219, 154, 4, 0.16);
        --badge-med-fg: #a15c00;
        --badge-low-bg: rgba(207, 34, 46, 0.14);
        --badge-low-fg: #c0392b;
        --badge-type-bg: rgba(107, 114, 128, 0.14);
        --badge-type-fg: #4b5563;
    }
    @media (prefers-color-scheme: dark) {
        :root {
            --card-border: #333844;
            --text-muted: #9aa0ab;
            --badge-high-bg: rgba(63, 185, 80, 0.18);
            --badge-high-fg: #56d364;
            --badge-med-bg: rgba(219, 154, 4, 0.22);
            --badge-med-fg: #e3b341;
            --badge-low-bg: rgba(248, 81, 73, 0.2);
            --badge-low-fg: #f85149;
            --badge-type-bg: rgba(155, 162, 173, 0.2);
            --badge-type-fg: #c3c9d1;
        }
    }

    .hero-banner {
        background: linear-gradient(135deg, var(--brand) 0%, var(--brand-dark) 100%);
        color: #ffffff;
        padding: 1.6rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.2rem;
        box-shadow: 0 6px 18px rgba(140, 12, 32, 0.25);
    }
    .hero-banner h1 {
        margin: 0 0 0.3rem 0;
        font-size: 1.7rem;
        color: #ffffff;
    }
    .hero-banner p {
        margin: 0 0 0.9rem 0;
        opacity: 0.92;
        font-size: 0.95rem;
    }
    .pipeline-steps {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
    }
    .pipeline-steps span {
        background: rgba(255, 255, 255, 0.16);
        border: 1px solid rgba(255, 255, 255, 0.35);
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        font-size: 0.78rem;
        white-space: nowrap;
    }

    .badge {
        display: inline-block;
        padding: 0.12rem 0.6rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-right: 0.35rem;
    }
    .badge-high { background: var(--badge-high-bg); color: var(--badge-high-fg); }
    .badge-medium { background: var(--badge-med-bg); color: var(--badge-med-fg); }
    .badge-low { background: var(--badge-low-bg); color: var(--badge-low-fg); }
    .badge-type { background: var(--badge-type-bg); color: var(--badge-type-fg); }

    .source-name { font-weight: 600; }
    .source-preview {
        color: var(--text-muted);
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }

    [data-testid="stSidebar"] button {
        text-align: left !important;
        border-radius: 10px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# SHARED DATA — suggested questions & source rendering
# =============================================================================

SUGGESTED_QUESTIONS = [
    "Học phí tại RMIT Vietnam là bao nhiêu?",
    "Làm sao để đặt phòng học nhóm ở thư viện?",
    "Điều kiện xin học bổng Academic Achievement?",
    "Dịch vụ hỗ trợ chỗ ở cho sinh viên như thế nào?",
    "Cách đăng ký học phần qua myRMIT?",
]

FALLBACK_THRESHOLD = 0.48  # đồng bộ với ngưỡng fallback Cosine ở Task 9


def confidence_badge(score: float) -> str:
    """Trả về badge HTML thể hiện độ tin cậy của 1 chunk, dựa trên điểm Cosine."""
    if score >= 0.70:
        return f'<span class="badge badge-high">🟢 high · {score:.3f}</span>'
    if score >= FALLBACK_THRESHOLD:
        return f'<span class="badge badge-medium">🟡 medium · {score:.3f}</span>'
    return f'<span class="badge badge-low">🔴 low · {score:.3f}</span>'


def render_sources(sources: list) -> None:
    """Hiển thị danh sách nguồn tham khảo dưới dạng card, kèm badge độ tin cậy."""
    with st.expander(f"📚 Nguồn tham khảo ({len(sources)} chunks)", expanded=False):
        for i, src in enumerate(sources, 1):
            meta = src.get("metadata", {})
            source_name = meta.get("source", "Unknown")
            doc_type = meta.get("type", "unknown")
            score = src.get("score", 0)
            with st.container(border=True):
                st.markdown(
                    f'<span class="source-name">[{i}] {source_name}</span> '
                    f'<span class="badge badge-type">{doc_type}</span> '
                    f'{confidence_badge(score)}',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="source-preview">{src.get("content", "")[:300]}...</div>',
                    unsafe_allow_html=True,
                )

# =============================================================================
# HERO HEADER
# =============================================================================

st.markdown(
    """
    <div class="hero-banner">
        <h1>🎓 University Services RAG Chatbot</h1>
        <p>Hỏi đáp về học phí, học bổng, ký túc xá, thư viện &amp; các dịch vụ đại học — trả lời kèm trích dẫn nguồn.</p>
        <div class="pipeline-steps">
            <span>🔎 Semantic Search</span>
            <span>🔤 BM25</span>
            <span>🔀 RRF Rerank</span>
            <span>🗂️ PageIndex Fallback</span>
            <span>✍️ LLM Generation + Citation</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# SIDEBAR — INFO & SETTINGS
# =============================================================================

with st.sidebar:
    st.markdown("### 🎓 University Services RAG")
    st.caption("Trợ lý hỏi đáp về dịch vụ và chính sách đại học")

    st.divider()

    st.markdown("**💡 Câu hỏi gợi ý**")
    for s in SUGGESTED_QUESTIONS:
        if st.button(s, use_container_width=True, key=f"sug_{s[:20]}"):
            st.session_state["pending_query"] = s

    st.divider()
    st.markdown("**⚙️ Thiết lập**")
    top_k = st.slider("Số chunks retrieval (top_k)", 3, 10, 5)

    if st.button("🗑️ Xoá lịch sử chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("**🧭 Kiến trúc hệ thống**")
    st.caption("Hybrid Retrieval (Semantic + BM25) → RRF Rerank → PageIndex Fallback (nếu Cosine < 0.48) → LLM Generation có Citation")

# =============================================================================
# SESSION STATE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# =============================================================================
# MAIN CHAT AREA
# =============================================================================

if not st.session_state.messages:
    with st.container(border=True):
        st.markdown("#### 👋 Bắt đầu bằng một câu hỏi")
        st.caption("Chọn nhanh một gợi ý bên dưới, hoặc gõ câu hỏi của riêng bạn ở khung chat phía dưới.")
        cols = st.columns(2)
        for i, s in enumerate(SUGGESTED_QUESTIONS):
            if cols[i % 2].button(s, use_container_width=True, key=f"main_sug_{i}"):
                st.session_state["pending_query"] = s

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    avatar = "🧑‍🎓" if msg["role"] == "user" else "🎓"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            render_sources(msg["sources"])

# =============================================================================
# QUERY HANDLING
# =============================================================================

# Xử lý khi bấm nút gợi ý hoặc nhập câu hỏi mới
user_input = st.chat_input("Nhập câu hỏi của bạn về chính sách/dịch vụ đại học...")
query = user_input or st.session_state.pending_query

if query:
    st.session_state.pending_query = None

    # Hiển thị câu hỏi của user
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(query)

    # Sinh câu trả lời từ RAG Pipeline
    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("Đang tìm kiếm tài liệu và tổng hợp câu trả lời..."):
            try:
                # TODO (Học viên): Tích hợp hàm sinh câu trả lời từ Task 10
                from src.task10_generation import generate_with_citation

                response = generate_with_citation(query, top_k=top_k)
                answer = response.get("answer", "Chưa thể trả lời.")
                sources = response.get("sources", [])

            except NotImplementedError:
                answer = "⚠️ **Task 10 chưa được implement.** Hãy hoàn thành `src/task10_generation.py` để kết nối pipeline vào UI!"
                sources = []
            except Exception as e:
                answer = f"❌ **Lỗi khi chạy RAG Pipeline:** {e}"
                sources = []

            st.markdown(answer)

            if sources:
                render_sources(sources)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
